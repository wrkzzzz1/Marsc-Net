# Filename: multiscale_contrastive_head.py
# Fine-Grained Memory Bank + Gated Negatives (Py3.8 / mmrotate 0.3.4 compatible)

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_HEADS


def _safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    norm = x.norm(p=2, dim=dim, keepdim=True)
    norm = norm.clamp_min(eps)
    return x / norm


class MLPProj(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim, feat_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@ROTATED_HEADS.register_module()
class MultiScaleContrastiveHead(nn.Module):
    def __init__(
        self,
        in_dim=192,
        feat_dim=512,
        num_scales=5,
        num_classes=31,
        temperature=0.1,
        learnable_temp=True,
        proto_momentum=0.99,
        lam_scale=0.003,
        lam_supcon=0.015,
        lam_proto=0.08,
        warmup_epochs=12,
        ramp_epochs=8,
        supcon_delay=3,
        ms_delay=3,
        # 保留旧参数以兼容老 cfg，但默认不用 queue
        use_queue=False,
        queue_size=65536,
        class_counts: Optional[List[int]] = None,
        hard_negatives_topk=256,
        weak_cls_boost=1.3,
        # ===== Fine-Grained Memory Bank =====
        use_finegrained_mb=True,
        mem_size_per_class=256,      # Q
        mem_select_q=8,              # q
        mem_iou_thr: Optional[float] = None,   # T1: 入库 IoU 阈值（可选）
        loss_iou_thr: Optional[float] = None,  # T2: 计算 loss IoU 阈值（可选）
        # ===== Confusing-class gated negatives =====
        use_gated_negatives=True,
        confusing_topr=3,            # R
        confusing_conf_thr=0.6,      # δ
        include_batch_keys=True,     # keys = batch + memory（更稳）
        **kwargs
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.feat_dim = int(feat_dim)
        self.num_scales = int(num_scales)
        self.num_classes = int(num_classes)

        self.init_temp = float(temperature)
        self.min_temp = 0.05
        self.learnable_temp = bool(learnable_temp)
        if self.learnable_temp:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))).reshape(1))
        else:
            self.register_buffer("temperature_buf", torch.tensor(float(temperature)))

        self.proto_momentum = float(proto_momentum)
        self.lam_scale = float(lam_scale)
        self.lam_supcon = float(lam_supcon)
        self.lam_proto = float(lam_proto)

        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.supcon_delay = int(supcon_delay)
        self.ms_delay = int(ms_delay)
        self._cur_epoch = 0

        self.weak_cls_boost = float(weak_cls_boost)

        self.proj = MLPProj(self.in_dim, self.feat_dim, hidden_dim=max(self.in_dim, self.feat_dim))

        # prototypes
        self.register_buffer("prototypes", torch.empty(0), persistent=True)
        self.register_buffer("protos_inited", torch.tensor(0, dtype=torch.uint8), persistent=False)

        # old queue (compat)
        self.use_queue = bool(use_queue)
        if self.use_queue:
            self.queue_size = int(queue_size)
            self.register_buffer("queue", torch.zeros(self.queue_size, self.feat_dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_labels", torch.full((self.queue_size,), -1, dtype=torch.long))
            self.register_buffer("queue_inited", torch.tensor(0, dtype=torch.uint8), persistent=False)
        self.hard_negatives_topk = int(hard_negatives_topk)

        # Fine-grained MB
        self.use_finegrained_mb = bool(use_finegrained_mb)
        self.mem_size = int(mem_size_per_class)
        self.mem_select_q = int(mem_select_q)
        self.mem_iou_thr = None if mem_iou_thr is None else float(mem_iou_thr)
        self.loss_iou_thr = None if loss_iou_thr is None else float(loss_iou_thr)

        self.use_gated_negatives = bool(use_gated_negatives)
        self.confusing_topr = int(confusing_topr)
        self.confusing_conf_thr = float(confusing_conf_thr)
        self.include_batch_keys = bool(include_batch_keys)

        if self.use_finegrained_mb:
            self.register_buffer("mem_bank", torch.zeros(self.num_classes, self.mem_size, self.feat_dim))
            self.register_buffer("mem_ptr", torch.zeros(self.num_classes, dtype=torch.long))
            self.register_buffer("mem_cnt", torch.zeros(self.num_classes, dtype=torch.long))

        # class frequency weights (rare class larger)
        if class_counts is not None:
            cc = torch.tensor(class_counts, dtype=torch.float32)
            w = 1.0 / torch.sqrt(cc.clamp_min(1.0))
            w = w * (w.numel() / w.sum())
            self.register_buffer("class_proto_weights", w)
        else:
            self.register_buffer("class_proto_weights", torch.ones(self.num_classes))

        self.init_weights()

    def set_epoch(self, epoch: int):
        self._cur_epoch = int(epoch)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # torch 1.8+ usually has trunc_normal_, otherwise fallback to normal_
                if hasattr(nn.init, "trunc_normal_"):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def _get_temperature(self, device: torch.device) -> torch.Tensor:
        if self.learnable_temp:
            epoch_factor = max(0.0, 1.0 - self._cur_epoch / float(self.warmup_epochs + self.ramp_epochs))
            sched_temp = self.min_temp + (self.init_temp - self.min_temp) * epoch_factor
            temp = self.log_temperature.exp() * (sched_temp / self.init_temp)
            return temp.clamp_min(1e-3).to(device)
        return self.temperature_buf.to(device)

    def _ensure_prototypes(self, device: torch.device):
        if (self.prototypes.numel() == 0) or (self.prototypes.size(0) != self.num_classes) or (
            self.prototypes.size(1) != self.feat_dim
        ):
            with torch.no_grad():
                p = torch.randn(self.num_classes, self.feat_dim, device=device)
                p = _safe_l2_normalize(p, dim=1)
                self.prototypes = p
                self.protos_inited.fill_(1)

    def _compute_stage_alphas(self):
        e = self._cur_epoch
        warm = self.warmup_epochs
        ramp = self.ramp_epochs

        if e < warm:
            alpha_proto = 0.0
        else:
            alpha_proto = min(1.0, (e - warm) / float(ramp))

        if e < warm + self.supcon_delay:
            alpha_sup = 0.0
        else:
            alpha_sup = min(1.0, (e - (warm + self.supcon_delay)) / float(ramp))

        if e < warm + self.supcon_delay + self.ms_delay:
            alpha_ms = 0.0
        else:
            alpha_ms = min(1.0, (e - (warm + self.supcon_delay + self.ms_delay)) / float(ramp))

        return alpha_proto, alpha_sup, alpha_ms

    @torch.no_grad()
    def _flatten_memory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns: (mem_feats [M,D], mem_labels [M])
        cnt = self.mem_cnt.clamp_min(0).clamp_max(self.mem_size)  # [C]
        device = self.mem_bank.device

        idx = torch.arange(self.mem_size, device=device).unsqueeze(0).expand(self.num_classes, -1)  # [C,Q]
        valid = idx < cnt.unsqueeze(1)  # [C,Q]
        if valid.sum().item() == 0:
            empty_feats = self.mem_bank.new_zeros((0, self.feat_dim))
            empty_labels = torch.empty((0,), dtype=torch.long, device=device)
            return empty_feats, empty_labels

        feats = self.mem_bank[valid]  # [M,D]
        labels = torch.arange(self.num_classes, device=device).unsqueeze(1).expand(-1, self.mem_size)[valid]  # [M]
        return feats, labels

    @torch.no_grad()
    def _update_memory_diverse(self, feats: torch.Tensor, labels: torch.Tensor, ious: Optional[torch.Tensor] = None):
        if not self.use_finegrained_mb:
            return
        if feats.numel() == 0:
            return

        device = self.mem_bank.device
        feats = feats.to(device)
        labels = labels.to(device)

        if ious is not None and self.mem_iou_thr is not None:
            ious = ious.to(device)
            keep = ious > float(self.mem_iou_thr)
            feats = feats[keep]
            labels = labels[keep]
            if feats.numel() == 0:
                return

        feats = _safe_l2_normalize(feats, dim=1)

        for k in range(self.num_classes):
            mask = labels == k
            if mask.sum().item() == 0:
                continue

            cand = feats[mask]  # [Nc,D]
            if cand.size(0) == 0:
                continue

            q = min(self.mem_select_q, cand.size(0))
            cnt_k = int(self.mem_cnt[k].item())

            if cnt_k > 0:
                mu = self.mem_bank[k, :cnt_k].mean(dim=0)
                mu = _safe_l2_normalize(mu, dim=0)
                sims = cand @ mu  # [Nc]
                sel_idx = torch.argsort(sims)[:q]  # 最不相似 q
                selected = cand[sel_idx]
            else:
                perm = torch.randperm(cand.size(0), device=device)
                selected = cand[perm[:q]]

            ptr = int(self.mem_ptr[k].item()) % self.mem_size
            for j in range(selected.size(0)):
                self.mem_bank[k, ptr].copy_(selected[j])
                ptr = (ptr + 1) % self.mem_size
                if cnt_k < self.mem_size:
                    cnt_k += 1

            self.mem_ptr[k] = torch.tensor(ptr, dtype=self.mem_ptr.dtype, device=device)
            self.mem_cnt[k] = torch.tensor(cnt_k, dtype=self.mem_cnt.dtype, device=device)

    def _build_sample_weights(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.numel() == 0:
            return labels.new_zeros((0,), dtype=torch.float32)

        if self._cur_epoch < 1:
            power = self.weak_cls_boost * 0.3
        elif self._cur_epoch < max(2, self.warmup_epochs // 2):
            power = self.weak_cls_boost * 0.6
        else:
            power = self.weak_cls_boost

        w = self.class_proto_weights.to(labels.device)[labels].float()
        return w.pow(power)

    def _supcon_loss_finegrained_mb(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        roi_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = feats.size(0)
        if N <= 1:
            return feats.new_zeros(())

        device = feats.device
        temp = self._get_temperature(device)

        feats = _safe_l2_normalize(feats, dim=1)

        mem_feats, mem_labels = self._flatten_memory()
        mem_feats = mem_feats.to(device)
        mem_labels = mem_labels.to(device)

        keys_feats = []
        keys_labels = []

        if self.include_batch_keys:
            keys_feats.append(feats.detach())
            keys_labels.append(labels)

        if mem_feats.numel() > 0:
            keys_feats.append(mem_feats.detach())
            keys_labels.append(mem_labels)

        if len(keys_feats) == 0:
            return feats.new_zeros(())

        keys_feats = torch.cat(keys_feats, dim=0)   # [K,D]
        keys_labels = torch.cat(keys_labels, dim=0) # [K]

        logits = (feats @ keys_feats.t()) / temp  # [N,K]
        batch_key_count = N if self.include_batch_keys else 0

        sample_weights = self._build_sample_weights(labels)

        losses = []
        weights = []

        for i in range(N):
            y = int(labels[i].item())

            same_cls = keys_labels == y
            not_self = torch.ones_like(same_cls, dtype=torch.bool, device=device)
            if batch_key_count > 0:
                not_self[i] = False

            pos_mask = same_cls & not_self

            valid_key_label = (keys_labels >= 0) & (keys_labels < self.num_classes)
            neg_mask = (~same_cls) & valid_key_label

            # confusing-class gated negatives
            if self.use_gated_negatives and (roi_probs is not None) and (self.confusing_topr > 0):
                p = roi_probs[i]
                if p is not None and p.numel() == self.num_classes:
                    if float(p[y].item()) >= self.confusing_conf_thr:
                        p2 = p.clone()
                        p2[y] = -1.0
                        topr = min(self.confusing_topr, self.num_classes - 1)
                        conf_classes = torch.topk(p2, k=topr, dim=0).indices
                        allowed = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
                        allowed[conf_classes] = True
                        neg_mask = neg_mask & allowed[keys_labels.clamp_min(0).clamp_max(self.num_classes - 1)]

            pos_logits = logits[i][pos_mask]
            neg_logits = logits[i][neg_mask]

            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue

            denom = torch.logsumexp(neg_logits, dim=0)
            loss_i = -(pos_logits - denom).mean()

            losses.append(loss_i)
            weights.append(sample_weights[i])

        if len(losses) == 0:
            return feats.new_zeros(())

        losses_t = torch.stack(losses, dim=0)
        weights_t = torch.stack(weights, dim=0).detach()
        return (losses_t * weights_t).mean()

    def _cosine_pair_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = _safe_l2_normalize(x, dim=1)
        y = _safe_l2_normalize(y, dim=1)
        return (1.0 - (x * y).sum(dim=1)).mean()

    @torch.no_grad()
    def _update_prototypes_ema(self, feats: torch.Tensor, labels: torch.Tensor):
        feats = _safe_l2_normalize(feats, dim=1)
        uniq = torch.unique(labels)

        rare_classes = torch.topk(
            self.class_proto_weights,
            k=max(1, int(self.num_classes * 0.3)),
            largest=True,
        ).indices
        rare_set = set(int(x) for x in rare_classes.tolist())

        for c in uniq:
            c_i = int(c.item())
            mask = labels == c
            if mask.sum().item() == 0:
                continue
            cur = feats[mask].mean(dim=0)
            old = self.prototypes[c_i]

            momentum_adj = 0.5 if c_i in rare_set else self.proto_momentum
            new = _safe_l2_normalize(momentum_adj * old + (1.0 - momentum_adj) * cur, dim=0)
            self.prototypes[c_i] = new

    # old queue enqueue (compat)
    @torch.no_grad()
    def _enqueue_dequeue(self, keys: torch.Tensor, labels: torch.Tensor):
        if not self.use_queue or keys.numel() == 0:
            return

        device_q = self.queue.device
        device_q_labels = self.queue_labels.device
        keys = keys.to(device_q)
        labels = labels.to(device_q_labels)

        B = keys.shape[0]
        Q = int(self.queue_size)

        if B >= Q:
            self.queue.copy_(keys[-Q:].detach().clone())
            self.queue_labels.copy_(labels[-Q:].detach().clone())
            self.queue_ptr[0] = torch.tensor(0, dtype=self.queue_ptr.dtype, device=self.queue_ptr.device)
            self.queue_inited.fill_(1)
            return

        ptr = int(self.queue_ptr[0].item()) % Q
        end = ptr + B

        if end <= Q:
            self.queue[ptr:end, :].copy_(keys.detach().clone())
            self.queue_labels[ptr:end].copy_(labels.detach().clone())
        else:
            first = Q - ptr
            rem = B - first
            if first > 0:
                self.queue[ptr:, :].copy_(keys[:first].detach().clone())
                self.queue_labels[ptr:].copy_(labels[:first].detach().clone())
            if rem > 0:
                self.queue[:rem, :].copy_(keys[first:].detach().clone())
                self.queue_labels[:rem].copy_(labels[first:].detach().clone())

        new_ptr = (ptr + B) % Q
        self.queue_ptr[0] = torch.tensor(new_ptr, dtype=self.queue_ptr.dtype, device=self.queue_ptr.device)
        self.queue_inited.fill_(1)

    def forward(
        self,
        roi_feats_per_scale: List[torch.Tensor],
        roi_labels: torch.Tensor,
        roi_probs: Optional[torch.Tensor] = None,
        roi_ious: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ):
        if epoch is not None:
            self._cur_epoch = int(epoch)

        device = roi_feats_per_scale[0].device
        self._ensure_prototypes(device)

        proj_per_scale = []
        for f in roi_feats_per_scale:
            if f.numel() == 0:
                proj_per_scale.append(torch.empty((0, self.feat_dim), device=device))
            else:
                p = self.proj(f)
                p = _safe_l2_normalize(p, dim=1)
                proj_per_scale.append(p)

        concat_feats = torch.cat(proj_per_scale, dim=0)  # [sum Ns, D]
        concat_labels = roi_labels.to(device)

        L = concat_feats.size(0)
        if concat_labels.size(0) != L:
            L2 = min(L, concat_labels.size(0))
            concat_feats = concat_feats[:L2]
            concat_labels = concat_labels[:L2]
            L = L2

        if roi_probs is not None:
            roi_probs = roi_probs.to(device)
            if roi_probs.size(0) != L:
                roi_probs = roi_probs[:L]
        if roi_ious is not None:
            roi_ious = roi_ious.to(device)
            if roi_ious.size(0) != L:
                roi_ious = roi_ious[:L]

        fg_mask = (concat_labels >= 0) & (concat_labels < self.num_classes)
        if roi_ious is not None and self.loss_iou_thr is not None:
            fg_mask = fg_mask & (roi_ious > float(self.loss_iou_thr))

        if fg_mask.sum().item() == 0:
            z = concat_feats.new_zeros(())
            return dict(
                loss_contrastive_total=z,
                loss_ms_consist=z,
                loss_supcon=z,
                loss_proto=z,
                contrastive_alpha=torch.tensor(0.0, device=device),
            )

        fg_feats = concat_feats[fg_mask]
        fg_labels = concat_labels[fg_mask]
        fg_probs = roi_probs[fg_mask] if roi_probs is not None else None
        fg_ious = roi_ious[fg_mask] if roi_ious is not None else None

        if self.use_finegrained_mb:
            supcon = self._supcon_loss_finegrained_mb(fg_feats, fg_labels, roi_probs=fg_probs)
        else:
            supcon = fg_feats.new_zeros(())

        if len(proj_per_scale) > 1:
            min_len = min([t.size(0) for t in proj_per_scale])
            if min_len >= 2:
                aligned = [t[:min_len] for t in proj_per_scale]
                pair_losses = [
                    self._cosine_pair_loss(aligned[i], aligned[j])
                    for i in range(len(aligned))
                    for j in range(i + 1, len(aligned))
                ]
                ms_consist = torch.stack(pair_losses).mean()
            else:
                ms_consist = supcon.new_zeros(())
        else:
            ms_consist = supcon.new_zeros(())

        self._update_prototypes_ema(fg_feats.detach(), fg_labels.detach())
        proto_targets = self.prototypes[fg_labels]
        proto_loss = self._cosine_pair_loss(fg_feats, proto_targets)

        if self.use_finegrained_mb:
            self._update_memory_diverse(fg_feats.detach(), fg_labels.detach(), ious=fg_ious)
        elif self.use_queue:
            self._enqueue_dequeue(fg_feats.detach(), fg_labels.detach())

        alpha_proto, alpha_sup, alpha_ms = self._compute_stage_alphas()
        scaled_proto = float(alpha_proto) * self.lam_proto * proto_loss
        scaled_sup = float(alpha_sup) * self.lam_supcon * supcon
        scaled_ms = float(alpha_ms) * self.lam_scale * ms_consist
        total = scaled_proto + scaled_sup + scaled_ms

        return dict(
            loss_contrastive_total=total,
            loss_ms_consist=scaled_ms,
            loss_supcon=scaled_sup,
            loss_proto=scaled_proto,
            contrastive_alpha=torch.tensor((alpha_proto, alpha_sup, alpha_ms), device=device),
        )
