from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models import ROTATED_HEADS


def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    norm = x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return x / norm


class MLPProj(nn.Module):
    """
    Two-layer MLP projection head used in DySC.
    """

    def __init__(self, in_dim: int, feat_dim: int, hidden_dim: Optional[int] = None):
        super(MLPProj, self).__init__()

        if hidden_dim is None:
            hidden_dim = max(in_dim, feat_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@ROTATED_HEADS.register_module()
class MultiScaleContrastiveHead(nn.Module):
    """
    Dynamic Scale-Consistency Contrastive Module.

    This module receives context-enhanced multi-scale RoI embeddings from ASGA:
        G_n^(s), s = 1, ..., S

    It projects them into a contrastive feature space and optimizes:
        1. multi-scale consistency loss L_ms
        2. prototype alignment loss L_proto
        3. supervised contrastive loss L_sup

    The three objectives are combined using stage-wise weights.
    """

    def __init__(
        self,
        in_dim=256,
        feat_dim=512,
        num_scales=5,
        num_classes=31,
        temperature=0.1,
        learnable_temp=True,
        proto_momentum=0.99,
        lambda_ms=0.003,
        lambda_sup=0.015,
        lambda_proto=0.08,
        warmup_epochs=12,
        ramp_epochs=8,
        supcon_delay=3,
        ms_delay=3,
        eps=1e-6,
        **kwargs
    ):
        super(MultiScaleContrastiveHead, self).__init__()

        self.in_dim = int(in_dim)
        self.feat_dim = int(feat_dim)
        self.num_scales = int(num_scales)
        self.num_classes = int(num_classes)

        self.lambda_ms = float(lambda_ms)
        self.lambda_sup = float(lambda_sup)
        self.lambda_proto = float(lambda_proto)

        self.proto_momentum = float(proto_momentum)
        self.eps = eps

        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.supcon_delay = int(supcon_delay)
        self.ms_delay = int(ms_delay)
        self.current_epoch = 0

        self.learnable_temp = bool(learnable_temp)
        if self.learnable_temp:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(float(temperature))).reshape(1)
            )
        else:
            self.register_buffer(
                "temperature_buf",
                torch.tensor(float(temperature))
            )

        self.proj = MLPProj(
            in_dim=self.in_dim,
            feat_dim=self.feat_dim,
            hidden_dim=max(self.in_dim, self.feat_dim)
        )

        self.register_buffer(
            "prototypes",
            torch.zeros(self.num_classes, self.feat_dim),
            persistent=True
        )
        self.register_buffer(
            "prototypes_initialized",
            torch.tensor(0, dtype=torch.uint8),
            persistent=False
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(nn.init, "trunc_normal_"):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    nn.init.normal_(m.weight, std=0.02)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def get_temperature(self, device: torch.device) -> torch.Tensor:
        if self.learnable_temp:
            return self.log_temperature.exp().clamp_min(1e-3).to(device)
        return self.temperature_buf.to(device)

    @torch.no_grad()
    def ensure_prototypes(self, device: torch.device):
        if self.prototypes_initialized.item() == 0:
            proto = torch.randn(
                self.num_classes,
                self.feat_dim,
                device=device
            )
            proto = safe_l2_normalize(proto, dim=1, eps=self.eps)
            self.prototypes.copy_(proto)
            self.prototypes_initialized.fill_(1)

    def compute_stage_weights(self) -> Tuple[float, float, float]:
        """
        Stage-wise weighting scheme.

        Returns:
            alpha_proto, alpha_sup, alpha_ms
        """

        e = self.current_epoch
        warm = self.warmup_epochs
        ramp = max(1, self.ramp_epochs)

        if e < warm:
            alpha_proto = 0.0
        else:
            alpha_proto = min(1.0, float(e - warm) / float(ramp))

        if e < warm + self.supcon_delay:
            alpha_sup = 0.0
        else:
            alpha_sup = min(
                1.0,
                float(e - warm - self.supcon_delay) / float(ramp)
            )

        if e < warm + self.supcon_delay + self.ms_delay:
            alpha_ms = 0.0
        else:
            alpha_ms = min(
                1.0,
                float(e - warm - self.supcon_delay - self.ms_delay) / float(ramp)
            )

        return alpha_proto, alpha_sup, alpha_ms

    def project_multiscale_embeddings(
        self,
        roi_feats_per_scale: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Project ASGA embeddings G_n^(s) into contrastive embeddings z_n^(s).
        """

        proj_feats = []

        for feat in roi_feats_per_scale:
            if feat.numel() == 0:
                proj_feats.append(
                    feat.new_zeros((0, self.feat_dim))
                )
            else:
                z = self.proj(feat)
                z = safe_l2_normalize(z, dim=1, eps=self.eps)
                proj_feats.append(z)

        return proj_feats

    def multi_scale_consistency_loss(
        self,
        proj_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        L_ms =
        2 / (S(S-1)) * sum_{s<t} [1 - 1/N * sum_n z_n^s · z_n^t]
        """

        num_scales = len(proj_feats)

        if num_scales <= 1:
            return proj_feats[0].new_zeros(())

        min_num = min([feat.size(0) for feat in proj_feats])

        if min_num == 0:
            return proj_feats[0].new_zeros(())

        aligned_feats = [feat[:min_num] for feat in proj_feats]

        pair_losses = []
        for s in range(num_scales):
            for t in range(s + 1, num_scales):
                sim = (aligned_feats[s] * aligned_feats[t]).sum(dim=1)
                pair_loss = 1.0 - sim.mean()
                pair_losses.append(pair_loss)

        if len(pair_losses) == 0:
            return proj_feats[0].new_zeros(())

        return torch.stack(pair_losses).mean()

    def prototype_alignment_loss(
        self,
        concat_feats: torch.Tensor,
        concat_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        L_proto =
        1 / (NS) sum_{s,n} [1 - z_n^(s) · p_{y_n}]
        """

        proto = self.prototypes[concat_labels]
        proto = safe_l2_normalize(proto, dim=1, eps=self.eps)

        sim = (concat_feats * proto).sum(dim=1)

        return 1.0 - sim.mean()

    def supervised_contrastive_loss(
        self,
        concat_feats: torch.Tensor,
        concat_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        SupCon-style supervised contrastive loss with negatives-only normalization.

        L_sup =
        1 / |I| sum_i [
            log sum_{a in N(i)} exp(z_i^T k_a / tau)
            -
            1 / |P(i)| sum_{p in P(i)} z_i^T k_p / tau
        ]
        """

        num_samples = concat_feats.size(0)

        if num_samples <= 1:
            return concat_feats.new_zeros(())

        device = concat_feats.device
        temp = self.get_temperature(device)

        logits = torch.matmul(concat_feats, concat_feats.t()) / temp

        labels = concat_labels.view(-1, 1)
        same_label = labels.eq(labels.t())

        self_mask = torch.eye(
            num_samples,
            dtype=torch.bool,
            device=device
        )

        pos_mask = same_label & (~self_mask)
        neg_mask = (~same_label)

        losses = []

        for i in range(num_samples):
            pos_logits = logits[i][pos_mask[i]]
            neg_logits = logits[i][neg_mask[i]]

            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue

            neg_logsumexp = torch.logsumexp(neg_logits, dim=0)
            pos_mean = pos_logits.mean()

            loss_i = neg_logsumexp - pos_mean
            losses.append(loss_i)

        if len(losses) == 0:
            return concat_feats.new_zeros(())

        return torch.stack(losses).mean()

    @torch.no_grad()
    def update_prototypes_ema(
        self,
        concat_feats: torch.Tensor,
        concat_labels: torch.Tensor
    ):
        """
        Update class prototypes with EMA.
        """

        unique_labels = torch.unique(concat_labels)

        for cls_id in unique_labels:
            cls_id_int = int(cls_id.item())

            if cls_id_int < 0 or cls_id_int >= self.num_classes:
                continue

            cls_mask = concat_labels == cls_id
            if cls_mask.sum().item() == 0:
                continue

            cur_proto = concat_feats[cls_mask].mean(dim=0)
            cur_proto = safe_l2_normalize(cur_proto, dim=0, eps=self.eps)

            old_proto = self.prototypes[cls_id_int]
            new_proto = (
                self.proto_momentum * old_proto
                + (1.0 - self.proto_momentum) * cur_proto
            )
            new_proto = safe_l2_normalize(new_proto, dim=0, eps=self.eps)

            self.prototypes[cls_id_int].copy_(new_proto)

    def forward(
        self,
        roi_feats_per_scale: List[torch.Tensor],
        roi_labels: torch.Tensor,
        epoch: Optional[int] = None
    ):
        """
        Args:
            roi_feats_per_scale:
                List of ASGA embeddings from different FPN levels.
                Each element has shape [N, C].
            roi_labels:
                RoI labels with shape [N].
                Negative samples should be marked as -1.
            epoch:
                Current training epoch for stage-wise weighting.

        Returns:
            dict of DySC losses.
        """

        if epoch is not None:
            self.set_epoch(epoch)

        assert isinstance(roi_feats_per_scale, (list, tuple)), \
            "roi_feats_per_scale should be a list or tuple."

        assert len(roi_feats_per_scale) > 0, \
            "roi_feats_per_scale should not be empty."

        device = roi_feats_per_scale[0].device
        self.ensure_prototypes(device)

        roi_labels = roi_labels.to(device=device, dtype=torch.long)

        valid_mask = (roi_labels >= 0) & (roi_labels < self.num_classes)

        if valid_mask.sum().item() == 0:
            zero = roi_feats_per_scale[0].new_zeros(())
            return dict(
                loss_contrastive_total=zero,
                loss_ms_consist=zero,
                loss_supcon=zero,
                loss_proto=zero,
                contrastive_alpha=torch.tensor(
                    [0.0, 0.0, 0.0],
                    device=device
                )
            )

        filtered_feats_per_scale = [
            feat[valid_mask] for feat in roi_feats_per_scale
        ]
        filtered_labels = roi_labels[valid_mask]

        proj_feats_per_scale = self.project_multiscale_embeddings(
            filtered_feats_per_scale
        )

        loss_ms = self.multi_scale_consistency_loss(proj_feats_per_scale)

        concat_feats = torch.cat(proj_feats_per_scale, dim=0)

        concat_labels = filtered_labels.repeat(len(proj_feats_per_scale))

        loss_sup = self.supervised_contrastive_loss(
            concat_feats,
            concat_labels
        )

        self.update_prototypes_ema(
            concat_feats.detach(),
            concat_labels.detach()
        )

        loss_proto = self.prototype_alignment_loss(
            concat_feats,
            concat_labels
        )

        alpha_proto, alpha_sup, alpha_ms = self.compute_stage_weights()

        scaled_proto = alpha_proto * self.lambda_proto * loss_proto
        scaled_sup = alpha_sup * self.lambda_sup * loss_sup
        scaled_ms = alpha_ms * self.lambda_ms * loss_ms

        loss_total = scaled_proto + scaled_sup + scaled_ms

        return dict(
            loss_contrastive_total=loss_total,
            loss_ms_consist=scaled_ms,
            loss_supcon=scaled_sup,
            loss_proto=scaled_proto,
            contrastive_alpha=torch.tensor(
                [alpha_proto, alpha_sup, alpha_ms],
                device=device
            )
        )
