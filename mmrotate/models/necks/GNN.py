import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_NECKS

# ======================
# Focal Loss
# ======================
def focal_loss(inputs, targets, alpha=0.3, gamma=2.0, reduction='mean', sample_weight=None):
    logpt = F.log_softmax(inputs, dim=1)
    pt = torch.exp(logpt)
    targets_onehot = F.one_hot(targets, inputs.size(1)).float()
    loss = -alpha * (1 - pt) ** gamma * logpt * targets_onehot
    loss = loss.sum(dim=1)
    if sample_weight is not None:
        loss = loss * sample_weight
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# ======================
# 各类 GNN 层
# ======================
class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.fc_q = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.fc_k = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.fc_v = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.fc_attn = nn.Linear(out_dim, 1, bias=False)
        self.fc_out = nn.Linear(out_dim * heads, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.fc_res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, adj=None):
        Q = self.fc_q(x).view(-1, self.heads, self.out_dim)
        K = self.fc_k(x).view(-1, self.heads, self.out_dim)
        V = self.fc_v(x).view(-1, self.heads, self.out_dim)
        N = Q.shape[0]
        attn_in = Q.unsqueeze(1) + K.unsqueeze(0)
        attn_score = self.fc_attn(attn_in).squeeze(-1).permute(2, 0, 1)
        if adj is not None:
            attn_score = attn_score + adj.unsqueeze(0)
        attn = F.softmax(attn_score, dim=-1)
        node_out = torch.einsum('hij,jhd->ihd', attn, V).reshape(N, -1)
        node_out = self.fc_out(node_out)
        res = self.fc_res(x) if self.fc_res else x
        node_out = self.norm(self.drop(node_out + res))
        return node_out


class GATLayerWeakLite(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.fc_out = nn.Linear(out_dim * heads, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.fc_res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, adj=None):
        N = x.size(0)
        h = self.W(x).view(N, self.heads, self.out_dim)
        h_i = h.unsqueeze(1).repeat(1, N, 1, 1)
        h_j = h.unsqueeze(0).repeat(N, 1, 1, 1)
        e = self.attn_fc(torch.cat([h_i, h_j], dim=-1)).squeeze(-1).permute(2, 0, 1)
        e = e / 1.5
        if adj is not None:
            e = e + adj.unsqueeze(0)
        attn = F.softmax(e, dim=-1)
        node_out = torch.einsum('hij,jhd->ihd', attn, h).reshape(N, -1)
        node_out = self.fc_out(node_out)
        res = self.fc_res(x) if self.fc_res else x
        node_out = self.norm(self.drop(node_out + 0.8 * res))
        return node_out


class GCNLayerSameAPI(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.fc_res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)
        adj = adj + torch.eye(adj.size(0), device=x.device)
        deg_inv_sqrt = adj.sum(-1).pow(-0.5)
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        h = torch.mm(adj_norm, self.fc(x))
        res = self.fc_res(x) if self.fc_res else x
        h = self.norm(self.drop(h + res))
        return h


class GraphSAGELayerSameAPI(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.fc_res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)
        neighbor_mean = torch.mm(adj, x) / (adj.sum(-1, keepdim=True) + 1e-6)
        h = self.fc(torch.cat([x, neighbor_mean], dim=-1))
        res = self.fc_res(x) if self.fc_res else x
        h = self.norm(self.drop(h + res))
        return h

# ======================
# StructGNN 模块注册
# ======================
@ROTATED_NECKS.register_module()
class StructGNN(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        num_layers=2,
        topk=8,
        num_classes=31,
        pooling='attn',
        max_rois=32,
        focal_alpha=0.3,
        focal_gamma=2.0,
        class_counts=None,
        gat_heads=4,
        gat_dropout=0.1,
        final_dropout=0.1,
        fusion_type='gated',
        gate_init=0.2,
        topk_method='grad_struct',
        lambda_mix=0.6,
        gnn_type='gatv2'   # 支持 gatv2/gat/gcn/sage
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.topk = topk
        self.num_classes = num_classes
        self.pooling = pooling
        self.max_rois = max_rois
        self.fusion_type = fusion_type
        self.lambda_mix = lambda_mix

        # ===== 根据 gnn_type 构造不同图层 =====
        if gnn_type.lower() == 'gatv2':
            layer_cls = GATv2Layer
        elif gnn_type.lower() == 'gat':
            layer_cls = GATLayerWeakLite
        elif gnn_type.lower() == 'gcn':
            layer_cls = GCNLayerSameAPI
        elif gnn_type.lower() == 'sage':
            layer_cls = GraphSAGELayerSameAPI
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.gnn_layers = nn.ModuleList([
            layer_cls(
                in_dim=in_channels if i == 0 else out_channels,
                out_dim=out_channels,
                heads=gat_heads,
                dropout=gat_dropout
            ) if 'gat' in gnn_type.lower() else
            layer_cls(
                in_dim=in_channels if i == 0 else out_channels,
                out_dim=out_channels,
                dropout=gat_dropout
            )
            for i in range(num_layers)
        ])

        if pooling == 'attn':
            self.pool_attn = nn.Linear(out_channels, 1)
        self.final_dropout = nn.Dropout(final_dropout)
        self.cls_head = nn.Linear(out_channels * 2, num_classes)
        self.ln = nn.LayerNorm(out_channels * 2)

        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32)
            weights = 1.0 / (counts.sqrt() + 1e-6)
            weights = weights / weights.max()
            self.register_buffer('class_weights', weights)
        else:
            self.class_weights = None

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if fusion_type == 'gated':
            self.gate_param = nn.Parameter(torch.ones(1) * gate_init)
        elif fusion_type == 'dynamic_gated':
            self.gate_fc = nn.Linear(out_channels * 2, out_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def extract_topk_nodes(self, roi_feat, grad_map=None, lambda_mix=0.6):
        C, H, W = roi_feat.shape
        flat = roi_feat.view(C, -1).permute(1, 0)
        if grad_map is not None:
            grad_score = grad_map.view(-1)
            grad_score = (grad_score - grad_score.min()) / (grad_score.max() - grad_score.min() + 1e-6)
        else:
            grad_score = flat.norm(dim=1)
        norm_nodes = F.normalize(flat, dim=-1)
        sim_matrix = torch.mm(norm_nodes, norm_nodes.t())
        struct_score = sim_matrix.sum(dim=1) / (H * W)
        struct_score = (struct_score - struct_score.min()) / (struct_score.max() - struct_score.min() + 1e-6)
        combined_score = lambda_mix * grad_score + (1 - lambda_mix) * struct_score
        k = min(self.topk, combined_score.size(0))
        return flat[torch.topk(combined_score, k=k)[1]]

    def build_adj(self, nodes):
        norm_nodes = F.normalize(nodes, dim=-1)
        return torch.mm(norm_nodes, norm_nodes.t())

    def forward(self, roi_feats, gt_labels=None, batch_inds=None, **kwargs):
        N = roi_feats.size(0)
        device = roi_feats.device
        if N > self.max_rois:
            idxs = torch.randperm(N, device=device)[:self.max_rois]
            roi_feats = roi_feats[idxs]
            if batch_inds is not None:
                batch_inds = batch_inds[idxs]
            if gt_labels is not None:
                gt_labels = gt_labels[idxs]
            N = self.max_rois
        if batch_inds is None:
            batch_inds = torch.zeros(N, dtype=torch.long, device=device)

        graph_feats = []
        for i in range(N):
            nodes = self.extract_topk_nodes(roi_feats[i], lambda_mix=self.lambda_mix)
            adj = self.build_adj(nodes)
            for gnn in self.gnn_layers:
                nodes = gnn(nodes, adj)
            attn = torch.softmax(self.pool_attn(nodes).squeeze(-1), 0) if hasattr(self, 'pool_attn') else None
            graph_feat = (nodes * attn.unsqueeze(-1)).sum(0) if attn is not None else nodes.mean(0)
            graph_feats.append(graph_feat)
        x_out = self.final_dropout(torch.stack(graph_feats, 0))

        batch_num = int(batch_inds.max().item()) + 1
        g_global = torch.stack([x_out[batch_inds == bi].mean(0) if (batch_inds == bi).any()
                                else x_out.new_zeros(self.out_channels)
                                for bi in range(batch_num)], 0)
        g_global_perroi = g_global[batch_inds]

        if self.fusion_type == 'dynamic_gated':
            gate = torch.sigmoid(self.gate_fc(torch.cat([x_out, g_global_perroi], 1)))
            feat_cat = torch.cat([gate * x_out, (1 - gate) * g_global_perroi], 1)
        elif self.fusion_type == 'gated':
            gate = torch.sigmoid(self.gate_param)
            feat_cat = torch.cat([gate * x_out, (1 - gate) * g_global_perroi], 1)
        else:
            feat_cat = torch.cat([x_out, g_global_perroi], 1)

        feat_cat = F.relu(self.ln(feat_cat))
        feat_cat = self.final_dropout(feat_cat)
        pred_cls = self.cls_head(feat_cat)

        output = dict(g_roi=x_out, g_global=g_global, pred_cls=pred_cls)
        if self.training and gt_labels is not None:
            sample_weight = self.class_weights[gt_labels] if self.class_weights is not None else None
            output['loss_cls'] = focal_loss(pred_cls, gt_labels,
                                            alpha=self.focal_alpha,
                                            gamma=self.focal_gamma,
                                            reduction='mean',
                                            sample_weight=sample_weight)
        return output
