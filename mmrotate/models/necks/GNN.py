import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_NECKS


class GATv2Layer(nn.Module):
    """GATv2 layer with similarity-based adjacency bias."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, negative_slope=0.2):
        super(GATv2Layer, self).__init__()

        assert out_dim % heads == 0, \
            f'out_dim should be divisible by heads, got out_dim={out_dim}, heads={heads}.'

        self.heads = heads
        self.head_dim = out_dim // heads
        self.out_dim = out_dim

        self.fc_src = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_val = nn.Linear(in_dim, out_dim, bias=False)

        self.attn = nn.Parameter(torch.Tensor(heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_src.weight)
        nn.init.xavier_uniform_(self.fc_dst.weight)
        nn.init.xavier_uniform_(self.fc_val.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.attn)

        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight)
            if self.res_proj.bias is not None:
                nn.init.zeros_(self.res_proj.bias)

    def forward(self, x, adj):
        """
        Args:
            x: Node features, shape [K, C].
            adj: Adaptive adjacency matrix, shape [K, K].

        Returns:
            Updated node features, shape [K, out_dim].
        """

        k = x.size(0)

        src = self.fc_src(x).view(k, self.heads, self.head_dim)
        dst = self.fc_dst(x).view(k, self.heads, self.head_dim)
        val = self.fc_val(x).view(k, self.heads, self.head_dim)

        # GATv2-style dynamic attention:
        # e_{pq}^{m} = a_m^T LeakyReLU(W_s h_p + W_t h_q)
        pair_feat = self.leaky_relu(src.unsqueeze(1) + dst.unsqueeze(0))
        attn_logits = (pair_feat * self.attn.unsqueeze(0).unsqueeze(0)).sum(-1)
        attn_logits = attn_logits.permute(2, 0, 1)  # [heads, K, K]

        # Use cosine-similarity adjacency as structural bias.
        attn_logits = attn_logits + adj.unsqueeze(0)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum('hij,jhd->ihd', attn_weights, val)
        out = out.reshape(k, self.out_dim)

        out = self.proj(out)
        out = self.dropout(out)

        out = self.norm(out + self.res_proj(x))

        return out


@ROTATED_NECKS.register_module()
class ASGA(nn.Module):
    """
    Adaptive Structured Graph Attention Module.

    Given RoI features, ASGA selects top-k informative spatial nodes according to
    hybrid activation-structural scores, constructs an adaptive cosine-similarity
    graph, performs GATv2-based message passing, and fuses RoI-level and image-level
    graph representations through a learnable gate.
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        num_layers=2,
        topk=8,
        lambda_mix=0.6,
        gat_heads=4,
        gat_dropout=0.1,
        pooling='attn',
        gate_init=0.2,
        eps=1e-6
    ):
        super(ASGA, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.topk = topk
        self.lambda_mix = lambda_mix
        self.pooling = pooling
        self.eps = eps

        self.gnn_layers = nn.ModuleList([
            GATv2Layer(
                in_dim=in_channels if i == 0 else out_channels,
                out_dim=out_channels,
                heads=gat_heads,
                dropout=gat_dropout
            )
            for i in range(num_layers)
        ])

        if pooling == 'attn':
            self.pool_attn = nn.Linear(out_channels, 1)
        elif pooling == 'mean':
            self.pool_attn = None
        else:
            raise ValueError(f'Unsupported pooling type: {pooling}')

        # Learnable scalar gate beta.
        self.gate_param = nn.Parameter(torch.ones(1) * gate_init)

        self._init_weights()

    def _init_weights(self):
        if self.pool_attn is not None:
            nn.init.xavier_uniform_(self.pool_attn.weight)
            if self.pool_attn.bias is not None:
                nn.init.zeros_(self.pool_attn.bias)

    def flatten_roi_feature(self, roi_feat):
        """
        Flatten RoI feature R_n from [C, H, W] to X_n with shape [H*W, C].
        """

        c, h, w = roi_feat.shape
        nodes = roi_feat.view(c, h * w).permute(1, 0).contiguous()

        return nodes, h, w

    def compute_activation_score(self, nodes):
        """
        Compute task-relevance score:
        s_j^{act} = ||X_j||.
        """

        act_score = torch.norm(nodes, p=2, dim=1)

        return act_score

    def compute_structural_score(self, nodes):
        """
        Compute structural-centrality score as the mean cosine similarity
        between one node and all other nodes.
        """

        num_nodes = nodes.size(0)

        norm_nodes = F.normalize(nodes, p=2, dim=1, eps=self.eps)
        sim_matrix = torch.matmul(norm_nodes, norm_nodes.t())

        if num_nodes > 1:
            eye = torch.eye(num_nodes, device=nodes.device, dtype=nodes.dtype)
            sim_without_self = sim_matrix * (1.0 - eye)
            struct_score = sim_without_self.sum(dim=1) / (num_nodes - 1)
        else:
            struct_score = sim_matrix.new_zeros(num_nodes)

        return struct_score

    def select_topk_nodes(self, roi_feat):
        """
        Select top-k nodes according to the hybrid positional score:
        s_j = lambda_mix * s_j^{act} + (1 - lambda_mix) * s_j^{struct}.
        """

        nodes, _, _ = self.flatten_roi_feature(roi_feat)

        act_score = self.compute_activation_score(nodes)
        struct_score = self.compute_structural_score(nodes)

        hybrid_score = self.lambda_mix * act_score + \
            (1.0 - self.lambda_mix) * struct_score

        k = min(self.topk, nodes.size(0))
        topk_indices = torch.topk(hybrid_score, k=k, dim=0).indices

        topk_nodes = nodes[topk_indices]

        return topk_nodes

    def build_adaptive_adjacency(self, nodes):
        """
        Build adaptive adjacency matrix using cosine similarity:
        A_{mn} = cos(X_m, X_n).
        """

        norm_nodes = F.normalize(nodes, p=2, dim=1, eps=self.eps)
        adj = torch.matmul(norm_nodes, norm_nodes.t())

        return adj

    def graph_reasoning(self, nodes):
        """
        Apply stacked GATv2 layers over selected graph nodes.
        """

        adj = self.build_adaptive_adjacency(nodes)

        for gnn_layer in self.gnn_layers:
            nodes = gnn_layer(nodes, adj)

        return nodes

    def node_attention_pooling(self, nodes):
        """
        Obtain RoI-level graph feature using learnable node-attention pooling.
        """

        if self.pooling == 'attn':
            attn_logits = self.pool_attn(nodes).squeeze(-1)
            gamma = F.softmax(attn_logits, dim=0)
            roi_graph_feat = torch.sum(nodes * gamma.unsqueeze(-1), dim=0)
        else:
            roi_graph_feat = nodes.mean(dim=0)

        return roi_graph_feat

    def compute_global_graph_feature(self, roi_graph_feats, batch_inds):
        """
        Average RoI-level graph features belonging to the same image to obtain
        image-level global graph representations.
        """

        num_imgs = int(batch_inds.max().item()) + 1

        global_feats = []
        for img_id in range(num_imgs):
            img_mask = batch_inds == img_id

            if img_mask.any():
                global_feat = roi_graph_feats[img_mask].mean(dim=0)
            else:
                global_feat = roi_graph_feats.new_zeros(self.out_channels)

            global_feats.append(global_feat)

        global_feats = torch.stack(global_feats, dim=0)

        return global_feats

    def forward(self, roi_feats, batch_inds=None):
        """
        Args:
            roi_feats: RoI features with shape [N, C, H, W].
            batch_inds: Image index of each RoI, shape [N].

        Returns:
            dict:
                g_roi: RoI-level graph features, shape [N, C_out].
                g_global: image-level global graph features, shape [B, C_out].
                g_global_perroi: corresponding global feature for each RoI, shape [N, C_out].
                g_fused: gated local-global graph embeddings, shape [N, C_out].
        """

        assert roi_feats.dim() == 4, \
            f'roi_feats should have shape [N, C, H, W], but got {roi_feats.shape}.'

        num_rois = roi_feats.size(0)
        device = roi_feats.device

        if batch_inds is None:
            batch_inds = torch.zeros(num_rois, dtype=torch.long, device=device)
        else:
            batch_inds = batch_inds.to(device=device, dtype=torch.long)

        roi_graph_feats = []

        for n in range(num_rois):
            selected_nodes = self.select_topk_nodes(roi_feats[n])
            refined_nodes = self.graph_reasoning(selected_nodes)
            roi_graph_feat = self.node_attention_pooling(refined_nodes)
            roi_graph_feats.append(roi_graph_feat)

        g_roi = torch.stack(roi_graph_feats, dim=0)

        g_global = self.compute_global_graph_feature(g_roi, batch_inds)
        g_global_perroi = g_global[batch_inds]

        gate = torch.sigmoid(self.gate_param)

        g_fused = gate * g_roi + (1.0 - gate) * g_global_perroi

        return {
            'g_roi': g_roi,
            'g_global': g_global,
            'g_global_perroi': g_global_perroi,
            'g_fused': g_fused
        }
