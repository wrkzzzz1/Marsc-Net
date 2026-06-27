import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_NECKS


class SELayer(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""

    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()

        hidden_channels = max(8, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        channel_weight = self.avg_pool(x).view(b, c)
        channel_weight = self.fc(channel_weight).view(b, c, 1, 1)

        return x * channel_weight


class SpatialAttentionMask(nn.Module):
    """Generate spatial attention mask from channel-weighted features."""

    def __init__(self, in_channels):
        super(SpatialAttentionMask, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        return attn


@ROTATED_NECKS.register_module()
class CAMM(nn.Module):
    """
    Causal Attention Modulation Module.

    For each pyramid feature P'_i, CAMM first applies SE channel attention,
    then generates a spatial attention mask. The factual output is obtained
    by residual fusion. During training, a counterfactual branch is optionally
    constructed by replacing the learned spatial mask with an object-constrained
    intervention mask within the ground-truth object region.
    """

    def __init__(
        self,
        in_channels=256,
        num_scales=5,
        reduction=16,
        mask_interv_prob=0.12,
        lambda_interv=2e-4
    ):
        super(CAMM, self).__init__()

        self.num_scales = num_scales
        self.mask_interv_prob = mask_interv_prob
        self.lambda_interv = lambda_interv

        self.se_blocks = nn.ModuleList([
            SELayer(in_channels, reduction=reduction)
            for _ in range(num_scales)
        ])

        self.spatial_attn_blocks = nn.ModuleList([
            SpatialAttentionMask(in_channels)
            for _ in range(num_scales)
        ])

    def generate_object_mask(self, bbox_list, feat, img_metas=None):
        """
        Generate binary object masks on the feature map according to GT boxes.

        Args:
            bbox_list (list[Tensor]): GT boxes of each image.
                Each box can be in horizontal format [x1, y1, x2, y2]
                or rotated format [xc, yc, w, h, angle].
            feat (Tensor): Feature map with shape [B, C, H, W].
            img_metas (list[dict], optional): Image meta information.

        Returns:
            Tensor: Object mask with shape [B, 1, H, W].
        """

        bsz, _, feat_h, feat_w = feat.shape
        device = feat.device

        object_mask = torch.zeros(
            bsz, 1, feat_h, feat_w,
            device=device,
            dtype=feat.dtype
        )

        if bbox_list is None:
            return object_mask

        for b in range(bsz):
            if b >= len(bbox_list) or bbox_list[b] is None:
                continue

            boxes = bbox_list[b]

            if len(boxes) == 0:
                continue

            if img_metas is not None:
                img_h, img_w = img_metas[b]['img_shape'][:2]
                scale_x = feat_w / float(img_w)
                scale_y = feat_h / float(img_h)
            else:
                scale_x = 1.0
                scale_y = 1.0

            for box in boxes:
                if isinstance(box, torch.Tensor):
                    box = box.detach().cpu().tolist()

                if len(box) == 4:
                    x1, y1, x2, y2 = [float(v) for v in box]

                elif len(box) >= 5:
                    xc, yc, w, h, _ = [float(v) for v in box[:5]]

                    x1 = xc - w / 2.0
                    y1 = yc - h / 2.0
                    x2 = xc + w / 2.0
                    y2 = yc + h / 2.0

                else:
                    continue

                xi1 = int(max(0, round(x1 * scale_x)))
                yi1 = int(max(0, round(y1 * scale_y)))
                xi2 = int(min(feat_w - 1, round(x2 * scale_x)))
                yi2 = int(min(feat_h - 1, round(y2 * scale_y)))

                if xi1 <= xi2 and yi1 <= yi2:
                    object_mask[b, 0, yi1:yi2 + 1, xi1:xi2 + 1] = 1.0

        return object_mask

    def generate_intervention_mask(self, spatial_mask, object_mask):
        """
        Generate object-constrained intervention mask.

        The learned spatial mask is kept unchanged outside the object region,
        while random perturbation is applied inside the ground-truth object region.

        Args:
            spatial_mask (Tensor): Learned attention mask A_s, shape [B, 1, H, W].
            object_mask (Tensor): Binary object mask, shape [B, 1, H, W].

        Returns:
            Tensor: Intervention mask \\bar{A}_s.
        """

        random_mask = torch.rand_like(spatial_mask)

        intervention_mask = spatial_mask * (1.0 - object_mask) + \
            random_mask * object_mask

        return intervention_mask

    def forward(self, inputs, img_metas=None, gt_bboxes=None, return_loss=False):
        """
        Args:
            inputs (list[Tensor] or tuple[Tensor]):
                Multi-scale input features P' = {P'_2, ..., P'_6}.
            img_metas (list[dict], optional):
                Image meta information.
            gt_bboxes (list[Tensor], optional):
                Ground-truth bounding boxes.
            return_loss (bool):
                Whether to compute intervention loss.

        Returns:
            dict:
                - feats: factual CAMM outputs P''
                - masks: learned spatial attention masks
                - loss: intervention loss dictionary
                - g_camm: global CAMM representation
        """

        assert len(inputs) == self.num_scales, \
            f'CAMM expects {self.num_scales} input feature maps, ' \
            f'but got {len(inputs)}.'

        outputs = []
        masks = []
        losses = {}

        for i, feat in enumerate(inputs):
            # Channel attention: SE(P'_i)
            se_feat = self.se_blocks[i](feat)

            # Spatial attention mask: A_s^{(i)}
            spatial_mask = self.spatial_attn_blocks[i](se_feat)
            masks.append(spatial_mask)

            # Factual branch:
            # P''_i = P'_i + SE(P'_i) * A_s^{(i)}
            factual_feat = feat + se_feat * spatial_mask.expand_as(feat)
            outputs.append(factual_feat)

            # Counterfactual branch for intervention loss
            if (
                self.training
                and return_loss
                and gt_bboxes is not None
                and torch.rand(1).item() < self.mask_interv_prob
            ):
                object_mask = self.generate_object_mask(
                    gt_bboxes,
                    feat,
                    img_metas=img_metas
                )

                intervention_mask = self.generate_intervention_mask(
                    spatial_mask,
                    object_mask
                )

                counterfactual_feat = feat + se_feat * intervention_mask.expand_as(feat)

                intervention_loss = torch.abs(
                    factual_feat - counterfactual_feat
                ).mean()

                losses[f'loss_interv_s{i}'] = self.lambda_interv * intervention_loss

            else:
                losses[f'loss_interv_s{i}'] = feat.new_zeros([])

        # Global CAMM representation, if required by the following modules.
        global_feats = []
        for feat in outputs:
            pooled_feat = F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
            global_feats.append(pooled_feat)

        g_camm = torch.stack(global_feats, dim=0).mean(dim=0)

        return {
            'feats': outputs,
            'masks': masks,
            'loss': losses,
            'g_camm': g_camm
        }
