import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_NECKS

def dice_loss(input, target, smooth=1e-5):
    B = input.size(0)
    input_flat = input.contiguous().view(B, -1)
    target_flat = target.contiguous().view(B, -1)
    inter = (input_flat * target_flat).sum(dim=1)
    union = input_flat.sum(dim=1) + target_flat.sum(dim=1)
    return 1 - (2 * inter + smooth) / (union + smooth)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(8, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SimpleAttnMask(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.se = SELayer(in_channels, reduction)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 4, 1, bias=False)
        self.last = nn.Conv2d(in_channels // 4, 1, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.last.bias, 4.0)

    def forward(self, x):
        x_se = self.se(x)
        attn = self.conv1x1(x_se)
        attn = self.last(attn)
        attn_map = self.sigmoid(attn)
        return attn_map

@ROTATED_NECKS.register_module()
class CAMM(nn.Module):
    '''
    Flagship CAMM: SOTA提升用（主类极度强化、因果干预、困难区mask放大loss）
    '''
    def __init__(
        self,
        in_channels=256,
        num_scales=5,
        use_mask=True,
        mask_interv_prob=0.12,
        lambda_interv=2e-4,
        lambda_gtmask=2e-4,
        lambda_gtmask_dice=2e-4,
        lambda_maincls=5.0,
        lambda_hard=3.5,
        warmup_iters=700,
        maincls_indices=(1,)
    ):
        super().__init__()
        self.num_scales = num_scales
        self.use_mask = use_mask
        self.mask_interv_prob = mask_interv_prob
        self.lambda_interv = lambda_interv
        self.lambda_gtmask = lambda_gtmask
        self.lambda_gtmask_dice = lambda_gtmask_dice
        self.lambda_maincls = lambda_maincls
        self.lambda_hard = lambda_hard
        self.warmup_iters = warmup_iters
        self.iter = 0
        self.maincls_indices = set(maincls_indices)
        self.se_blocks = nn.ModuleList([SELayer(in_channels) for _ in range(num_scales)])
        self.attn_blocks = nn.ModuleList([SimpleAttnMask(in_channels) for _ in range(num_scales)])
        self.bce_loss = nn.BCELoss(reduction='none')

    def generate_box_mask(self, bbox_list, feat, img_shape=None, gt_labels=None):
        B, _, H, W = feat.shape
        device = feat.device
        mask = torch.zeros(B, 1, H, W, device=device)
        maincls_mask = torch.zeros_like(mask)
        for b in range(B):
            if bbox_list is None or len(bbox_list[b]) == 0:
                continue
            labels = gt_labels[b] if (gt_labels is not None and len(gt_labels[b]) == len(bbox_list[b])) else None
            for j, box in enumerate(bbox_list[b]):
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy()
                # box可以是list或者ndarray
                if len(box) == 4:
                    x1, y1, x2, y2 = [float(v) for v in box]
                elif len(box) == 5:
                    xc, yc, w, h, angle = [float(v) for v in box[:5]]
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2
                else:
                    continue
                if img_shape is not None:
                    scale_x = W / img_shape[b][1]
                    scale_y = H / img_shape[b][0]
                else:
                    scale_x = scale_y = 1.0

                # 修正：保证全部输入float，round无报错
                xi1 = int(max(0, round(x1 * scale_x)))
                yi1 = int(max(0, round(y1 * scale_y)))
                xi2 = int(min(W - 1, round(x2 * scale_x)))
                yi2 = int(min(H - 1, round(y2 * scale_y)))
                if xi1 <= xi2 and yi1 <= yi2:
                    mask[b, 0, yi1:yi2 + 1, xi1:xi2 + 1] = 1.0
                    if labels is not None and int(labels[j]) in self.maincls_indices:
                        maincls_mask[b, 0, yi1:yi2 + 1, xi1:xi2 + 1] = 1.0
        return mask, maincls_mask

    def forward(self, inputs, img_metas=None, gt_bboxes=None, gt_labels=None, return_loss=False):
        self.iter += 1
        outputs = []
        masks_list = []
        losses = {}
        for i, feat in enumerate(inputs):
            B, C, H, W = feat.shape
            x_se = self.se_blocks[i](feat)

            # 1. mask生成
            if not self.use_mask or self.iter < self.warmup_iters:
                mask = feat.new_ones(B, 1, H, W) * 0.98
            else:
                mask = self.attn_blocks[i](x_se)
                mask = torch.clamp(mask, 0.0, 0.98)
            masks_list.append(mask)

            mask_exp = mask.expand_as(feat)

            # 2. 主分支因果门控
            do_interv = (torch.rand(1).item() < self.mask_interv_prob and self.iter > self.warmup_iters)
            if do_interv:
                rand_mask = torch.rand_like(mask) * 0.98
                feat_interv = x_se * rand_mask.expand_as(feat) + feat
                outputs.append(feat_interv)
                interv_loss = (feat_interv - (x_se * mask_exp + feat)).abs().mean()
                losses[f'loss_interv_s{i}'] = self.lambda_interv * interv_loss
            else:
                outputs.append(x_se * mask_exp + feat)
                losses[f'loss_interv_s{i}'] = feat.new_zeros([])

            # 3. mask合理性loss
            if gt_bboxes is not None and self.iter >= self.warmup_iters:
                img_shape = [meta['img_shape'][:2] for meta in img_metas] if (img_metas is not None and 'img_shape' in img_metas[0]) else None
                mask_gt, mask_main = self.generate_box_mask(gt_bboxes, feat, img_shape, gt_labels)
                pos = (mask_gt > 0.5).float()
                mainpos = (mask_main > 0.5).float()
                mask_hard = ((mask > 0.12) & (mask < 0.88)).float() * pos
                weights = pos + mainpos * (self.lambda_maincls - 1) + mask_hard * (self.lambda_hard - 1)
                bce = self.bce_loss(mask, mask_gt)
                bce_loss = (bce * weights).sum() / (weights.sum() + 1e-7)
                dice1 = dice_loss(mask * pos, pos)
                dice2 = dice_loss(mask * mainpos, mainpos)
                losses[f'loss_gtmask_s{i}'] = self.lambda_gtmask * bce_loss + self.lambda_gtmask_dice * (dice1 + dice2)
            else:
                losses[f'loss_gtmask_s{i}'] = feat.new_zeros([])

        # 4. 全局结构特征 g_camm
        global_feats = []
        for feat in outputs:
            pooled = F.adaptive_avg_pool2d(feat, 1).view(feat.shape[0], -1)
            global_feats.append(pooled)
        g_camm = torch.stack(global_feats, dim=0).mean(0)   # [B,C]

        # shape保险
        if g_camm.dim() > 2:
            g_camm = g_camm.view(g_camm.shape[0], -1)
        assert g_camm.dim() == 2, f"g_camm shape必须为 [B, C]，当前为 {g_camm.shape}"

        return {
            'feats': outputs,    # list
            'masks': masks_list, # list
            'loss': losses,      # dict
            'g_camm': g_camm     # [B, C]
        }
