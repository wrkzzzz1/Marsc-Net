# mmrotate/models/dense_heads/kld_reppoints_head.py
import torch
import torch.nn as nn

from mmcv.cnn import normal_init, bias_init_with_prob
from mmcv.runner import force_fp32

# 正确的位置导入 build_loss
from mmdet.models.builder import build_loss
# RepPointsHead 实现来自 mmdet
from mmdet.models.dense_heads.reppoints_head import RepPointsHead

from mmrotate.models.builder import ROTATED_HEADS
# 从 mmdet.core 导入通用工具
from mmdet.core import multiclass_nms, build_assigner, build_sampler


@ROTATED_HEADS.register_module()
class KLDRepPointsHead(RepPointsHead):
    """KLD-based RepPoints head, aligned to mmrotate 0.3.4.

    Notes:
      - Accepts cfg keys used in your config (e.g. use_reassign, topk, anti_factor).
      - Uses build_loss from mmdet.models.builder.
      - Uses RepPointsHead as base (from mmdet).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 stacked_convs=3,
                 feat_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 gradient_mul=0.1,
                 point_strides=(8, 16, 32, 64, 128),
                 point_base_scale=4,
                 loss_cls=dict(type='FocalLoss',
                               use_sigmoid=True,
                               gamma=2.0,
                               alpha=0.25,
                               loss_weight=1.0),
                 loss_bbox_init=dict(type='KLDRepPointsLoss', loss_weight=0.5),
                 loss_bbox_refine=dict(type='KLDRepPointsLoss', loss_weight=1.0),
                 transform_method='moment',
                 use_grid_points=False,
                 center_init=True,
                 # --- these are the cfg keys that previously caused unexpected-kw errors ---
                 use_reassign=False,
                 topk=6,
                 anti_factor=0.75,
                 **kwargs):
        # call parent ctor (do NOT forward unknown extra cfg keys to parent)
        super(KLDRepPointsHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            point_feat_channels=point_feat_channels,
            stacked_convs=stacked_convs,
            feat_channels=feat_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            gradient_mul=gradient_mul,
            point_strides=point_strides,
            point_base_scale=point_base_scale,
            loss_cls=loss_cls,
            loss_bbox_init=loss_bbox_init,
            loss_bbox_refine=loss_bbox_refine,
            transform_method=transform_method,
            use_grid_points=use_grid_points,
            center_init=center_init,
            **kwargs)

        # store cfg keys so behavior can later reference them if needed
        self.use_reassign = use_reassign
        self.topk = topk
        self.anti_factor = anti_factor

        # build KLD losses via mmdet builder (this expects your KLDRepPointsLoss is registered)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)

        # assigner & sampler - align with common mmrotate defaults
        # if build_assigner/build_sampler are available in mmdet.core, they'll be used
        # otherwise user must ensure these names exist in their environment
        self.assigner_init = build_assigner(dict(
            type='MaxIoUAssignerR',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1))
        self.assigner_refine = build_assigner(dict(type='ATSSAssignerR', topk=9))
        self.sampler = build_sampler(dict(type='PseudoSampler'))

    def init_weights(self):
        """Init weights in the same style as RepPointsHead / mmrotate."""
        super(KLDRepPointsHead, self).init_weights()
        # consistent bias init for classification head if exists
        if hasattr(self, 'reppoints_cls_out'):
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.reppoints_cls_out.bias, bias_init)
        if hasattr(self, 'reppoints_pts_init_out'):
            normal_init(self.reppoints_pts_init_out, std=0.01)
        if hasattr(self, 'reppoints_pts_refine_out'):
            normal_init(self.reppoints_pts_refine_out, std=0.01)

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine,
                    labels, label_weights, bbox_gt, bbox_weights,
                    stride, num_total_samples):
        """Compute single-level loss for init + refine stages."""
        # flatten labels & weights
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)

        # bbox (KLD) losses
        pts_pred_init = pts_pred_init.reshape(-1, self.num_points * 2)
        pts_pred_refine = pts_pred_refine.reshape(-1, self.num_points * 2)
        bbox_gt = bbox_gt.reshape(-1, bbox_gt.size(-1))
        bbox_weights = bbox_weights.reshape(-1, bbox_weights.size(-1))

        loss_bbox_init = self.loss_bbox_init(
            pts_pred_init, bbox_gt, bbox_weights, stride=stride, avg_factor=num_total_samples)
        loss_bbox_refine = self.loss_bbox_refine(
            pts_pred_refine, bbox_gt, bbox_weights, stride=stride, avg_factor=num_total_samples)

        return loss_cls, loss_bbox_init, loss_bbox_refine

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """Forward train with assign+sample and full loss aggregation.

        This implements the standard mmrotate pattern. The function intentionally
        avoids forwarding unused cfg keys to parent class constructors.
        """
        cls_scores, pts_preds_init, pts_preds_refine = self.forward(x)

        device = cls_scores[0].device
        featmap_sizes = [(f.shape[2], f.shape[3]) for f in cls_scores]

        # flatten predictions per-level
        cls_flatten = []
        pts_init_flatten = []
        pts_refine_flatten = []
        for cls_score, pts_init, pts_refine in zip(cls_scores, pts_preds_init, pts_preds_refine):
            B, C, H, W = cls_score.shape
            cls_flatten.append(cls_score.permute(0, 2, 3, 1).reshape(B, -1, C))
            pts_init_flatten.append(pts_init.permute(0, 2, 3, 1).reshape(B, -1, self.num_points, 2))
            pts_refine_flatten.append(pts_refine.permute(0, 2, 3, 1).reshape(B, -1, self.num_points, 2))

        cls_all = torch.cat(cls_flatten, dim=1)            # (B, L_all, C)
        pts_init_all = torch.cat(pts_init_flatten, dim=1)  # (B, L_all, P, 2)
        pts_refine_all = torch.cat(pts_refine_flatten, dim=1)  # (B, L_all, P, 2)

        # produce targets for this batch
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list = \
            self.get_targets([torch.zeros(1, 1, h, w, device=device) for h, w in featmap_sizes],
                             gt_bboxes, gt_labels, img_metas)

        # aggregate losses per image
        loss_cls_all = []
        loss_init_all = []
        loss_refine_all = []
        num_total_samples = 0
        for labels in labels_list:
            num_total_samples += max(1, (labels > 0).sum().item())
        num_total_samples = max(1, num_total_samples)

        for img_id in range(len(labels_list)):
            labels = labels_list[img_id]
            label_weights = label_weights_list[img_id]
            bbox_targets = bbox_targets_list[img_id]
            bbox_weights = bbox_weights_list[img_id]

            cls_pred = cls_all[img_id]
            pts_init_pred = pts_init_all[img_id]
            pts_refine_pred = pts_refine_all[img_id]

            # classification loss
            loss_cls_val = self.loss_cls(cls_pred, labels, label_weights, avg_factor=num_total_samples)

            # flatten points for bbox loss
            pts_init_flat = pts_init_pred.reshape(-1, self.num_points * 2)
            pts_refine_flat = pts_refine_pred.reshape(-1, self.num_points * 2)

            # call KLD losses (note: loss implementations are expected to accept these shapes)
            loss_init_val = self.loss_bbox_init(
                pts_init_flat,
                bbox_targets.reshape(-1, bbox_targets.shape[-2], bbox_targets.shape[-1]),
                bbox_weights.reshape(-1, bbox_weights.shape[-2], bbox_weights.shape[-1]),
                avg_factor=num_total_samples)

            loss_refine_val = self.loss_bbox_refine(
                pts_refine_flat,
                bbox_targets.reshape(-1, bbox_targets.shape[-2], bbox_targets.shape[-1]),
                bbox_weights.reshape(-1, bbox_weights.shape[-2], bbox_weights.shape[-1]),
                avg_factor=num_total_samples)

            loss_cls_all.append(loss_cls_val)
            loss_init_all.append(loss_init_val)
            loss_refine_all.append(loss_refine_val)

        loss_dict = dict(
            loss_cls=torch.stack(loss_cls_all).mean() if len(loss_cls_all) else torch.tensor(0., device=device),
            loss_pts_init=torch.stack(loss_init_all).mean() if len(loss_init_all) else torch.tensor(0., device=device),
            loss_pts_refine=torch.stack(loss_refine_all).mean() if len(loss_refine_all) else torch.tensor(0., device=device),
        )
        return loss_dict

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds_refine,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Decode bboxes from bbox_preds_refine using self.point_coder.decode."""
        assert len(cls_scores) == len(bbox_preds_refine)
        mlvl_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds_refine, self.point_strides):
            scores = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.num_points * 2)

            # decode using parent-provided point_coder
            bboxes = self.point_coder.decode(bbox_pred, stride, self.transform_method)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes, mlvl_scores,
                cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores
