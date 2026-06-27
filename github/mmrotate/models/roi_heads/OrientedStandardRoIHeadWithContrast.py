import torch
from mmrotate.core import rbbox2roi
from ..builder import ROTATED_HEADS
from .rotate_standard_roi_head import RotatedStandardRoIHead
from mmrotate.models import ROTATED_NECKS

@ROTATED_HEADS.register_module()
class OrientedStandardRoIHeadWithContrast(RotatedStandardRoIHead):
    def __init__(self, *args, contrastive_head=None, gnn_module=None, contrastive_alpha=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_head = ROTATED_HEADS.build(contrastive_head) if contrastive_head is not None else None
        self.gnn_module = None
        if gnn_module is not None:
            self.gnn_module = ROTATED_NECKS.build(gnn_module) if isinstance(gnn_module, dict) else gnn_module
        self.contrastive_alpha = contrastive_alpha
        self.current_epoch = 0  # 初始化

    def forward_dummy(self, x, proposals):
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        return outs

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        gt_masks=None,
        return_sampling_results=False,
        epoch=None,
        **kwargs
    ):
        # === 方案A: 在forward阶段直接拿到当前epoch ===
        if epoch is not None:
            self.current_epoch = int(epoch)
        elif 'epoch' in img_metas[0]:
            # 如果 dataloader 在 img_metas 中传了 epoch
            self.current_epoch = int(img_metas[0]['epoch'])
        elif hasattr(self, 'current_epoch') and self.current_epoch is not None:
            pass  # 保持之前的值
        else:
            self.current_epoch = 0

        # 同步给 contrastive_head
        if self.contrastive_head is not None:
            self.contrastive_head.set_epoch(self.current_epoch)

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            if gt_bboxes[i].numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                    (0, gt_bboxes[0].size(-1))).zero_()
            else:
                sampling_result.pos_gt_bboxes = gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
            sampling_results.append(sampling_result)

        losses = dict()
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x, sampling_results, gt_bboxes, gt_labels, img_metas)
            losses.update(bbox_results['loss_bbox'])

        # ====== 对比约束分支 ======
        if self.contrastive_head is not None and self.gnn_module is not None:
            rois = rbbox2roi([res.bboxes for res in sampling_results])

            num_scales = self.contrastive_head.num_scales if hasattr(self.contrastive_head, 'num_scales') else len(x)
            gnn_feats_per_scale = []
            for lvl in range(num_scales):
                roi_feat = self.bbox_roi_extractor([x[lvl]], rois)  # [N_total, C, H, W]
                batch_inds = rois[:, 0].long() if rois.size(1) > 5 else None
                gnn_out = self.gnn_module(roi_feat, batch_inds=batch_inds)
                gnn_feats_per_scale.append(gnn_out['g_roi'])  # [N_total, C]
            roi_feats_per_scale = gnn_feats_per_scale

            # 构造 roi_labels
            roi_labels_list = []
            for res in sampling_results:
                n = res.bboxes.size(0)
                labels = torch.full((n,), -1, dtype=torch.long, device=res.bboxes.device)
                if hasattr(res, "pos_inds") and len(res.pos_inds) > 0:
                    valid_mask = res.pos_inds < n
                    good_inds = res.pos_inds[valid_mask]
                    good_labels = res.pos_gt_labels[valid_mask]
                    labels[good_inds] = good_labels
                roi_labels_list.append(labels)
            roi_labels = torch.cat(roi_labels_list, 0)

            num_classes = self.bbox_head.num_classes
            valid_mask_global = (roi_labels >= -1) & (roi_labels < num_classes)

            num_scales = len(roi_feats_per_scale)
            per_scale_num = roi_feats_per_scale[0].shape[0]
            total_rois = per_scale_num * num_scales

            roi_labels = roi_labels[:total_rois]
            valid_mask_global = valid_mask_global[:total_rois]

            roi_feats_per_scale_new = []
            roi_labels_new = []
            for i in range(num_scales):
                start = i * per_scale_num
                end = (i + 1) * per_scale_num
                mask_i = valid_mask_global[start:end]
                roi_feats_per_scale_new.append(roi_feats_per_scale[i][mask_i])
                roi_labels_new.append(roi_labels[start:end][mask_i])
            roi_feats_per_scale = roi_feats_per_scale_new
            roi_labels = torch.cat(roi_labels_new, dim=0)

            roi_feats_per_scale_detached = [f.detach() for f in roi_feats_per_scale]
            contrastive_losses = self.contrastive_head(
                roi_feats_per_scale_detached, roi_labels, epoch=self.current_epoch
            )
            contrastive_losses = {k: v * self.contrastive_alpha for k, v in contrastive_losses.items()}
            losses.update(contrastive_losses)

        if return_sampling_results:
            return losses, sampling_results
        else:
            return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward(self, x, rois):
        roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        roi_feats_flat = roi_feats.flatten(1)
        cls_score, bbox_pred = self.bbox_head(roi_feats_flat)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, roi_feats=roi_feats)
        return bbox_results

    def simple_test_bboxes(
        self, x, img_metas, proposals, rcnn_test_cfg, rescale=False
    ):
        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        det_bboxes, det_labels = [], []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
