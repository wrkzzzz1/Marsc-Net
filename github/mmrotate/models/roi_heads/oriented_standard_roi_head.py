import torch
from mmrotate.core import rbbox2roi
from ..builder import ROTATED_HEADS, build_head
from .rotate_standard_roi_head import RotatedStandardRoIHead

@ROTATED_HEADS.register_module()
class OrientedStandardRoIHead(RotatedStandardRoIHead):
    """Oriented ROI head，扩展版：支持 MultiScaleContrastiveHead（弱类增强 + Fine-Grained MB）"""

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """初始化分类/回归 head，并可附加对比学习 head。"""
        super().init_bbox_head(bbox_roi_extractor, bbox_head)
        if hasattr(self.bbox_head, "contrastive_cfg"):
            c_cfg = self.bbox_head.contrastive_cfg
            self.contrastive_head = build_head(c_cfg)
        else:
            self.contrastive_head = None

    def forward_dummy(self, x, proposals):
        """推理测试用 dummy forward。"""
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        return outs

    def _bbox_forward(self, x, rois):
        """常规 bbox forward。"""
        roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)  # [N, C, 7, 7]
        roi_feats_flat = roi_feats.flatten(1)              # [N, C*7*7]
        cls_score, bbox_pred = self.bbox_head(roi_feats_flat)
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            roi_feats=roi_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """bbox head 训练阶段 forward + loss。"""
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox,
                            rois=rois,
                            bbox_targets=bbox_targets,
                            sampling_results=sampling_results)
        return bbox_results

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
    ):
        """含对比学习分支的训练 forward。"""
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(len(img_metas))]

        # === assign & sample proposals ===
        sampling_results = []
        for i in range(len(img_metas)):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            if gt_bboxes[i].numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes[i].new((0, gt_bboxes[0].size(-1))).zero_()
            else:
                sampling_result.pos_gt_bboxes = \
                    gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
            sampling_results.append(sampling_result)

        losses = dict()

        # === box head ===
        bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)
        losses.update(bbox_results['loss_bbox'])

        # === 对比学习 head (weak class + multi-scale) ===
        if self.contrastive_head is not None:
            # 每个 scale 的 roi feats（有的 FPN 有多个）
            if isinstance(bbox_results['roi_feats'], torch.Tensor):
                roi_feats_per_scale = [bbox_results['roi_feats']]
            else:
                roi_feats_per_scale = bbox_results['roi_feats']

            # 来自 sampling_results 的 ROI 标签
            roi_labels = []
            roi_ious = []
            roi_probs = []

            device = roi_feats_per_scale[0].device
            for res in sampling_results:
                num_pos = len(res.pos_bboxes)
                num_neg = len(res.neg_bboxes)
                # label: 正样本类号 + -1 对应背景
                label_full = torch.full((num_pos + num_neg,), -1, dtype=torch.long, device=device)
                if num_pos > 0:
                    label_full[:num_pos] = res.pos_gt_labels.to(device)
                roi_labels.append(label_full)

                # IoU：assigner计算时的 overlaps 一般在 res.assign_result 中
                if hasattr(res, 'max_overlaps'):
                    ious_this = res.max_overlaps
                else:
                    ious_this = torch.cat([
                        res.pos_assigned_gt_inds.new_zeros((num_pos,)),
                        res.pos_assigned_gt_inds.new_zeros((num_neg,))
                    ])
                roi_ious.append(ious_this.to(device))

                # roi_probs：从 cls_score softmax
                if 'cls_score' in bbox_results:
                    cls_probs = bbox_results['cls_score'].softmax(dim=1)
                    roi_probs.append(cls_probs.detach())

            roi_labels = torch.cat(roi_labels, dim=0) if len(roi_labels) > 0 else torch.empty((0,), dtype=torch.long, device=device)
            roi_ious = torch.cat(roi_ious, dim=0) if len(roi_ious) > 0 else torch.empty((0,), dtype=torch.float32, device=device)
            roi_probs = torch.cat(roi_probs, dim=0) if len(roi_probs) > 0 else None

            c_outputs = self.contrastive_head.forward(
                roi_feats_per_scale=roi_feats_per_scale,
                roi_labels=roi_labels,
                roi_probs=roi_probs,
                roi_ious=roi_ious,
                epoch=epoch)

            losses.update({
                'loss_cont_total': c_outputs['loss_contrastive_total'],
                'loss_ms_consist': c_outputs['loss_ms_consist'],
                'loss_supcon': c_outputs['loss_supcon'],
                'loss_proto': c_outputs['loss_proto'],
            })

        return (losses, sampling_results) if return_sampling_results else losses

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """标准推理阶段：输出检测框。"""
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
                bbox_pred = self.bbox_head.bbox_pred_split(bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

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