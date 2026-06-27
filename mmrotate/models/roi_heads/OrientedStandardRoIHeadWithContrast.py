import torch

from mmrotate.core import rbbox2roi
from mmrotate.models import ROTATED_HEADS, ROTATED_NECKS
from .rotate_standard_roi_head import RotatedStandardRoIHead


@ROTATED_HEADS.register_module()
class OrientedStandardRoIHeadWithContrast(RotatedStandardRoIHead):
    """
    Oriented RoI head with ASGA and DySC.

    Detection loss:
        L_det

    Contrastive loss:
        L_contrast = lambda_proto L_proto + lambda_sup L_sup + lambda_ms L_ms
    """

    def __init__(
        self,
        *args,
        contrastive_head=None,
        gnn_module=None,
        **kwargs
    ):
        super(OrientedStandardRoIHeadWithContrast, self).__init__(*args, **kwargs)

        self.contrastive_head = (
            ROTATED_HEADS.build(contrastive_head)
            if contrastive_head is not None else None
        )

        self.gnn_module = (
            ROTATED_NECKS.build(gnn_module)
            if isinstance(gnn_module, dict) else gnn_module
        )

        self.current_epoch = 0

    def forward_dummy(self, x, proposals):
        outs = ()

        rois = rbbox2roi([proposals])

        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (
                bbox_results['cls_score'],
                bbox_results['bbox_pred']
            )

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
        """
        Args:
            x:
                Multi-scale feature maps.
            img_metas:
                Image meta information.
            proposal_list:
                Proposals of each image.
            gt_bboxes:
                Ground-truth bounding boxes.
            gt_labels:
                Ground-truth labels.
        """

        if epoch is not None:
            self.current_epoch = int(epoch)
        elif len(img_metas) > 0 and 'epoch' in img_metas[0]:
            self.current_epoch = int(img_metas[0]['epoch'])

        if self.contrastive_head is not None:
            self.contrastive_head.set_epoch(self.current_epoch)

        num_imgs = len(img_metas)

        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]

        sampling_results = []

        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i],
                gt_bboxes[i],
                gt_bboxes_ignore[i],
                gt_labels[i]
            )

            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x]
            )

            if gt_bboxes[i].numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes[i].new_zeros(
                    (0, gt_bboxes[0].size(-1))
                )
            else:
                sampling_result.pos_gt_bboxes = gt_bboxes[i][
                    sampling_result.pos_assigned_gt_inds,
                    :
                ]

            sampling_results.append(sampling_result)

        losses = dict()

        if self.with_bbox:
            bbox_results = self._bbox_forward_train(
                x,
                sampling_results,
                gt_bboxes,
                gt_labels,
                img_metas
            )
            losses.update(bbox_results['loss_bbox'])

        if self.contrastive_head is not None and self.gnn_module is not None:
            contrastive_losses = self._contrastive_forward_train(
                x,
                sampling_results,
                epoch=self.current_epoch
            )
            losses.update(contrastive_losses)

        if return_sampling_results:
            return losses, sampling_results

        return losses

    def _bbox_forward_train(
        self,
        x,
        sampling_results,
        gt_bboxes,
        gt_labels,
        img_metas
    ):
        rois = rbbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(
            sampling_results,
            gt_bboxes,
            gt_labels,
            self.train_cfg
        )

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def _bbox_forward(self, x, rois):
        roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois
        )

        roi_feats_flat = roi_feats.flatten(1)

        cls_score, bbox_pred = self.bbox_head(roi_feats_flat)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            roi_feats=roi_feats
        )

        return bbox_results

    def _build_roi_labels(self, sampling_results):
        """
        Build RoI labels for all sampled RoIs.

        Positive RoIs use their GT labels.
        Negative RoIs are assigned -1 and ignored in DySC.
        """

        roi_labels_list = []

        for res in sampling_results:
            num_rois = res.bboxes.size(0)

            labels = torch.full(
                (num_rois,),
                -1,
                dtype=torch.long,
                device=res.bboxes.device
            )

            if hasattr(res, 'pos_inds') and res.pos_inds.numel() > 0:
                labels[res.pos_inds] = res.pos_gt_labels

            roi_labels_list.append(labels)

        roi_labels = torch.cat(roi_labels_list, dim=0)

        return roi_labels

    def _contrastive_forward_train(
        self,
        x,
        sampling_results,
        epoch=None
    ):
        """
        Extract multi-scale RoI embeddings from ASGA and compute DySC loss.
        """

        rois = rbbox2roi([res.bboxes for res in sampling_results])

        if rois.numel() == 0:
            zero = x[0].new_zeros(())
            return dict(
                loss_contrastive_total=zero,
                loss_ms_consist=zero,
                loss_supcon=zero,
                loss_proto=zero
            )

        batch_inds = rois[:, 0].long()

        num_scales = min(
            len(x),
            getattr(self.contrastive_head, 'num_scales', len(x))
        )

        roi_feats_per_scale = []

        for lvl in range(num_scales):
            roi_feat = self.bbox_roi_extractor(
                [x[lvl]],
                rois
            )

            gnn_out = self.gnn_module(
                roi_feat,
                batch_inds=batch_inds
            )

            if 'g_fused' in gnn_out:
                graph_feat = gnn_out['g_fused']
            elif 'g_roi' in gnn_out:
                graph_feat = gnn_out['g_roi']
            else:
                raise KeyError(
                    "ASGA output should contain 'g_fused' or 'g_roi'."
                )

            roi_feats_per_scale.append(graph_feat)

        roi_labels = self._build_roi_labels(sampling_results)

        contrastive_losses = self.contrastive_head(
            roi_feats_per_scale=roi_feats_per_scale,
            roi_labels=roi_labels,
            epoch=epoch
        )

        return contrastive_losses

    def simple_test_bboxes(
        self,
        x,
        img_metas,
        proposals,
        rcnn_test_cfg,
        rescale=False
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
                    bbox_pred,
                    num_proposals_per_img
                )
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
                cfg=rcnn_test_cfg
            )

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels
