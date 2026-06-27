from mmcv.runner import auto_fp16
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors import OrientedRCNN
from mmrotate.models import build_neck
import torch
import torch.nn.functional as F

@ROTATED_DETECTORS.register_module()
class MyCustomDetector_SGM_CAMM(OrientedRCNN):
    def __init__(self,
                 backbone, neck, rpn_head, roi_head,
                 train_cfg=None, test_cfg=None, pretrained=None,
                 sgm_module=None, camm_module=None,
                 **kwargs):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.sgm_module = build_neck(sgm_module) if sgm_module is not None else None
        self.camm_module = build_neck(camm_module) if camm_module is not None else None
        self.fp16_enabled = False

    @auto_fp16(apply_to=("img",))
    def extract_feat(self, img, img_metas=None, gt_bboxes=None, gt_labels=None, return_loss=False):
        x = self.backbone(img)
        fpn_feats = self.neck(x)
        sgm_feats = self.sgm_module(fpn_feats) if self.sgm_module is not None else fpn_feats
        if self.camm_module is not None:
            # 训练时应传入gt_bboxes/gt_labels，启用loss分支
            camm_out = self.camm_module(
                sgm_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                return_loss=return_loss
            )
            camm_feats = camm_out['feats']     # list [P3~P7]
            camm_masks = camm_out.get('masks', None)
            g_camm = camm_out.get('g_camm', None)
            camm_losses = camm_out.get('loss', {}) # <--- 补充
        else:
            camm_feats = sgm_feats
            camm_masks = None
            g_camm = None
            camm_losses = {}
        return camm_feats, camm_masks, g_camm, camm_losses

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        losses = {}
        # 保证训练时 return_loss=True
        camm_feats, camm_masks, g_camm, camm_losses = self.extract_feat(
            img, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)

        # 添加camm的loss
        if camm_losses is not None and len(camm_losses) > 0:
            losses.update(camm_losses)

        feats_for_det = camm_feats
        rpn_losses, proposals = self.rpn_head.forward_train(
            feats_for_det, img_metas, gt_bboxes, gt_bboxes_ignore)
        losses.update(rpn_losses)
        roi_losses = self.roi_head.forward_train(
            feats_for_det, img_metas, proposals, gt_bboxes, gt_labels,
            gt_bboxes_ignore,
        )
        losses.update(roi_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        camm_feats, camm_masks, g_camm, _ = self.extract_feat(img, img_metas=img_metas, return_loss=False)
        feats = camm_feats
        proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        results = self.roi_head.simple_test(
            feats, proposal_list, img_metas, rescale=rescale,
        )
        return results
