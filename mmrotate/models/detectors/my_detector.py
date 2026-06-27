import torch
from mmcv.runner import auto_fp16
from mmrotate.models import ROTATED_NECKS
from mmrotate.models.detectors import OrientedRCNN
from mmdet.models import DETECTORS

@DETECTORS.register_module()
class MyCustomDetector_SGM_CAMM_GNN_Con(OrientedRCNN):
    def __init__(self, backbone, neck, rpn_head, roi_head,
                 train_cfg=None, test_cfg=None, pretrained=None,
                 sgm_module=None, camm_module=None, gnn_module=None, **kwargs):
        super().__init__(backbone=backbone, neck=neck, rpn_head=rpn_head,
                         roi_head=roi_head, train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained)
        self.sgm_module = ROTATED_NECKS.build(sgm_module) if sgm_module is not None and isinstance(sgm_module, dict) else sgm_module
        self.camm_module = ROTATED_NECKS.build(camm_module) if camm_module is not None and isinstance(camm_module, dict) else camm_module
        self.gnn_module = ROTATED_NECKS.build(gnn_module) if gnn_module is not None and isinstance(gnn_module, dict) else gnn_module

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, gt_bboxes=None, gt_labels=None):
        x = self.backbone(img)
        fpn_feats = self.neck(x)
        if self.sgm_module is not None:
            fpn_feats = self.sgm_module(fpn_feats)
        camm_loss = {}
        g_camm = None
        if self.camm_module is not None:
            camm_out = self.camm_module(
                fpn_feats, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=self.training
            )
            feats = camm_out['feats']
            camm_loss = camm_out.get('loss', {})
            g_camm = camm_out.get('g_camm', None)
        else:
            feats = fpn_feats

        return {'feats': feats, 'camm_loss': camm_loss, 'g_camm': g_camm}

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        feat_dict = self.extract_feat(img, img_metas, gt_bboxes, gt_labels)
        feats = feat_dict['feats']
        losses = {}
        if feat_dict.get('camm_loss'):
            losses.update(feat_dict['camm_loss'])

        rpn_losses, proposals = self.rpn_head.forward_train(
            feats, img_metas, gt_bboxes, gt_bboxes_ignore)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(
            feats, img_metas, proposals, gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses.update(roi_losses)

        if self.gnn_module is not None and feat_dict['g_camm'] is not None:
            proposal_tensors = [p for p in proposals if (p is not None and p.numel() > 0)]
            if len(proposal_tensors) == 0:
                return losses
            rois = torch.cat(proposal_tensors, 0)
            if rois.numel() == 0:
                return losses
            batch_size = len(feats[0]) if (isinstance(feats, list) and isinstance(feats[0], torch.Tensor)) else img.shape[0]
            if rois[:, 0].max().item() >= batch_size:
                rois[:, 0] = rois[:, 0] % batch_size
            roi_feats = self.roi_head.bbox_roi_extractor(feats, rois)
            roi_batch_inds = rois[:, 0].to(torch.long)
            if isinstance(roi_feats, (list, tuple)):
                roi_feats = torch.cat(roi_feats, dim=0)
            if isinstance(roi_batch_inds, (list, tuple)):
                roi_batch_inds = torch.cat(roi_batch_inds, dim=0)
            g_camm = feat_dict['g_camm']
            if isinstance(g_camm, (list, tuple)):
                g_camm = torch.cat(g_camm, dim=0)
            if g_camm.dim() > 2:
                g_camm = g_camm.view(g_camm.shape[0], -1)
            assert roi_feats.dim() == 4 or roi_feats.dim() == 2
            assert roi_batch_inds.dim() == 1
            assert g_camm.dim() == 2
            gnn_out = self.gnn_module(roi_feats, roi_batch_inds)
            if isinstance(gnn_out, dict) and 'loss_cls' in gnn_out and gnn_out['loss_cls'] is not None:
                losses["gnn_loss_cls"] = gnn_out['loss_cls']
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        feat_dict = self.extract_feat(img)
        feats = feat_dict['feats']
        proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        results = self.roi_head.simple_test(feats, proposal_list, img_metas, rescale=rescale)
        return results
