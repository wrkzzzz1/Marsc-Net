from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors import OrientedRCNN
from mmcv.runner import auto_fp16


@ROTATED_DETECTORS.register_module()
class MyCustomDetector(OrientedRCNN):
    def __init__(self, 
                 backbone, 
                 neck, 
                 rpn_head, 
                 roi_head, 
                 train_cfg=None, 
                 test_cfg=None, 
                 pretrained=None,
                 sgm_module=None,  # SGM模块
                 **kwargs):  # 👈 这行很关键，能接收掉“type”这种多余参数
        super(MyCustomDetector, self).__init__(backbone=backbone, 
                                              neck=neck, 
                                              rpn_head=rpn_head, 
                                              roi_head=roi_head,
                                              train_cfg=train_cfg, 
                                              test_cfg=test_cfg, 
                                              pretrained=pretrained)
        
        from mmrotate.models import build_neck

        if sgm_module is not None:
            self.sgm_module = build_neck(sgm_module)
        else:
            self.sgm_module = None

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        if self.sgm_module is not None:
            x = self.sgm_module(x)
        return x
