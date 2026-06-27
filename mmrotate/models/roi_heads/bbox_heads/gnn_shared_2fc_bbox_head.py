from mmrotate.models.roi_heads.bbox_heads.convfc_rbbox_head import RotatedShared2FCBBoxHead
from mmdet.models.builder import HEADS

@HEADS.register_module()
class GNNShared2FCBBoxHead(RotatedShared2FCBBoxHead):
    def forward(self, x):
        for fc in self.shared_fcs:
            x = self.relu(fc(x))
        cls_score = self.fc_cls(x)
        bbox_pred = self.fc_reg(x)
        return cls_score, bbox_pred