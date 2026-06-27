# Copyright (c) OpenMMLab. All rights reserved.
from .base import RotatedBaseDetector
from .gliding_vertex import GlidingVertex
from .oriented_rcnn import OrientedRCNN
from .r3det import R3Det
from .redet import ReDet
from .roi_transformer import RoITransformer
from .rotate_faster_rcnn import RotatedFasterRCNN
from .rotated_fcos import RotatedFCOS
from .rotated_reppoints import RotatedRepPoints
from .rotated_retinanet import RotatedRetinaNet
from .s2anet import S2ANet
from .single_stage import RotatedSingleStageDetector
from .two_stage import RotatedTwoStageDetector
from .my_custom_detector_SGM import MyCustomDetector

from .my_detector_SGM_CAMM import MyCustomDetector_SGM_CAMM

from .my_detector_SGM_CAMM_GNN import MyCustomDetector_SGM_CAMM_GNN

from .my_detector_SGM_CAMM_GNN_Constrastive import MyCustomDetector_SGM_CAMM_GNN_Con

__all__ = [
    'RotatedRetinaNet', 'RotatedFasterRCNN', 'OrientedRCNN', 'RoITransformer',
    'GlidingVertex', 'ReDet', 'R3Det', 'S2ANet', 'RotatedRepPoints',
    'RotatedBaseDetector', 'RotatedTwoStageDetector',
    'RotatedSingleStageDetector', 'RotatedFCOS',
    'MyCustomDetector',
    'MyCustomDetector_SGM_CAMM',
    'MyCustomDetector_SGM_CAMM_GNN',
    'MyCustomDetector_SGM_CAMM_GNN_Con'
]
