from mmdet.models import BACKBONES, NECKS, ROI_EXTRACTORS, SHARED_HEADS, HEADS, LOSSES, DETECTORS
import warnings
ROTATED_BACKBONES = BACKBONES
ROTATED_NECKS = NECKS
ROTATED_ROI_EXTRACTORS = ROI_EXTRACTORS
ROTATED_SHARED_HEADS = SHARED_HEADS
ROTATED_HEADS = HEADS
ROTATED_LOSSES = LOSSES
ROTATED_DETECTORS = DETECTORS

def build_backbone(cfg):
    return ROTATED_BACKBONES.build(cfg)

def build_neck(cfg):
    return ROTATED_NECKS.build(cfg)

def build_roi_extractor(cfg):
    return ROTATED_ROI_EXTRACTORS.build(cfg)

def build_shared_head(cfg):
    return ROTATED_SHARED_HEADS.build(cfg)

def build_head(cfg):
    return ROTATED_HEADS.build(cfg)

def build_loss(cfg):
    return ROTATED_LOSSES.build(cfg)

def build_detector(cfg, train_cfg=None, test_cfg=None):
    if train_cfg is not None or test_cfg is not None:
        warnings.warn('train_cfg and test_cfg is deprecated, '
                      'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None
    assert cfg.get('test_cfg') is None or test_cfg is None
    return ROTATED_DETECTORS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

print("Registered detectors:", ROTATED_DETECTORS.module_dict.keys())