data_root = 'data/'
angle_version = 'le90'
dataset_type = 'HRSCDataset'
classes = ('ship', 'aircraft carrier', 'warcraft', 'merchant ship', 'Nimitz',
           'Enterprise', 'Arleigh Burke', 'WhidbeyIsland', 'Perry',
           'Sanantonio', 'Ticonderoga', 'Kitty Hawk', 'Kuznetsov', 'Abukuma',
           'Austen', 'Tarawa', 'Blue Ridge', 'Container', 'OXo|--)',
           'Car carrier([]==[])', 'Hovercraft', 'yacht',
           'CntShip(_|.--.--|_]=', 'Cruise', 'submarine', 'lute', 'Medical',
           'Car carrier(======|', 'Ford-class', 'Midway-class',
           'Invincible-class')
num_classes = 31
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='RandomRotate', prob=0.5, angle_range=180, version='le90'),
    dict(type='PhotoMetricDistortion')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='HRSCDataset',
        ann_file='data//ImageSets/trainval.txt',
        ann_subdir='Annotations',
        img_prefix='data/Train/JPEGImages/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(1024, 1024)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version='le90'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classwise=True,
        classes=('ship', 'aircraft carrier', 'warcraft', 'merchant ship',
                 'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
                 'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty Hawk',
                 'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue Ridge',
                 'Container', 'OXo|--)', 'Car carrier([]==[])', 'Hovercraft',
                 'yacht', 'CntShip(_|.--.--|_]=', 'Cruise', 'submarine',
                 'lute', 'Medical', 'Car carrier(======|', 'Ford-class',
                 'Midway-class', 'Invincible-class'),
        version='le90'),
    test=dict(
        type='HRSCDataset',
        ann_file='data/ImageSets/test.txt',
        img_prefix='data/Test/JPEGImages/',
        ann_subdir='Annotations',
        classes=('ship', 'aircraft carrier', 'warcraft', 'merchant ship',
                 'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
                 'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty Hawk',
                 'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue Ridge',
                 'Container', 'OXo|--)', 'Car carrier([]==[])', 'Hovercraft',
                 'yacht', 'CntShip(_|.--.--|_]=', 'Cruise', 'submarine',
                 'lute', 'Medical', 'Car carrier(======|', 'Ford-class',
                 'Midway-class', 'Invincible-class'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classwise=True,
        version='le90'),
    val=dict(
        type='HRSCDataset',
        ann_file='data/ImageSets/test.txt',
        img_prefix='data/Test/JPEGImages/',
        ann_subdir='Annotations',
        classes=('ship', 'aircraft carrier', 'warcraft', 'merchant ship',
                 'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
                 'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty Hawk',
                 'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue Ridge',
                 'Container', 'OXo|--)', 'Car carrier([]==[])', 'Hovercraft',
                 'yacht', 'CntShip(_|.--.--|_]=', 'Cruise', 'submarine',
                 'lute', 'Medical', 'Car carrier(======|', 'Ford-class',
                 'Midway-class', 'Invincible-class'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classwise=True,
        version='le90'))
train_loop = dict(type='EpochBasedTrainLoop', max_epochs=70, val_interval=1)
evaluation = dict(interval=1, metric='mAP', iou_thr=0.5)
optimizer = dict(
    type='AdamW',
    lr=0.00025,
    betas=(0.9, 0.999),
    weight_decay=0.02,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1))))
runner = dict(type='EpochBasedRunner', max_epochs=70)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=3e-05,
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=0.001)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = '/root/mmdetection/work_dirs/custom_cfg_new_ablation-study-24turns-redet/epoch_50.pth'
load_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='ReDet',
    backbone=dict(
        type='ReResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        pretrained=
        '/root/mmdetection/work_dirs/re_resnet50_c8_batch256-25b16846.pth'),
    neck=dict(
        type='ReFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RotatedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='le90',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='RoITransRoIHead',
        version='le90',
        num_stages=2,
        stage_loss_weights=[1, 1],
        bbox_roi_extractor=[
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RiRoIAlignRotated',
                    out_size=7,
                    num_samples=2,
                    num_orientations=8,
                    clockwise=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
        ],
        bbox_head=[
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=31,
                bbox_coder=dict(
                    type='DeltaXYWHAHBBoxCoder',
                    angle_range='le90',
                    norm_factor=2,
                    edge_swap=True,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=31,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range='le90',
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.05]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.25,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.45,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.45,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.005,
            nms=dict(iou_thr=0.2),
            max_per_img=300)))
work_dir = '/root/mmdetection/work_dirs/custom_cfg_new_ablation-study-24turns-redet'
auto_resume = False
gpu_ids = range(0, 1)
seed = 4
deterministic = True
custom_hooks = [dict(type='EpochSyncHook')]
