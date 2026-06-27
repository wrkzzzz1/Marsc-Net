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
        ann_file='data/ImageSets/trainval.txt',
        img_prefix='data/Train/JPEGImages/',
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
train_loop = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=6)
evaluation = dict(interval=6, metric='mAP', iou_thr=0.5)
optimizer = dict(type='SGD', lr=0.0004, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,
    warmup_ratio=0.001,
    step=[15, 20],
    gamma=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=6)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='RotatedFCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=31,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=True,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        h_bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_angle=dict(type='L1Loss', loss_weight=0.2),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))
work_dir = '/root/mmdetection/work_dirs/rotated_fcos'
auto_resume = False
gpu_ids = range(0, 1)
seed = 4
deterministic = True