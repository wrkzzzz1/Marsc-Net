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
    samples_per_gpu=4,
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
train_loop = dict(type='EpochBasedTrainLoop', max_epochs=120, val_interval=10)
evaluation = dict(interval=10, metric='mAP', iou_thr=0.5)
optimizer = dict(
    type='AdamW',
    lr=0.000125,
    betas=(0.9, 0.999),
    weight_decay=0.02,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1))))
runner = dict(type='EpochBasedRunner', max_epochs=120)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=3e-05,
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=0.001)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = '/root/mmdetection/work_dirs/rotated_reppoints/epoch_80.pth'
load_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
contrast_size = 512
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='RotatedRepPoints',
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
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    bbox_head=dict(
        type='RotatedRepPointsHead',
        num_classes=31,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.375),
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),
        transform_method='rotrect',
        use_reassign=False,
        topk=6,
        anti_factor=0.75),
    train_cfg=dict(
        init=dict(
            assigner=dict(type='ConvexAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxConvexIoUAssigner',
                pos_iou_thr=0.45,
                neg_iou_thr=0.4,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.005,
        nms=dict(iou_thr=0.2),
        max_per_img=300))
work_dir = '/root/mmdetection/work_dirs/rotated_reppoints'
auto_resume = False
gpu_ids = range(0, 1)
seed = 4
deterministic = True
custom_hooks = [dict(type='EpochSyncHook')]
