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
                version=angle_version),
            dict(type='Normalize', **img_norm_cfg),
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
train_loop = dict(type='EpochBasedTrainLoop', max_epochs=160, val_interval=1)
evaluation = dict(interval=1, metric='mAP', iou_thr=0.5)
optimizer = dict(
    type='AdamW',
    lr=0.00025,
    betas=(0.9, 0.999),
    weight_decay=0.02,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1))))
runner = dict(type='EpochBasedRunner', max_epochs=160)
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
resume_from = '/root/mmdetection/work_dirs/r3det-fixed/epoch_130.pth'
load_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='R3Det',
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
        num_outs=5),
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=31,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90',
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    frm_cfgs=[
        dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128]),
        dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128])
    ],
    num_refine_stages=2,
    refine_heads=[
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=31,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='le90',
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=31,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='le90',
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0))
    ],
    train_cfg=dict(
        s0=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.25,
                ignore_iof_thr=-1,
                match_low_quality=True,  # 从rpn拷过来
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        sr=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.45,
                    match_low_quality=False,  # 从rpn拷过来
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0, 1.0]),
    test_cfg=dict(
        s0=dict(
            nms_pre=3000,
            score_thr=0.005,
            nms=dict(type='nms', iou_thr=0.7),
            max_per_img=100),
        sr=[
            dict(
                nms_pre=2000,
                score_thr=0.005,
                nms=dict(type='nms', iou_thr=0.2),
                max_per_img=100),
            dict(
                nms_pre=2000,
                score_thr=0.005,
                nms=dict(type='nms', iou_thr=0.2),
                max_per_img=100)
        ]))
work_dir = '/root/mmdetection/work_dirs/r3det-fixed'
auto_resume = False
gpu_ids = range(0, 1)
seed = 4
deterministic = True
