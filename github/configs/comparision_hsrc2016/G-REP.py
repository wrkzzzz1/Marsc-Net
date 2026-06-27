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
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='PolyRandomRotate',
         rotate_ratio=0.5,
         mode='value',
         angles_range=[0, 90, 180, 270],
         auto_bound=True,  # 关键
         version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
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
        classes=classes,
        version='le90'),
    test=dict(
        type='HRSCDataset',
        ann_file='data/ImageSets/test.txt',
        img_prefix='data/Test/JPEGImages/',
        ann_subdir='Annotations',
        classes=classes,
        pipeline=test_pipeline,
        classwise=True,
        version='le90'),
    val=dict(
        type='HRSCDataset',
        ann_file='data/ImageSets/trainval.txt',
        img_prefix='data/Train/JPEGImages/',
        ann_subdir='Annotations',
        classes=classes,
        pipeline=test_pipeline,
        classwise=True,
        version='le90')
)  # 这里结束 data，原来你是 ) )
train_loop = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=6)
evaluation = dict(interval=6, metric='mAP', iou_thr=0.5)
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1))))
runner = dict(type='EpochBasedRunner', max_epochs=24)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[12, 20],
    gamma=0.5)
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
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='KLDRepPointsHead',
        num_classes=num_classes,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=1.0,
            alpha=0.25, loss_weight=1.0),
        loss_bbox_init=dict(type='KLDRepPointsLoss'),
        loss_bbox_refine=dict(type='KLDRepPointsLoss'),
        transform_method='rotrect',
        use_reassign=False,
        topk=6,
        anti_factor=0.75
    ),
    train_cfg=dict(
        init=dict(
            assigner=dict(
                type='ConvexAssigner',
                scale=4,
                pos_num=1
            ),
            allowed_border=-1,
            allowed_empty=True,       # ✅ 新增这个
            pos_weight=-1,
            debug=False
        ),
        refine=dict(
            assigner=dict(
                type='ATSSKldAssigner',
                topk=9,
                use_reassign=False
            ),
            allowed_border=-1,
            allowed_empty=True,       # ✅ 新增这个
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000)
)  

work_dir = '/root/mmdetection/work_dirs/G-REP'
auto_resume = False
gpu_ids = range(0, 1)
seed = 4
deterministic = True
