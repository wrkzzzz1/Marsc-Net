# dataset settings
dataset_type = 'HRSCDataset'
data_root = 'data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + '/ImageSets/trainval.txt',
        ann_subdir='Annotations',          # 只写相对路径
        img_subdir='JPEGImages',           # 只写相对路径
        img_prefix=data_root + '/Train',   # 只写根路径
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + '/ImageSets/trainval.txt',
        ann_subdir='Annotations',
        img_subdir='JPEGImages',
        img_prefix=data_root + '/Train',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + '/ImageSets/test.txt',
        ann_subdir='Annotations',
        img_subdir='JPEGImages',
        img_prefix=data_root + '/Test',
        pipeline=test_pipeline),
)