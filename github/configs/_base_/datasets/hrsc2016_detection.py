data_root = 'data/'
angle_version = 'le90'
dataset_type = 'HRSCDataset'
#classes = ('ship',)
classes = ('ship', 'aircraft carrier', 'warcraft', 'merchant ship',
                    'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
                    'Perry', 'Sanantonio', 'Ticonderoga', 'Kitty Hawk',
                    'Kuznetsov', 'Abukuma', 'Austen', 'Tarawa', 'Blue Ridge',
                    'Container', 'OXo|--)', 'Car carrier([]==[])',
                    'Hovercraft', 'yacht', 'CntShip(_|.--.--|_]=', 'Cruise',
                    'submarine', 'lute', 'Medical', 'Car carrier(======|',
                    'Ford-class', 'Midway-class', 'Invincible-class')

#classes = ('ship', 'Nimitz', 'Enterprise', 'Arleigh Burke', 'WhidbeyIsland',
#                    'Perry', 'Sanantonio', 'Ticonderoga',  'Austen', 'Tarawa', 
#                    'Container', 'OXo|--)', 'Car carrier([]==[])',
#                    'Hovercraft',  'CntShip(_|.--.--|_]=', 'Cruise',
#                    'submarine', 'lute', 'Medical', 'Car carrier(======|',
#                    'Midway-class')
num_classes = 31
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PhotoMetricDistortion',  # 新增光度 distortion
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomRotate',  # 新增随机旋转
        prob=0.6,
        angle_range=30), 
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    


    
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
    type=dataset_type,
    ann_file=data_root + '/ImageSets/trainval.txt',
    ann_subdir='Annotations',
    img_prefix=data_root + 'Train/JPEGImages/',  # 正确路径
    pipeline=train_pipeline,
    classwise=True,
    classes=classes,),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/test.txt',
        img_prefix=data_root + 'Test/JPEGImages/',
        ann_subdir='Annotations',
        classes=classes,
        pipeline=test_pipeline,
        classwise=True,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/trainval.txt',
        img_prefix=data_root + 'Train/JPEGImages/',
        ann_subdir='Annotations',
        classes=classes,
        pipeline=test_pipeline,
        classwise=True,
    ),
)



train_loop = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)