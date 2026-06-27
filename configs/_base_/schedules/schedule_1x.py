# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(
    type='AdamW',  # 改用AdamW优化器
    lr=0.00013,    # 初始学习率
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # 骨干网络更低的学习率
            'sgm_module': dict(lr_mult=0.7)  # SGM模块更高的学习率
        }))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',  # 改用余弦退火
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    min_lr=1e-7)
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=5)
