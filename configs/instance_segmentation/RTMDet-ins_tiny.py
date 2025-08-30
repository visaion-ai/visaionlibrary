"""
This config is for the RTMDet-ins_tiny model.
comment explaination:
auto config -- the config should be set internally by the invoker
manual config -- the config should be set by the user
"""
default_scope = 'mmdet'
# model config ----------------------------------------------------------------------------------------
in_channels = 3  # auto config
# in object detection and instance segmentation, the number of classes is exclusive of background, due to classification loss (eg. focal loss)
num_classes = 1  # auto config
batch_size = 8  # manual config | this is for a single GPU
load_from = '/root/projects/visaionlib/pretrained_weights/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'

model = dict(
    type='RTMDet',
    _scope_="mmdet",
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        _scope_="mmengine",
        batch_augments=None,
        bgr_to_rgb=False,
        grayscale_to_color=True,
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        pad_size_divisor=32
    ),
    backbone=dict(
        type='CSPNeXt',
        _scope_="mmdet",
        arch='P5',
        act_cfg=dict(type='SiLU', inplace=True),
        channel_attention=True,
        widen_factor=0.375,
        deepen_factor=0.167,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        _scope_="mmdet",
        act_cfg=dict(type='SiLU', inplace=True),
        expand_ratio=0.5,
        in_channels=[96, 192, 384],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=1,
        out_channels=96,
    ),
    bbox_head=dict(
        type='VisaionRTMDetInsSepBNHead',  # use VisaionRTMDetInsSepBNHead instead of RTMDetInsSepBNHead due to the bug of RTMDetInsSepBNHead
        act_cfg=dict(type='SiLU', inplace=True),
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0, 
            strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        feat_channels=96,
        in_channels=96,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            beta=2.0,
            loss_weight=1.0,
            use_sigmoid=True
        ),
        loss_mask=dict(
            type='DiceLoss',
            eps=5e-06, 
            loss_weight=2.0,
            reduction='mean'
        ),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_classes=num_classes,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
    ),
    test_cfg=dict(
        mask_thr_binary=0.5,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.6),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        debug=False,
        pos_weight=-1),
    )

# data config ----------------------------------------------------------------------------------------
num_workers = 4  # set fixed
bg_ratio = 0.2  # manual config
meta_info = dict(
    classes=("balloon"), palette=[(255, 0, 0)]
)  # auto config


data_root = '/root/data/balloon-train1'
color_type = "color"  # auto config
val_data_root = "/root/data/balloon-val1"  # auto config
test_data_root = "/root/data/balloon-val1"  # auto config

random_flip_prob = 0.5  # manual_config
crop_size = 640  # manual config

train_pipeline = [
        dict(
            type='LoadImageFromFile', 
            _scope_="mmdet", 
            color_type='color', 
            imdecode_backend='cv2'
        ),
        dict(
            type='LoadAnnotations',
            _scope_="mmdet",
            poly2mask=False,
            with_bbox=True,
            with_mask=True),
        dict(
            type='CachedMosaic', 
            _scope_="mmdet", 
            max_cached_images=40,
            img_scale=(crop_size, crop_size), 
            pad_val=114.0
        ),
        dict(
            type='RandomResize',
            _scope_="mmdet",
            scale=(crop_size*2, crop_size*2),
            ratio_range=(0.1, 2.0),
            keep_ratio=True),
        dict(
            type='RandomCrop',
            _scope_="mmdet",
            crop_size=(crop_size, crop_size),
            recompute_bbox=True,
            allow_negative_crop=True
        ),
        dict(type='YOLOXHSVRandomAug', _scope_="mmdet"),
        dict(type='RandomFlip', _scope_="mmdet", prob=random_flip_prob),
        dict(
            type='Pad', 
            _scope_="mmdet", 
            size=(crop_size, crop_size), 
            pad_val=dict(img=(114, 114, 114))
        ),
        dict(
            type='CachedMixUp',
            _scope_="mmdet",
            img_scale=(crop_size, crop_size),
            ratio_range=(1.0, 1.0),
            max_cached_images=20,
            pad_val=(114, 114, 114)),
        dict(type='FilterAnnotations', _scope_="mmdet", min_gt_bbox_wh=(1, 1)),
        dict(
            meta_keys=('img_id', 'img_path', 'ori_shape','instances', 'img_shape', 'scale_factor'),
            type='PackDetInputs', _scope_="mmdet"),
    ]

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        type='VisaionRTMDETInsDataset',
        _scope_="mmengine",
        metainfo=meta_info,
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=16),
        data_prefix=dict(img=""),  # "" must be set for right work
        pipeline=train_pipeline),
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type="VisaionInfiniteSampler",
        shuffle=True,
        bg_ratio=bg_ratio,
        batch_size=batch_size,
    ))

val_pipeline = [
    dict(
        type='LoadImageFromFile', 
        _scope_="mmdet", 
        color_type='color', 
        imdecode_backend='cv2'
    ),
    dict(
        type='Resize', 
        _scope_="mmdet", 
        scale=(crop_size, crop_size), 
        keep_ratio=True
    ),
    dict(
        type='Pad', 
        _scope_="mmdet", 
        size=(crop_size, crop_size), 
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(
        type='LoadAnnotations',
        _scope_="mmdet",
        with_bbox=True,
        with_mask=True,
        poly2mask=False
    ),
    dict(
        type='PackDetInputs',
        _scope_="mmdet",
        meta_keys=('img_id', 'img_path', 'ori_shape','instances', 'img_shape', 'scale_factor'),
    ),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='VisaionRTMDETInsDataset',
        _scope_="mmengine",
        metainfo=meta_info,
        data_root=val_data_root,
        data_prefix=dict(img=""),  # "" must be set for right work
        pipeline=val_pipeline,
        test_mode=True),
    drop_last=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler',shuffle=False))


test_pipeline = [
    dict(
        type='LoadImageFromFile', 
        _scope_="mmdet", 
        color_type='color', 
        imdecode_backend='cv2'
    ),
    dict(
        type='Resize', 
        _scope_="mmdet", 
        scale=(crop_size, crop_size), 
        keep_ratio=True
    ),
    dict(
        type='Pad', 
        _scope_="mmdet", 
        size=(crop_size, crop_size), 
        pad_val=dict(img=(114, 114, 114))
    ),
    dict(
        type='LoadAnnotations',
        _scope_="mmdet",
        with_bbox=True,
        with_mask=True,
        poly2mask=False
    ),
    dict(
        type='PackDetInputs',
        _scope_="mmdet",
        meta_keys=('img_id', 'img_path', 'ori_shape','instances', 'img_shape', 'scale_factor'),
    ),
]


test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='VisaionRTMDETInsDataset',
        _scope_="mmengine",
        metainfo=meta_info,
        data_root=test_data_root,
        data_prefix=dict(img=""),  # "" must be set for right work
        pipeline=test_pipeline,
        test_mode=True),
    drop_last=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler',shuffle=False))

# training config ----------------------------------------------------------------------------------------
launcher = 'pytorch'
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)

max_iters = 3000  # manual config
val_interval = 500  # manual config
_experiment_name = "test"  # auto config
log_print_interval = 50  # manual config

lr = 0.004  # manual config
weight_decay = 0.05  # manual config
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr,  weight_decay=weight_decay),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
)
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0, by_epoch=False, end=int(max_iters*0.05), start_factor=1e-05,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=max_iters,
        begin=int(max_iters*0.05),
        end=max_iters,
        by_epoch=False,        
        eta_min=lr*0.01,
    ),
]


train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=val_interval,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='CocoMetric',
    backend_args=None,
    format_only=False,
    metric=['bbox', 'segm'],
    proposal_nums=(100, 1, 10)
)
test_evaluator = [
    # dict(type='PrecisionAndRecallsMetricforVisaion', _scope_="mmengine"),
    dict(
        type='CocoMetric',
        backend_args=None,
        format_only=False,
        metric=['bbox', 'segm'],
        proposal_nums=(100, 1, 10)        
    )
]

visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

train_pipeline_stage2 = [
   dict(
        type='LoadImageFromFile', 
        _scope_="mmdet", 
        color_type='color', 
        imdecode_backend='cv2'
    ),
    dict(
        type='LoadAnnotations',
        _scope_="mmdet",
        with_bbox=True,
        with_mask=True,
        poly2mask=False
    ),
    dict(
        type='RandomResize',
        _scope_="mmdet",
        scale=(crop_size, crop_size),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        _scope_="mmdet",
        crop_size=(crop_size, crop_size),
        recompute_bbox=True,
        allow_negative_crop=True
    ),
    dict(type='FilterAnnotations', _scope_="mmdet", min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug', _scope_="mmdet"),
    dict(type='RandomFlip', _scope_="mmdet", prob=random_flip_prob),
    dict(type='Pad', _scope_="mmdet", size=(crop_size, crop_size), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', _scope_="mmdet", meta_keys=('img_id', 'img_path', 'ori_shape','instances', 'img_shape', 'scale_factor'))
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,        
        update_buffers=True,
        begin_iter=int(max_iters*0.8)
    ),
    dict(
        type='VisaionPipelineSwitchHook',
        _scope_="mmengine",
        switch_iter=int(max_iters*0.9),
        switch_pipeline=train_pipeline_stage2
    )
]
backend = "nccl"  # auto config
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        interval=val_interval, 
        max_keep_ckpts=3,
        by_epoch=False,
        filename_tmpl="checkpoint_{:06d}.pth",
        save_last=True,
    ),
    logger=dict(
        type="LoggerHook",
        interval=log_print_interval,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend=backend),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
randomness = dict(seed=1024, diff_rank_seed=False, deterministic=False)
work_dir = '/root/projects/visaionlib/workspace/tasks/coco_ins/rtmdet-ins_tiny_8xb32-300e_coco/test_result'