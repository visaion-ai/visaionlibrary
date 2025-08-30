"""
This config is for the YOLOv5-det_s model.
comment explaination:
auto config -- the config should be set internally by the invoker
manual config -- the config should be set by the user
"""
default_scope = 'mmyolo'
# model config ----------------------------------------------------------------------------------------
in_channels = 3  # auto config
# in object detection and instance segmentation, the number of classes is exclusive of background, due to classification loss (eg. focal loss)
num_classes = 4  # auto config
batch_size = 8  # manual config | this is for a single GPU
load_from = '/data/projects/visaionlib/pretrained_weights/yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230304_033134-8e0cd271.pth'

crop_size = 640  # manual config
anchor_scale = crop_size / 640

deepen_factor = 0.33
widen_factor = 0.5

model = dict(
    type='YOLODetector',
    _scope_="mmyolo",
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        _scope_="mmyolo",
        batch_augments=None,
        bgr_to_rgb=True,
        mean=[0.0],
        std=[255.0],
        pad_size_divisor=32
    ),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        _scope_="mmyolo",
        act_cfg=dict(type='SiLU', inplace=True),
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='SyncBN'),
        input_channels=in_channels,
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        _scope_="mmyolo",
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=3,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024]
    ),
    bbox_head=dict(
        type='YOLOv5Head',
        _scope_="mmyolo",
        head_module=dict(
            type='YOLOv5HeadModule',
            featmap_strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            num_base_priors=3,
            num_classes=num_classes,
            widen_factor=widen_factor),
        loss_bbox=dict(
            bbox_format='xywh',
            eps=1e-07,
            iou_mode='ciou',
            loss_weight=0.05,
            reduction='mean',
            return_iou=True,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5*(num_classes/80),
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_mask=dict(
            reduction='none', type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        loss_obj=dict(
            loss_weight=1.0*((crop_size/640)**2),
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        obj_level_weights=[4.0, 1.0, 0.4],
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[
                [(int(10*anchor_scale), int(13*anchor_scale)), (int(16*anchor_scale), int(30*anchor_scale)), (int(33*anchor_scale), int(23*anchor_scale))],
                [(int(30*anchor_scale), int(61*anchor_scale)), (int(62*anchor_scale), int(45*anchor_scale)), (int(59*anchor_scale), int(119*anchor_scale))],
                [(int(116*anchor_scale), int(90*anchor_scale)), (int(156*anchor_scale), int(198*anchor_scale)), (int(373*anchor_scale), int(326*anchor_scale))]
            ],
            strides=[8, 16, 32]
        ),
        prior_match_thr=4.0,
        ),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(type='nms', iou_threshold=0.6),
        nms_pre=30000,
        score_thr=0.001),
    )

# data config ----------------------------------------------------------------------------------------
num_workers = 4  # set fixed
bg_ratio = 0.2  # manual config
meta_info = dict(
    classes=("Penny", "Dime", "Nickel", "Quarter"), 
    palette=[(255, 0, 0), (100, 230, 4), (255, 255, 0), (0, 0, 255)]
)  # auto config


train_data_root = "/data/datasets/coins/coins-train"
color_type = "color"
val_data_root = "/data/datasets/coins/coins-val"
test_data_root = "/data/datasets/coins/coins-val"

random_flip_prob = 0.5  # manual_config

train_pipeline = [
        dict(
            type='LoadImageFromFile', 
            _scope_="mmyolo", 
            color_type='color', 
            imdecode_backend='cv2'
        ),
        dict(
            type='LoadAnnotations',
             _scope_="mmyolo",
            with_bbox=True,
        ),
        dict(
            type='Mosaic', 
            _scope_="mmyolo",
            img_scale=(crop_size, crop_size),
            pad_val=114.0,
            pre_transform=[
                dict(
                    type='LoadImageFromFile', 
                    _scope_="mmyolo", 
                    color_type='color', 
                    imdecode_backend='cv2'
                ),
                dict(
                    type='LoadAnnotations',
                    _scope_="mmyolo",
                    mask2bbox=True, 
                    with_bbox=True,
                    with_mask=True
                )
            ]
        ),
        dict(
            type='YOLOv5RandomAffine',
            _scope_="mmyolo",
            border=(-crop_size//2, -crop_size//2),
            border_val=(114, 114, 114),
            max_aspect_ratio=100,
            max_rotate_degree=0.0,
            max_shear_degree=0.0,
            min_area_ratio=0.01,
            scaling_ratio_range=(0.5, 1.5),            
            use_mask_refine=True, 
        ),
        dict(
            type='mmdet.Albu', 
            _scope_="mmyolo",
            bbox_params=dict(
                format='pascal_voc',
                label_fields=[
                    'gt_bboxes_labels',
                    'gt_ignore_flags',
                ],
                type='BboxParams'),
            keymap=dict(gt_bboxes='bboxes', img='image'),
            transforms=[
                dict(p=0.01, type='Blur'),
                dict(p=0.01, type='MedianBlur'),
                dict(p=0.01, type='ToGray'),
                dict(p=0.01, type='CLAHE'),
            ]
        ),
        dict(type='YOLOv5HSVRandomAug', _scope_="mmyolo"),
        dict(type='mmdet.RandomFlip', _scope_="mmyolo",prob=random_flip_prob),
        dict(type='Polygon2Mask', _scope_="mmyolo", downsample_ratio=4, mask_overlap=True),
        dict(
            type='PackDetInputs', _scope_="mmyolo",
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'flip',
                'instances',
                'flip_direction',
            )
        )
    ]

train_dataloader = dict(
    batch_size=batch_size,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type='VisaionYOLOv5InsDataset',
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
        _scope_="mmyolo", 
        color_type='color', 
        imdecode_backend='cv2'
    ),
    dict(
        type='YOLOv5KeepRatioResize',
        scale=(crop_size, crop_size), 
        _scope_="mmyolo"
    ),
    dict(
        type='LetterResize', 
        _scope_="mmyolo",
        allow_scale_up=False,
        half_pad_param=True,
        pad_val=dict(img=114),
        scale=(crop_size, crop_size),
    ),
    dict(
        type='LoadAnnotations',
        _scope_='mmdet', 
        with_bbox=True
    ),
    dict(
        type='PackDetInputs',
        _scope_="mmdet",
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'instances',
            'scale_factor',
            'pad_param',
        )
    )
]

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='VisaionYOLOv5InsDataset',
        _scope_="mmengine",
        metainfo=meta_info,
        data_root=val_data_root,
        data_prefix=dict(img=""),  # "" must be set for right work
        pipeline=val_pipeline,
        test_mode=True,
        batch_shapes_cfg=dict(
            batch_size=1,
            extra_pad_ratio=0.5,
            img_size=crop_size,
            size_divisor=32,
            type='BatchShapePolicy', _scope_="mmyolo")
    ),
    drop_last=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler',shuffle=False))


test_pipeline = [
    dict(
        type='LoadImageFromFile', 
        _scope_="mmyolo", 
        color_type='color', 
        imdecode_backend='cv2'
    ),
    dict(
        type='YOLOv5KeepRatioResize',
        scale=(crop_size, crop_size), 
        _scope_="mmyolo"
    ),
    dict(
        type='LetterResize', 
        _scope_="mmyolo",
        allow_scale_up=False,
        half_pad_param=True,
        pad_val=dict(img=114),
        scale=(crop_size, crop_size),
    ),
    dict(
        type='LoadAnnotations',
        _scope_='mmdet', 
        with_bbox=True
    ),
    dict(
        type='PackDetInputs',
        _scope_="mmdet",
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'instances',
            'scale_factor',
            'pad_param',
        )
    )
]


test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='VisaionYOLOv5InsDataset',
        _scope_="mmengine",
        metainfo=meta_info,
        data_root=test_data_root,
        data_prefix=dict(img=""),  # "" must be set for right work
        pipeline=test_pipeline,
        test_mode=True,
        batch_shapes_cfg=dict(
            batch_size=1,
            extra_pad_ratio=0.5,
            img_size=crop_size,
            size_divisor=32,
            type='BatchShapePolicy', _scope_="mmyolo")
    ),
    drop_last=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler',shuffle=False)
)

# training config ----------------------------------------------------------------------------------------
launcher = 'pytorch'
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)

max_iters = 3000  # manual config
val_interval = 500  # manual config
_experiment_name = "test"  # auto config
log_print_interval = 50  # manual config

lr = 0.01  # manual config
weight_decay = 0.0005  # manual config
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=batch_size,
        lr=lr,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=weight_decay)
)
param_scheduler = [
    dict(type='LinearLR', start_factor=lr*0.1, by_epoch=False, begin=0, end=int(max_iters*0.05)),
    dict(type='MultiStepLR', _scope_="mmseg",
       milestones=[int(max_iters / 2),
                   int(max_iters / 2 + max_iters / 4),
                   int(max_iters / 2 + max_iters / 4 + max_iters / 8)],
       by_epoch=False, 
       gamma=0.1)
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
    _scope_="mmdet",
    backend_args=None,
    format_only=False,
    metric=['bbox', 'segm'],
    proposal_nums=(100, 1, 10)
)
test_evaluator = [
    # dict(type='PrecisionAndRecallsMetricforVisaion'),
    dict(
        type='CocoMetric',
        _scope_="mmdet",
        backend_args=None,
        format_only=False,
        metric=['bbox', 'segm'],
        proposal_nums=(100, 1, 10)        
    )
]

visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    _scope_="mmdet",
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

train_pipeline_stage2 = [
    dict(
        type='LoadImageFromFile', 
        _scope_="mmyolo", 
        color_type='color', 
        imdecode_backend='cv2'
    ),
    dict(
            type='LoadAnnotations',
             _scope_="mmyolo",
            mask2bbox=True, 
            with_bbox=True,
            with_mask=True
    ),
    dict(
        type='YOLOv5KeepRatioResize',
        scale=(crop_size, crop_size), 
        _scope_="mmyolo"
    ),
    dict(
        type='LetterResize', 
        _scope_="mmyolo",
        allow_scale_up=False,
        half_pad_param=True,
        pad_val=dict(img=114),
        scale=(crop_size, crop_size)
    ),
    dict(
        type='PackDetInputs',
        _scope_="mmdet",
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'instances',
            'scale_factor',
            'pad_param',
        )
    )
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
    visualization=dict(type='DetVisualizationHook', _scope_="mmdet"))

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend=backend),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
randomness = dict(seed=1024, diff_rank_seed=False, deterministic=False)
work_dir = '/root/projects/visaionlib/workspace/tasks/coco_ins/rtmdet-ins_tiny_8xb32-300e_coco/test_result'