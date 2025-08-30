"""
This config is for the RTMDet-ins_middle model.
comment explaination:
auto config -- the config should be set internally by the invoker
manual config -- the config should be set by the user
"""
# Model config
num_classes = {{"$VISAION_NUM_CLASSES:1"}}  # auto config
in_channels = 3 #{{'$VISAION_IN_CHANNELS:3'}}  # fixed
val_iou_threshold = 0.6 # {{"$VISAION_VAL_IOU_THRESHOLD:0.6"}}  # set fixed
val_iou_threshold = float(val_iou_threshold)
val_score_threshold = {{"$VISAION_VAL_SCORE_THRESHOLD:0.05"}}  # manual config
val_score_threshold = float(val_score_threshold)
val_max_per_img = {{"$VISAION_VAL_MAX_PER_IMG:100"}}  # manual config
val_max_per_img = int(val_max_per_img)
val_nms_pre = {{"$VISAION_VAL_NMS_PRE:1000"}}  # manual config
val_nms_pre = int(val_nms_pre)
backbone_widen_factor = 0.75  # set fixed
backbone_deepen_factor = 0.67  # set fixed
neck_in_channels = [192, 384, 768]  # set fixed
neck_out_channels = 192   # set fixed
bbox_head_feat_channels = 192  # set fixed
bbox_head_in_channels = 192  # set fixed
neck_num_csp_blocks = 2  # set fixed


# Dataloader config
# in object detection and instance segmentation, the number of classes is exclusive of background, due to classification loss (eg. focal loss)
batch_size = {{"$VISAION_BATCH_SIZE:8"}}  # manual config | this is for a single GPU
batch_size = int(batch_size)
num_workers = 4  # {{"$VISAION_NUM_WORKERS:4"}}  # set fixed    
bg_ratio = {{"$VISAION_NEG_RATIO:0.2"}}  # manual config
bg_ratio = float(bg_ratio)
meta_info = {{"$VISAION_META_INFO:None"}}  # auto config
data_root = '{{"$VISAION_TRAIN_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_ins"}}' # auto config
color_type = 'color'  # "{{'$VISAION_COLOR_TYPE:color'}}" # fixed
val_data_root = '{{"$VISAION_VAL_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_ins"}}' # auto config
test_data_root = '{{"$VISAION_TEST_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_ins"}}' # auto config
base_scale_factor = {{"$VISAION_BASE_SCALE_FACTOR:1.0"}}  # manual config
base_scale_factor = float(base_scale_factor)
scale_range = {{"$VISAION_SCALE_RANGE:0.1"}}  # manual config
scale_range = float(scale_range)
random_rescale_prob = {{"$VISAION_RANDOM_RESCALE_PROB:0.5"}}  # manual_config
random_rescale_prob = float(random_rescale_prob)
random_flip_prob = {{"$VISAION_RANDOM_FLIP_PROB:0.5"}}  # manual_config
random_flip_prob = float(random_flip_prob)
crop_size = {{"$VISAION_CROP_SIZE:1024"}}  # manual config
crop_size = int(crop_size)

# Runtime config
_work_dir = '{{"$VISAION_WORK_DIR:/root/data/visaion/visaionlib/work_dirs/rtmdet-ins-s"}}'  # auto config
max_iters = {{"$VISAION_MAX_ITERS:2000"}}  # manual config
max_iters = int(max_iters)
val_interval = {{"$VISAION_VAL_INTERVAL:100"}}  # manual config
val_interval = int(val_interval)
_experiment_name = "{{'$VISAION_EXPERIMENT_NAME:test'}}"  # auto config
log_print_interval = {{"$VISAION_LOG_PRINT_INTERVAL:50"}}  # manual config
_load_from = '{{"$VISAION_LOAD_FROM:/root/data/visaion/visaionlib/pretrained_weights/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"}}'    # auto config

# Schedules
lr = {{"$VISAION_LR:0.004"}}  # manual config
lr = float(lr)
weight_decay = {{"$VISAION_WEIGHT_DECAY:0.05"}}  # manual config
weight_decay = float(weight_decay)
backend = '{{"$VISAION_BACKEND:nccl"}}' # auto config



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
        widen_factor=backbone_widen_factor,
        deepen_factor=backbone_deepen_factor,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        _scope_="mmdet",
        act_cfg=dict(type='SiLU', inplace=True),
        expand_ratio=0.5,
        in_channels=neck_in_channels,
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=neck_num_csp_blocks,
        out_channels=neck_out_channels,
    ),
    bbox_head=dict(
        type='VisaionRTMDetInsSepBNHead',  # use VisaionRTMDetInsSepBNHead instead of RTMDetInsSepBNHead due to the bug of RTMDetInsSepBNHead
        act_cfg=dict(type='SiLU', inplace=True),
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0, 
            strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        feat_channels=bbox_head_feat_channels,
        in_channels=bbox_head_in_channels,
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
        max_per_img=val_max_per_img,
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=val_iou_threshold),
        nms_pre=val_nms_pre,
        score_thr=val_score_threshold),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        debug=False,
        pos_weight=-1),
    )


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


val_evaluator = dict(
    type='CocoMetricVisaion',
    backend_args=None,
    format_only=False,
    metric=['bbox', 'segm'],
    proposal_nums=(100, 1, 10)
)
test_evaluator = [
    dict(type='PrecisionAndRecallsMetricforVisaion', _scope_="mmengine"),
    dict(
        type='CocoMetricVisaion',
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


env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend=backend),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))


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

default_scope = 'mmdet'
work_dir = _work_dir
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
log_level = 'INFO'
randomness = dict(seed=1024, diff_rank_seed=False, deterministic=False)
experiment_name = _experiment_name
load_from = _load_from
launcher = 'pytorch'
