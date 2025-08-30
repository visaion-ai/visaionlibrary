"""
This config is for the YOLOv5-ins_s model.
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
crop_size = {{"$VISAION_CROP_SIZE:1024"}}  # manual config
crop_size = int(crop_size)
anchor_scale = float(crop_size) / 640 # auto config
widen_factor = 0.5 # set fixed
deepen_factor = 0.33 # set fixed
loss_cls_loss_weight = 0.5 # set fixed
loss_obj_loss_weight = 1.0 # set fixed


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
random_flip_prob = {{"$VISAION_RANDOM_FLIP_PROB:0.5"}}  # manual_config
random_flip_prob = float(random_flip_prob)


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

# Model配置
in_channels = 3 # set fixed
num_classes = {{'$VISAION_NUM_CLASSES:1'}} # auto config

# Dataloader 配置
num_workers = 4 # set fixed
batch_size = {{"$VISAION_BATCH_SIZE:16"}}
image_scale = {{"$VISAION_IMAGE_SCALE:640"}}
meta_info = {{"$VISAION_META_INFO:None"}}
train_data_root = '{{"$VISAION_TRAIN_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_ins"}}'
color_type = "{{'$VISAION_COLOR_TYPE:color'}}"
val_data_root = '{{"$VISAION_VAL_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_ins"}}'
test_data_root = '{{"$VISAION_TEST_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_ins"}}'
random_flip_prob = {{"$VISAION_RANDOM_FLIP_PROB:0.5"}}

# Runtime
_work_dir = '{{"$VISAION_WORK_DIR:/root/data/visaion/visaionlib/work_dirs/yolov5s_ins_seg_all_in_one_visaion"}}'
max_iters = {{"$VISAION_MAX_ITERS:3000"}}   # manual config
max_iters = int(max_iters) 
val_interval = {{"$VISAION_VAL_INTERVAL:100"}}  # manual config
val_interval = int(val_interval) 
_experiment_name = '{{"$VISAION_EXPERIMENT_NAME:yolov5s_ins_seg_all_in_one_visaion"}}'
_load_from = '{{"$VISAION_LOAD_FROM:https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance_20230426_012542-3e570436.pth"}}'

# Schedules
lr = {{"$VISAION_LR:0.01"}} # manual config
lr = float(lr)
weight_decay = {{"$VISAION_WEIGHT_DECAY:0.0005"}} # manual config
weight_decay = float(weight_decay)
backend = '{{"$VISAION_BACKEND:nccl"}}'


model = dict(
    type='YOLODetector',
    _scope_="mmyolo",
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        _scope_="mmyolo",
        batch_augments=None,
        bgr_to_rgb=False,
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
        type='YOLOv5InsHead',
        _scope_="mmyolo",
        head_module=dict(
            type='YOLOv5InsHeadModule',
            featmap_strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            mask_channels=32,
            num_base_priors=3,
            num_classes=num_classes,
            proto_channels=256,            
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
            loss_weight=loss_cls_loss_weight,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_mask=dict(
            reduction='none', type='mmdet.CrossEntropyLoss', use_sigmoid=True),
        loss_mask_weight=0.05,
        loss_obj=dict(
            loss_weight=loss_obj_loss_weight,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        mask_overlap=True,
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
        fast_test=True,
        mask_thr_binary=0.5,
        max_per_img=val_max_per_img,
        min_bbox_size=0,
        multi_label=True,
        nms=dict(type='nms', iou_threshold=val_iou_threshold),
        nms_pre=val_nms_pre,
        score_thr=val_score_threshold),
    )


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
            mask2bbox=True, 
            with_bbox=True,
            with_mask=True
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


val_evaluator = dict(
    type='CocoMetricVisaion',
    _scope_="mmdet",
    backend_args=None,
    format_only=False,
    metric=['bbox', 'segm'],
    proposal_nums=(100, 1, 10)
)
test_evaluator = [
    dict(type='PrecisionAndRecallsMetricforVisaion'),
    dict(
        type='CocoMetricVisaion',
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
    visualization=dict(type='DetVisualizationHook', _scope_="mmdet"))
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



default_scope = 'mmyolo'
work_dir = _work_dir
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
log_level = 'INFO'
load_from = _load_from
resume = False
randomness = dict(seed=42, diff_rank_seed=False, deterministic=False)
experiment_name = _experiment_name
