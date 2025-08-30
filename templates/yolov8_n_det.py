"""
This config is for the YOLOv8-det_n model.
comment explaination:
auto config -- the config should be set internally by the invoker
manual config -- the config should be set by the user
"""
# Model config
num_classes = {{"$VISAION_NUM_CLASSES:1"}}  # auto config
in_channels = 3 #{{'$VISAION_IN_CHANNELS:3'}}  # fixed
deepen_factor = 0.33  # set fixed
widen_factor = 0.25  # set fixed
backbone_last_stage_out_channels = 1024  # set fixed
neck_in_channels = [256, 512, 1024]  # set fixed
neck_out_channels = [256, 512, 1024] # set fixed
bbox_head_in_channels = [256, 512, 1024]  # set fixed
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

# Dataloader config
# in object detection and instance segmentation, the number of classes is exclusive of background, due to classification loss (eg. focal loss)
batch_size = {{"$VISAION_BATCH_SIZE:8"}}  # manual config | this is for a single GPU
batch_size = int(batch_size)
num_workers = 4  # {{"$VISAION_NUM_WORKERS:4"}}  # set fixed    
bg_ratio = {{"$VISAION_NEG_RATIO:0.2"}}  # manual config
bg_ratio = float(bg_ratio)
meta_info = {{"$VISAION_META_INFO:None"}}  # auto config
train_data_root = '{{"$VISAION_TRAIN_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_ins"}}' # auto config
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



model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        batch_augments=None,
        bgr_to_rgb=True,
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        pad_size_divisor=32
    ),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        input_channels=in_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        last_stage_out_channels=backbone_last_stage_out_channels,
        norm_cfg=dict(type='SyncBN')
    ),
    neck=dict(
        type='YOLOv8PAFPN',
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=deepen_factor,
        in_channels=neck_in_channels,
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=3,
        out_channels=neck_out_channels,
        widen_factor=widen_factor
    ),
    bbox_head=dict(
        type='YOLOv8Head',
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            type='YOLOv8HeadModule',
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[8, 16, 32],
            in_channels=bbox_head_in_channels,
            norm_cfg=dict(type='SyncBN'),
            num_classes=num_classes,
            reg_max=16,
            widen_factor=widen_factor
        ),
        loss_bbox=dict(
            type='IoULoss',
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            loss_weight=0.5,
            reduction='none',
            use_sigmoid=True
        ),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            loss_weight=0.375,
            reduction='mean'
        ),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator',
            offset=0.5, 
            strides=[8, 16, 32] 
        )
    ),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=num_classes,
            topk=10,
            use_ciou=True)
        ),
    test_cfg=dict(
        max_per_img=val_max_per_img,
        multi_label=True,
        nms=dict(iou_threshold=val_iou_threshold, type='nms'),
        nms_pre=val_nms_pre,
        score_thr=val_score_threshold))



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
                with_bbox=True,
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
    dict(
        type='PackDetInputs', _scope_="mmdet",
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
        with_bbox=True,
    ),
    dict(
        type='YOLOv5KeepRatioResize',
        scale=(crop_size, crop_size), 
        _scope_="mmyolo"
    ),
    dict(
        type='LetterResize', 
        _scope_="mmyolo",
        allow_scale_up=True,
        pad_val=dict(img=114),
        scale=(crop_size, crop_size),
    ),
    dict(
        type='YOLOv5RandomAffine',
        _scope_="mmyolo",
        border_val=(114, 114, 114),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),            
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
    dict(
        type='PackDetInputs', _scope_="mmdet",
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
        type='VisaionDetDataset',
        data_root=train_data_root,
        data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=False, min_size=16),
        metainfo=meta_info,
        pipeline=train_pipeline
    ),
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(
        type="VisaionInfiniteSampler",
        shuffle=True,
        bg_ratio=bg_ratio,
        batch_size=batch_size,
    )
)
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, color_type=color_type, imdecode_backend='cv2'),
    dict(
        type='YOLOv5KeepRatioResize',
        scale=(crop_size, crop_size), 
        _scope_="mmyolo"
    ),
    dict(
        type='LetterResize', 
        _scope_="mmyolo",
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(crop_size, crop_size),
    ),
    dict(type='LoadAnnotations', _scope_='mmdet', with_bbox=True),
    dict(
        _scope_="mmdet",
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'instances',
            'scale_factor',
            'pad_param'
        ),
        type='PackDetInputs'),
]
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='VisaionDetDataset',
        batch_shapes_cfg=None,
        data_root=val_data_root,
        data_prefix=dict(img=""),
        metainfo=meta_info,
        pipeline=val_pipeline,
        test_mode=True
    ),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False)
)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, color_type=color_type, imdecode_backend='cv2'),
    dict(
        type='YOLOv5KeepRatioResize',
        scale=(crop_size, crop_size), 
        _scope_="mmyolo"
    ),
    dict(
        type='LetterResize', 
        _scope_="mmyolo",
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(crop_size, crop_size),
    ),
    dict(type='LoadAnnotations', _scope_='mmdet', with_bbox=True),
    dict(
        _scope_="mmdet",
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'instances',
            'scale_factor',
            'pad_param'
        ),
        type='PackDetInputs'),
]
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='VisaionDetDataset',
        batch_shapes_cfg=None,
        data_root=test_data_root,
        data_prefix=dict(img=""),
        metainfo=meta_info,
        pipeline=test_pipeline,
        test_mode=True
    ),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False))


val_evaluator = [
    dict(type='CocoMetricVisaion', _scope_='mmdet', metric=['bbox'], format_only=False, classwise=True,
         metric_items=['mAP', 'mAP_50'])
]
test_evaluator = [
    dict(type='PrecisionAndRecallsMetricforVisaion'),
    dict(type='CocoMetricVisaion', _scope_='mmdet', metric=['bbox'], format_only=False, classwise=True,
         metric_items=['mAP', 'mAP_50'])
]


visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])


optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
)


param_scheduler = [
    dict(type='LinearLR', start_factor=lr*0.1, by_epoch=False, begin=0, end=int(max_iters*0.05)),
    dict(type='MultiStepLR', _scope_="mmseg",
       milestones=[int(max_iters / 2),
                   int(max_iters / 2 + max_iters / 4),
                   int(max_iters / 2 + max_iters / 4 + max_iters / 8)],
       by_epoch=False, gamma=0.1)]


train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_begin=1, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))


default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmdet'),
    logger=dict(
        type='LoggerHook',
        _scope_='mmdet',
        interval=log_print_interval,
        log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmdet'),
    checkpoint=dict(
        type='CheckpointHook',
        _scope_='mmdet',
        by_epoch=False,
        interval=val_interval,
        save_last=True,
        filename_tmpl='checkpoint_{:06d}.pth',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
)
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49,
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
randomness = dict(seed=3407, diff_rank_seed=False, deterministic=False)
experiment_name = _experiment_name