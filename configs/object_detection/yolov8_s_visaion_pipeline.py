import os

# Model配置
in_channels = {{'$VISAION_IN_CHANNELS:3'}}
num_classes = {{'$VISAION_NUM_CLASSES:2'}}

# Dataloader 配置
num_workers = 4
neg_ratio = {{"$VISAION_NEG_RATIO:0.3"}}
img_scale = {{"$VISAION_IMG_SCALE:640"}}
batch_size = {{"$VISAION_BATCH_SIZE:12"}}
train_data_root = {{"$VISAION_TRAIN_DATA_ROOT:'/root/data/visaion/visaionlib/data/demo_det'"}}
meta_info = {{"$VISAION_META_INFO:None"}}
color_type = {{"$VISAION_COLOR_TYPE:'color'"}}
val_data_root = {{"$VISAION_VAL_DATA_ROOT:'/root/data/visaion/visaionlib/data/demo_det'"}}
test_data_root = {{"$VISAION_TEST_DATA_ROOT:'/root/data/visaion/visaionlib/data/demo_det'"}}

# Runtime
_work_dir = {{"$VISAION_WORK_DIR:'work_dirs/yolov8_s_visaion_pipeline'"}}
max_iters = {{"$VISAION_MAX_ITERS:4000"}}
max_epochs = {{"$VISAION_MAX_EPOCHS:40"}}
interval = {{"$VISAION_INTERVAL:1000"}}
_experiment_name = {{"$VISAION_EXPERIMENT_NAME:'yolov8_s'"}}
_load_from = {{"$VISAION_LOAD_FROM:'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'"}}

# Schedules
lr = {{"$VISAION_LR:0.01"}}
weight_decay = {{"$VISAION_WEIGHT_DECAY:0.0005"}}
backend = {{"$VISAION_BACKEND:'nccl'"}}


model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            0.0,
        ],
        std=[
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        input_channels=in_channels,
        deepen_factor=0.33,
        frozen_stages=4,
        last_stage_out_channels=1024,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=0.5),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv8PAFPN',
        widen_factor=0.5),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=num_classes,
            reg_max=16,
            type='YOLOv8HeadModule',
            widen_factor=0.5),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv8Head'),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=num_classes,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    type='YOLODetector')


train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile', color_type=color_type, imdecode_backend='cv2'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            img_scale,
            img_scale,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        type='Mosaic'),
    dict(
        border=(
            -320,
            -320,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
    dict(
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
        ],
        type='mmdet.Albu'),
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'instances',
            'flip_direction',
        ),
        type='mmdet.PackDetInputs'),
]
train_dataloader = dict(
    batch_size=batch_size,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        data_root=train_data_root,
        data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=('overflow', 'collapse'), palette=[
            (
                20,
                220,
                60,
            ),
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=train_pipeline,
        type='VisaionDetDataset'),
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='InfiniteSamplerBatch', neg_ratio=neg_ratio, batch_size=batch_size),
    batch_sampler=dict(type='BatchSamplerConstantOne', drop_last=False),
)
val_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile', color_type=color_type, imdecode_backend='cv2'),
    dict(scale=(
        img_scale,
        img_scale,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            img_scale,
            img_scale,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'instances',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        batch_shapes_cfg=None,
        data_root=val_data_root,
        data_prefix=dict(img=""),
        metainfo=dict(classes=('overflow', 'collapse'), palette=[
            (
                20,
                220,
                60,
            ),
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=val_pipeline,
        test_mode=True,
        type='VisaionDetDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile', color_type=color_type, imdecode_backend='cv2'),
    dict(scale=(
        img_scale,
        img_scale,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            img_scale,
            img_scale,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'instances',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        batch_shapes_cfg=None,
        data_root=test_data_root,
        data_prefix=dict(img=""),
        metainfo=dict(classes=('overflow', 'collapse'), palette=[
            (
                20,
                220,
                60,
            ),
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=test_pipeline,
        test_mode=True,
        type='VisaionDetDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))


val_evaluator = [
    dict(type='CocoMetric', _scope_='mmdet', metric=['bbox'], format_only=False, classwise=True,
         metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@100'])
]
test_evaluator = [
    dict(type='PrecisionAndRecallsMetricforVisaion', output_dir=os.path.join(_work_dir, "metrics")),
    dict(type='CocoMetric', _scope_='mmdet', metric=['bbox'], format_only=False, classwise=True,
         metric_items=['mAP', 'mAP_50'])
]


visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])


optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay)
)


param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', _scope_="mmseg",
       milestones=[int(max_iters / 2),
                   int(max_iters / 2 + max_iters / 4),
                   int(max_iters / 2 + max_iters / 4 + max_iters / 8)],
       by_epoch=False, gamma=0.1)]


train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_begin=1, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))


default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmdet'),
    logger=dict(
        type='LoggerHook',
        _scope_='mmdet',
        interval=50,
        log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmdet'),
    checkpoint=dict(
        type='CheckpointHook',
        _scope_='mmdet',
        by_epoch=False,
        interval=interval,
        save_last=True,
        filename_tmpl='checkpoint_{:06d}.pth',
        save_best=['coco/bbox_mAP', 'coco/bbox_mAP_50'],
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        draw=True,
        test_out_dir='show_result',
        type='mmdet.DetVisualizationHook'))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]


default_scope = 'mmyolo'
work_dir = _work_dir
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
log_level = 'INFO'
load_from = _load_from
resume = False
randomness = dict(seed=3407, diff_rank_seed=False, deterministic=False)
experiment_name = _experiment_name