"""
This config is for the YOLOv5-det_s model.
comment explaination:
auto config -- the config should be set internally by the invoker
manual config -- the config should be set by the user
"""

# model config ----------------------------------------------------------------------------------------
in_channels = 3
num_classes = 4


# Dataloader 配置
num_workers = 4
batch_size = 8
bg_ratio = 0.1
meta_info = dict(
    classes=("Penny", "Dime", "Nickel", "Quarter"), 
    palette=[(255, 0, 0), (100, 230, 4), (255, 255, 0), (0, 0, 255)]
)  # auto config
train_data_root = "/data/datasets/coins/coins-train"
color_type = "color"
val_data_root = "/data/datasets/coins/coins-val"
test_data_root = "/data/datasets/coins/coins-val"
# Runtime
_work_dir = "/root/projects/visaionlib/workspace/tasks/coins_det/YOLOv5-det-s"
max_iters = 3000
val_interval = 500
_experiment_name = "yolov5_s"
_load_from = "/data/projects/visaionlib/pretrained_weights/yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230304_033134-8e0cd271.pth"

# Schedules
lr = 0.001
weight_decay = 0.0005
backend = "nccl"

crop_size = 640  # manual config
anchor_scale = crop_size / 640

deepen_factor = 0.33
widen_factor = 0.5

# model = dict(
#     type='YOLODetector',
#     _scope_="mmyolo",
#     data_preprocessor=dict(
#         type='YOLOv5DetDataPreprocessor',
#         _scope_="mmyolo",
#         batch_augments=None,
#         bgr_to_rgb=True,
#         mean=[0.0],
#         std=[255.0],
#         pad_size_divisor=32
#     ),
#     backbone=dict(
#         type='YOLOv5CSPDarknet',
#         _scope_="mmyolo",
#         act_cfg=dict(type='SiLU', inplace=True),
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         norm_cfg=dict(type='SyncBN'),
#         input_channels=in_channels,
#     ),
#     neck=dict(
#         type='YOLOv5PAFPN',
#         _scope_="mmyolo",
#         act_cfg=dict(type='SiLU', inplace=True),
#         norm_cfg=dict(type='SyncBN'),
#         num_csp_blocks=3,
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         in_channels=[256, 512, 1024],
#         out_channels=[256, 512, 1024]
#     ),
#     bbox_head=dict(
#         type='YOLOv5Head',
#         _scope_="mmyolo",
#         head_module=dict(
#             type='YOLOv5HeadModule',
#             featmap_strides=[8, 16, 32],
#             in_channels=[256, 512, 1024],
#             num_base_priors=3,
#             num_classes=num_classes,
#             widen_factor=widen_factor),
#         loss_bbox=dict(
#             bbox_format='xywh',
#             eps=1e-07,
#             iou_mode='ciou',
#             loss_weight=0.05,
#             reduction='mean',
#             return_iou=True,
#             type='IoULoss'),
#         loss_cls=dict(
#             loss_weight=0.5,
#             reduction='mean',
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True),
#         loss_obj=dict(
#             loss_weight=1.0*((crop_size/640)**2),
#             reduction='mean',
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True),
#         obj_level_weights=[4.0, 1.0, 0.4],
#         prior_generator=dict(
#             type='mmdet.YOLOAnchorGenerator',
#             base_sizes=[
#                 [(int(10*anchor_scale), int(13*anchor_scale)), (int(16*anchor_scale), int(30*anchor_scale)), (int(33*anchor_scale), int(23*anchor_scale))],
#                 [(int(30*anchor_scale), int(61*anchor_scale)), (int(62*anchor_scale), int(45*anchor_scale)), (int(59*anchor_scale), int(119*anchor_scale))],
#                 [(int(116*anchor_scale), int(90*anchor_scale)), (int(156*anchor_scale), int(198*anchor_scale)), (int(373*anchor_scale), int(326*anchor_scale))]
#             ],
#             strides=[8, 16, 32]
#         ),
#         prior_match_thr=4.0,
#         ),
#     test_cfg=dict(
#         max_per_img=300,
#         multi_label=True,
#         nms=dict(type='nms', iou_threshold=0.6),
#         nms_pre=30000,
#         score_thr=0.001),
# )

model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv5CSPDarknet',
        widen_factor=0.5),
    bbox_head=dict(
        head_module=dict(
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
            num_base_priors=3,
            num_classes=num_classes,
            type='YOLOv5HeadModule',
            widen_factor=0.5),
        loss_bbox=dict(
            bbox_format='xywh',
            eps=1e-07,
            iou_mode='ciou',
            loss_weight=0.05,
            reduction='mean',
            return_iou=True,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        obj_level_weights=[
            4.0,
            1.0,
            0.4,
        ],
        prior_generator=dict(
            base_sizes=[
                [
                    (
                        10,
                        13,
                    ),
                    (
                        16,
                        30,
                    ),
                    (
                        33,
                        23,
                    ),
                ],
                [
                    (
                        30,
                        61,
                    ),
                    (
                        62,
                        45,
                    ),
                    (
                        59,
                        119,
                    ),
                ],
                [
                    (
                        116,
                        90,
                    ),
                    (
                        156,
                        198,
                    ),
                    (
                        373,
                        326,
                    ),
                ],
            ],
            strides=[
                8,
                16,
                32,
            ],
            type='mmdet.YOLOAnchorGenerator'),
        prior_match_thr=4.0,
        type='YOLOv5Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOv5DetDataPreprocessor'),
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
        type='YOLOv5PAFPN',
        widen_factor=0.5),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    type='YOLODetector')

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
    dict(type='CocoMetric', _scope_='mmdet', metric=['bbox'], format_only=False, classwise=True,
         metric_items=['mAP', 'mAP_50'])
]
test_evaluator = [
    dict(type='PrecisionAndRecallsMetricforVisaion'),
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

log_print_interval = 50
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