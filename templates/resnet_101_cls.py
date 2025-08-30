# Model配置
in_channels = 3 # {{'$VISAION_IN_CHANNELS:3'}} set fixed
num_classes = {{'$VISAION_NUM_CLASSES:2'}}
backbone_depth = 101 # set fixed
head_in_channels = 2048 # set fixed

# Dataloader 配置
num_workers = 4 # set fixed
batch_size = {{"$VISAION_BATCH_SIZE:16"}}  # manual config
image_scale = {{"$VISAION_CROP_SIZE:640"}}  # manual config
meta_info = {{"$VISAION_META_INFO:None"}}  # auto config
train_data_root = '{{"$VISAION_TRAIN_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_cls"}}'  # auto config
color_type = 'color' # "{{'$VISAION_COLOR_TYPE:color'}}"  # set fixed
val_data_root = '{{"$VISAION_VAL_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_cls"}}'  # auto config
test_data_root = '{{"$VISAION_TEST_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_cls"}}'  # auto config
random_flip_prob = {{"$VISAION_RANDOM_FLIP_PROB:0.5"}}  # manual config

# Runtime
_work_dir = '{{"$VISAION_WORK_DIR:/root/data/visaion/visaionlib/work_dirs/resnet_34_cls"}}'  # auto config
interval = {{"$VISAION_VAL_INTERVAL:100"}}  # manual config
max_iter = {{"$VISAION_MAX_ITER:3000"}}  # manual config
_experiment_name = '{{"$VISAION_EXPERIMENT_NAME:resnet_34_cls"}}' # auto config
_load_from = '{{"$VISAION_LOAD_FROM:https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth"}}' # auto config

# Schedules
lr = {{"$VISAION_LR:0.01"}}  # manual config
weight_decay = {{"$VISAION_WEIGHT_DECAY:0.0005"}}  # manual config
backend = '{{"$VISAION_BACKEND:nccl"}}'  # auto config


model = dict(
    type='ImageClassifier',
    _scope_="mmpretrain",
    data_preprocessor = dict(
        _scope_="mmpretrain",
        num_classes=num_classes,
        # RGB format normalization parameters
        mean=[0,0,0],
        std=[255.0,255.0,255.0],
        # convert image from BGR to RGB
        to_rgb=False,
    ),
    backbone=dict(
        type='ResNet',
        depth=backbone_depth,
        in_channels=in_channels,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=head_in_channels,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type='VisaionClsDataset',
        metainfo=meta_info,
        data_root=train_data_root,
        data_prefix=dict(),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type=color_type,
                imdecode_backend='cv2',
                ),
            dict(type='Resize', scale=image_scale, _scope_="mmcv"),
            dict(type='RandomFlip', prob=random_flip_prob, direction='horizontal'),
            dict(type='PackInputs',
                 _scope_="mmpretrain",
                 meta_keys=[
                    'img_path',
                    'dataset_metainfo'
                ])
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='VisaionClsDataset',
        metainfo=meta_info,
        data_root=val_data_root,
        data_prefix=dict(),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                color_type=color_type,
                imdecode_backend='cv2',
                ),
            dict(type='Resize', scale=image_scale, _scope_="mmcv"),
            dict(type='PackInputs',
                 _scope_="mmpretrain",
                 meta_keys=[
                    'img_path',
                    'dataset_metainfo'
                ])
        ]))
test_dataloader = val_dataloader


val_evaluator = [
    dict(type='Accuracy', topk=(1, 2), _scope_="mmpretrain")
]
test_evaluator = [
    dict(type='AccuracyVisaionHttp', topk=(1, 2), _scope_="mmengine")
]



vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends, _scope_="mmpretrain")


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr,  weight_decay=weight_decay),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
)


param_scheduler = dict(
    type='MultiStepLR', by_epoch=False, milestones=[max_iter * 0.5, max_iter * 0.75, max_iter * 0.9], gamma=0.1)


train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iter, val_interval=interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)


# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=interval, 
        max_keep_ckpts=1,
        save_last=False,
        filename_tmpl='checkpoint_{:06d}.pth',
        ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False, _scope_="mmpretrain"),
)


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=batch_size)


default_scope = 'mmpretrain'
work_dir = _work_dir
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
log_level = 'INFO'
load_from = _load_from
resume = False
randomness = dict(seed=42, diff_rank_seed=False, deterministic=False)
experiment_name = _experiment_name

