# Model配置
in_channels = {{'$VISAION_IN_CHANNELS:3'}}
num_classes = {{'$VISAION_NUM_CLASSES:2'}}

# Dataloader 配置
num_workers = 4
batch_size = {{"$VISAION_BATCH_SIZE:16"}}
image_scale = {{"$VISAION_IMAGE_SCALE:640"}}
# meta_info = {{"$VISAION_META_INFO:None"}}
meta_info = dict(classes=['plastic', 'glass'], palette=[(0,255,0), (255,0,0)])
train_data_root = '{{"$VISAION_TRAIN_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_cls"}}'
color_type = "{{'$VISAION_COLOR_TYPE:color'}}"
val_data_root = '{{"$VISAION_VAL_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_cls"}}'
test_data_root = '{{"$VISAION_TEST_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_cls"}}'

# Runtime
_work_dir = '{{"$VISAION_WORK_DIR:/root/data/visaion/visaionlib/work_dirs/resnet_34_cls"}}'
interval = {{"$VISAION_INTERVAL:100"}}
max_epochs = {{"$VISAION_MAX_EPOCHS:300"}}
_experiment_name = '{{"$VISAION_EXPERIMENT_NAME:resnet_34_cls"}}'
_load_from = '{{"$VISAION_LOAD_FROM:/root/data/visaion/visaionserver/.visaion/weights/resnet34.pth"}}'
# _load_from = '{{"$VISAION_LOAD_FROM:https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth"}}'

# Schedules
lr = {{"$VISAION_LR:0.01"}}
weight_decay = {{"$VISAION_WEIGHT_DECAY:0.0005"}}
backend = '{{"$VISAION_BACKEND:nccl"}}'


data_preprocessor = dict(
    _scope_="mmpretrain",
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[0,],
    std=[255.0],
    # convert image from BGR to RGB
    to_rgb=False,
)
model = dict(
    type='ImageClassifier',
    _scope_="mmpretrain",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=34,
        in_channels=in_channels,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
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
                color_type='color',
                imdecode_backend='cv2',
                ),
            dict(type='RandomResizedCrop', scale=image_scale, _scope_="mmpretrain"),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs',
                 _scope_="mmpretrain",
                 meta_keys=[
                    'img_path', 'sample_global_id',
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
                color_type='color',
                imdecode_backend='cv2',
                ),
            dict(type='RandomResizedCrop', scale=image_scale, _scope_="mmpretrain"),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs',
                 _scope_="mmpretrain",
                 meta_keys=[
                    'img_path', 'sample_global_id',
                    'dataset_metainfo'
                ])
        ]))
test_dataloader = val_dataloader


val_evaluator = dict(type='Accuracy', topk=(1, 2), _scope_="mmpretrain")
test_evaluator = val_evaluator


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends, _scope_="mmpretrain")


optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))


param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)


train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=interval)
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
    logger=dict(type='LoggerHook', interval=50),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=interval),
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
log_level = 'INFO'
load_from = _load_from
resume = False
randomness = dict(seed=42, diff_rank_seed=False, deterministic=False)
experiment_name = _experiment_name

