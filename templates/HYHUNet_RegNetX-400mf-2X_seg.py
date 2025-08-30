"""
This config is for the HYHUNet-RegNetX-400mf-2X model.
comment explaination:
auto config -- the config should be set internally by the invoker
manual config -- the config should be set by the user
"""

# Model config
in_channels = 3 #{{'$VISAION_IN_CHANNELS:3'}}  # auto config
num_classes = {{'$VISAION_NUM_CLASSES:2'}}  # auto config
num_classes = int(num_classes)
slide_infer_mode = {{'$VISAION_SLIDE_INFER_MODE:True'}}  # manual config
crop_size = {{'$VISAION_INFER_CROP_SIZE:1024'}}  # manual config
crop_size = int(crop_size)
infer_stride = {{'$VISAION_STRIDE:512'}}  # manual config
infer_stride = int(infer_stride)
checkpoint = "{{'$VISAION_CHECKPOINT:/root/data/visaion/visaionserver/.visaion/weights/regnetx-400mf_8xb128_in1k_20211213-89bfc226.pth'}}" # auto config

# Dataloader config
batch_size = {{"$VISAION_BATCH_SIZE:8"}}  # manual config | this is for a single GPU
batch_size = int(batch_size)
num_workers = 4  # set fixed
bg_ratio = {{"$VISAION_NEG_RATIO:0.2"}}  # manual config
bg_ratio = float(bg_ratio)
meta_info = {{"$VISAION_META_INFO:None"}}  # auto config
train_data_root = '{{"$VISAION_TRAIN_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_seg"}}'
color_type = 'color' # {{'$VISAION_COLOR_TYPE:color'}}" fixed
val_data_root = '{{"$VISAION_VAL_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_seg"}}'
test_data_root = '{{"$VISAION_TEST_DATA_ROOT:/root/data/visaion/visaionlib/data/demo_seg"}}'
base_scale_factor = {{"$VISAION_BASE_SCALE_FACTOR:1.0"}}  # manual config
base_scale_factor = float(base_scale_factor)    # 有时候前端传回来的是int, 需要转换为float
scale_range = {{"$VISAION_SCALE_RANGE:0.1"}}  # manual config
scale_range = float(scale_range)    # 有时候前端传回来的是int, 需要转换为float
random_rescale_prob = {{"$VISAION_RANDOM_RESCALE_PROB:0.5"}}  # manual_config
random_rescale_prob = float(random_rescale_prob)    # 有时候前端传回来的是int, 需要转换为float
random_flip_prob = {{"$VISAION_RANDOM_FLIP_PROB:0.5"}}  # manual_config
random_flip_prob = float(random_flip_prob)    # 有时候前端传回来的是int, 需要转换为float

# evaluation
iou_threshold = {{"$VISAION_VAL_IOU_THRESHOLD:0.25"}}
iou_threshold = float(iou_threshold)
min_area = {{"$VISAION_VAL_MIN_AREA:60"}}
min_area = int(min_area)

# Runtime
_work_dir = '{{"$VISAION_WORK_DIR:/root/data/visaion/visaionlib/work_dirs/HYHUNet-RegNetX-400mf-2X"}}'
max_iters = {{"$VISAION_MAX_ITERS:2000"}}  # manual config
max_iters = int(max_iters)
val_interval = {{"$VISAION_VAL_INTERVAL:20"}}  # manual config
val_interval = int(val_interval)
_experiment_name = '{{"$VISAION_EXPERIMENT_NAME:test"}}'  # auto config
log_print_interval = {{"$VISAION_LOG_PRINT_INTERVAL:50"}}  # manual config
log_print_interval = int(log_print_interval)

# Schedules
lr = {{"$VISAION_LR:0.001"}}  # manual config
weight_decay = {{"$VISAION_WEIGHT_DECAY:0.0005"}}  # manual config
backend = "nccl"  # auto config



model = dict(
    type="EncoderDecoder",
    _scope_="mmengine",
    infer_batch_size=batch_size,
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        _scope_="mmengine",
        # make sure the norm parameters and rgb order are aligned with the pre-trained model
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        # if the pre-trained model requires color input, then set this to True
        # here regnetx-800mf requires color input,so set this to True permanently
        grayscale_to_color=True,
        # in all cases, the pad_val is 0
        pad_val=0,
        # in all cases, the seg_pad_val is 0, which means background
        seg_pad_val=0,
        size_divisor=32,
        test_cfg=(
            dict(size=(crop_size, crop_size))
            if slide_infer_mode
            else dict(size_divisor=32)
        ),  # 测试时，如果使用的是滑动窗口，那么需要设置测试的crop_size，否则设置为size_divisor的整数倍
    ),
    backbone=dict(
        type="RegNetVisaion",
        arch="regnetx_400mf",
        out_stem=True,
        in_channels=in_channels,
        stem_channels=32,
        base_channels=32,
        strides=(2, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=True,
        init_cfg=dict(type="Pretrained", prefix="backbone", checkpoint=checkpoint),
    ),
    decode_head=dict(
        type="HYHUNetHead",
        prediction_channels=64,  # 开始设置的16太小了，需要设置为64
        num_classes=num_classes,
        fusion_mode="concate",
        fusion_density_mode="common",
        block_drop_path_rate=0.0,
        upsample_mode="interpolate",
        upsample_interpolate_mode="bilinear",
        sigmoid_before_loss=False,
        num_block_list=[1, 1, 1, 1, 1],
        block_channel_list=[256, 128, 64, 32, 16],
        input_channel_list=[32, 32, 64, 160, 384],
        last_scale_factor=2,
        conv_cfg=None,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        act_cfg=dict(type="ReLU", inplace=True),
        sampler=dict(
            type="OHEMPixelSampler", _scope_="mmseg", thresh=0.7, min_kept=10000
        ),
        loss_decode=dict(
            type="CrossEntropyLoss",
            _scope_="mmseg",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.7] + [1] * (num_classes - 1),
        ),
    ),
    auxiliary_head=dict(
        type="STDCHead",
        _scope_="mmseg",
        in_channels=64,  # CAUTION: this value is from the backbone output channels, index is in_index of the outputs of the backbone
        channels=64,
        num_convs=1,
        num_classes=num_classes,
        boundary_threshold=0.1,
        in_index=2,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        concat_input=False,
        align_corners=True,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                loss_name="loss_ce",
                use_sigmoid=True,
                loss_weight=1.0,
            ),
            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=1.0),
        ],
    ),
    train_cfg=dict(),
    test_cfg=(
        dict(
            mode='slide',
            crop_size=(crop_size, crop_size),
            stride=(infer_stride, infer_stride),
        )
        if slide_infer_mode
        else dict(mode='whole')
    ),
)


train_pipeline = [
    dict(type="LoadImageFromFile", color_type=color_type, imdecode_backend="cv2"),
    dict(type="LoadSegAnnotations", reduce_zero_label=False),
    dict(
        type="RandomRescale",
        base_scale_factor=base_scale_factor,
        scale_range=scale_range,
        prob=random_rescale_prob,
    ),
    dict(type="SegRegionSampler", region_size=crop_size),
    dict(type="RandomFlip", prob=random_flip_prob, direction="horizontal"),
    dict(
        type="PackSegInputsVisaion",
        meta_keys=[
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "coord",
            "sample_idx",
            "dataset_metainfo",
        ],
    ),
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(
        type="VisaionInfiniteSampler",
        shuffle=True,
        bg_ratio=bg_ratio,
        batch_size=batch_size,
    ),
    dataset=dict(
        type="VisaionSegDataset",
        metainfo=meta_info,
        data_root=train_data_root,
        pipeline=train_pipeline,
    ),
)
val_pipeline = [
    dict(type="LoadImageFromFile", color_type=color_type, imdecode_backend="cv2"),
    dict(type="Resize", scale_factor=base_scale_factor, keep_ratio=True),
    dict(type="LoadSegAnnotations"),
    dict(
        type="PackSegInputs",
        _scope_="mmseg",
        meta_keys=[
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "dataset_metainfo",
        ],
    ),
]
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="VisaionSegDataset",
        metainfo=meta_info,
        data_root=val_data_root,
        pipeline=val_pipeline,
    ),
)
test_pipeline = [
    dict(type="LoadImageFromFile", color_type=color_type, imdecode_backend="cv2"),
    dict(type="Resize", scale_factor=base_scale_factor, keep_ratio=True),
    dict(type="LoadSegAnnotations"),
    dict(
        type="PackSegInputs",
        _scope_="mmseg",
        meta_keys=[
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "dataset_metainfo",
        ],
    ),
]
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="VisaionSegDataset",
        metainfo=meta_info,
        data_root=test_data_root,
        pipeline=test_pipeline,
    ),
)


val_evaluator = [
    dict(
        type="IoUMetric",
        iou_metrics=["mIoU"],
    ),
    dict(
        type="VisaionMetric",
        iou_threshold=iou_threshold,
        min_area=min_area,
    ),
]
test_evaluator = [
    dict(
        type="VisaionMetricHttp",
        iou_threshold=iou_threshold,
        min_area=min_area,
        output_dir=_work_dir,
    )
]


visualizer = dict(
    type="SegLocalVisualizer",
    _scope_="mmseg",
    vis_backends=[dict(type="LocalVisBackend", _scope_="mmseg")],
    name="visualizer",
)


optim_wrapper = dict(
    type="OptimWrapper",
    _scope_="mmseg",
    optimizer=dict(type="AdamW", _scope_="mmseg", lr=lr, weight_decay=weight_decay),
    clip_grad=None,
)


param_scheduler = dict(
    type="MultiStepLR",
    _scope_="mmseg",
    milestones=[int(0.5 * max_iters), int(0.75 * max_iters), int(0.875 * max_iters)],
    by_epoch=False,
    gamma=0.1,
)


train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=max_iters,
    val_interval=val_interval,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend=backend),
)


default_hooks = dict(
    timer=dict(type="IterTimerHook", _scope_="mmseg"),
    logger=dict(
        type="LoggerHook",
        _scope_="mmseg",
        interval=log_print_interval,
        log_metric_by_epoch=False,
    ),
    param_scheduler=dict(type="ParamSchedulerHook", _scope_="mmseg"),
    checkpoint=dict(
        type="CheckpointHook",
        _scope_="mmseg",
        by_epoch=False,
        interval=val_interval,
        filename_tmpl="checkpoint_{:06d}.pth",
        save_last=True,
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook", _scope_="mmseg"),
    visualization=dict(type="SegVisualizationHook", _scope_="mmseg"),
)
custom_hooks = []


default_scope = "mmseg"
work_dir = _work_dir
log_processor = dict(by_epoch=False, type="LogProcessor", window_size=10)
log_level = "INFO"
resume = False
randomness = dict(seed=1024, diff_rank_seed=False, deterministic=False)
experiment_name = _experiment_name
