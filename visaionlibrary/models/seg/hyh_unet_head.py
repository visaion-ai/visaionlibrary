import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer, build_norm_layer)
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from .decode_head import BaseDecodeHead

from mmengine.registry import MODELS

eps = 1e-5


class BasicBlock(BaseModule):
    """
    Similar to BasicBlock in ResNet
    """

    def __init__(self,
                 in_channels,
                 block_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.block_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, self.in_channels, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            self.block_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            self.block_channels,
            self.in_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.act = build_activation_layer(self.act_cfg)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = self.drop_path(out)

        out += identity

        out = self.act(out)

        return out


class UpSample(BaseModule):

    def __init__(self,
                 in_channels=None,
                 mode='interpolate',
                 interpolate_mode='nearest',
                 upsample_factor=2
                 ):
        super(UpSample, self).__init__()
        assert mode in ['interpolate', 'deconv']
        if mode == 'deconv':
            assert in_channels is not None
        if mode == 'interpolate':
            assert interpolate_mode in ['nearest', 'bilinear', 'bicubic']

        self.in_channels = in_channels
        self.mode = mode
        self.upsample_factor = upsample_factor
        self.interpolate_mode = interpolate_mode

        if mode == 'interpolate':
            self.upsample = nn.Upsample(scale_factor=self.upsample_factor, mode=self.interpolate_mode)
        elif mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=self.in_channels,
                                               out_channels=self.in_channels,
                                               kernel_size=3,
                                               stride=self.upsample_factor,
                                               padding=1,
                                               bias=True)
        else:
            raise ValueError

    def forward(self, x):
        if self.mode == 'interpolate':
            out = self.upsample(x)
        elif self.mode == 'deconv':
            shape = list(x.shape)
            shape[-2] = shape[-2] * self.upsample_factor
            shape[-1] = shape[-1] * self.upsample_factor
            out = self.upsample(x, output_size=shape)
        else:
            raise ValueError

        return out


class DecoderStage(BaseModule):
    def __init__(self,
                 in_channels,
                 block_channels,
                 skip_channels=0,
                 num_blocks=1,
                 block_drop_path_rate=0.,
                 upsample_mode='interpolate',
                 upsample_interpolate_mode='nearest',
                 upsample_factor=2,
                 enable_fusion=True,
                 fusion_mode='concate',
                 fusion_density_mode='common',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 ):
        super(DecoderStage, self).__init__()
        self.in_channels = in_channels
        self.block_channels = block_channels
        self.skip_channels = skip_channels
        self.num_blocks = num_blocks
        self.block_drop_path_rate = block_drop_path_rate
        self.upsample_mode = upsample_mode
        self.upsample_interpolate_mode = upsample_interpolate_mode
        self.upsample_factor = upsample_factor
        self.enable_fusion = enable_fusion
        assert fusion_mode in ['concate', 'add']
        self.fusion_mode = fusion_mode
        assert fusion_density_mode in ['common', 'dense']
        self.fusion_density_mode = fusion_density_mode
        # 当fusion_mode = concate 的情况下， fusion_density_mode 只能是 common
        if self.fusion_mode == 'concate':
            assert self.fusion_density_mode == 'common'

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # 如果是add，需要把skip和input的channel数对齐
        if self.enable_fusion:
            if self.fusion_mode == 'add':
                self.skip_align = ConvModule(
                    in_channels=self.skip_channels,
                    out_channels=self.block_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
                self.input_align = ConvModule(
                    in_channels=self.in_channels,
                    out_channels=self.block_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            else:  # self.fusion_mode == 'concate'
                self.concate_align = ConvModule(
                    in_channels=self.in_channels + self.skip_channels,
                    out_channels=self.block_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
        else:
            self.input_align = ConvModule(
                in_channels=self.in_channels,
                out_channels=self.block_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )

        self._make_stage()
        assert len(self.stage_list) == (num_blocks + 1)

    def _make_stage(self):
        self.stage_list = nn.ModuleList()
        for i in range(self.num_blocks):
            block = BasicBlock(
                in_channels=self.block_channels,
                block_channels=self.block_channels,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                drop_path_rate=self.block_drop_path_rate,
                act_cfg=self.act_cfg,
            )
            self.stage_list.append(block)

        upsample = UpSample(
            in_channels=self.block_channels,
            mode=self.upsample_mode,
            interpolate_mode=self.upsample_interpolate_mode,
            upsample_factor=self.upsample_factor
        )
        self.stage_list.append(upsample)

    def forward(self, x, skip=None):
        if skip is None:
            assert not self.enable_fusion

        if self.enable_fusion:
            if self.fusion_mode == 'add':
                skip_align = self.skip_align(skip)
                x = self.input_align(x)

            for i in range(self.num_blocks):
                if self.fusion_mode == 'concate':
                    if i == 0:
                        assert self.fusion_density_mode == 'common'
                        x = torch.cat([x, skip], dim=1)
                        x = self.concate_align(x)
                        x = self.stage_list[i](x)
                    else:
                        x = self.stage_list[i](x)
                else:  # self.fusion_mode == 'add'
                    if self.fusion_density_mode == 'common':
                        if i == 0:
                            x = x + skip_align
                        x = self.stage_list[i](x)
                    else:  # self.fusion_density_mode == 'dense'
                        x = x + skip_align
                        x = self.stage_list[i](x)
        else:
            x = self.input_align(x)
            for i in range(self.num_blocks):
                x = self.stage_list[i](x)

        out = self.stage_list[-1](x)
        return out


@MODELS.register_module()
class HYHUNetHead(BaseDecodeHead):
    def __init__(self,
                 prediction_channels=128,
                 num_classes=2,
                 fusion_mode='concate',
                 fusion_density_mode='common',
                 block_drop_path_rate=0.,
                 upsample_mode='interpolate',
                 upsample_interpolate_mode='nearest',
                 num_block_list=[1, 1, 1, 1],
                 block_channel_list=[256, 128, 64, 32],
                 input_channel_list=[64, 128, 256, 512],
                 last_scale_factor=2,
                 is_dual=False,
                 dual_weight=10,
                 threshold=0.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=[  # 怎么设置：https://mmengine.readthedocs.io/en/latest/advanced_tutorials/initialize.html
                     dict(type='Kaiming', layer=['Conv2d', 'ConvTranspose2d']),
                 ],
                 **kwargs
                 ):
        super().__init__(in_channels=0, channels=prediction_channels, num_classes=num_classes, init_cfg=init_cfg, **kwargs)

        assert len(input_channel_list) == len(block_channel_list)
        assert len(block_channel_list) == len(num_block_list)

        self.prediction_channels = prediction_channels
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self.fusion_density_mode = fusion_density_mode
        self.block_drop_path_rate = block_drop_path_rate
        self.upsample_mode = upsample_mode
        self.upsample_interpolate_mode = upsample_interpolate_mode
        self.num_block_list = num_block_list
        self.block_channel_list = block_channel_list
        self.input_channel_list = input_channel_list
        self.last_scale_factor = last_scale_factor
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.is_dual = is_dual
        self.dual_weight = dual_weight
        self.num_stages = len(num_block_list)
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = DecoderStage(
                in_channels=self.input_channel_list[self.num_stages - 1] if i == 0 else self.block_channel_list[i - 1],
                block_channels=self.block_channel_list[i],
                skip_channels=0 if i == 0 else self.input_channel_list[self.num_stages - 1 - i],
                num_blocks=self.num_block_list[i],
                block_drop_path_rate=self.block_drop_path_rate,
                upsample_mode=self.upsample_mode,
                upsample_interpolate_mode=self.upsample_interpolate_mode,
                upsample_factor=self.last_scale_factor if i == (self.num_stages - 1) else 2,
                enable_fusion=False if i == 0 else True,
                fusion_mode=self.fusion_mode,
                fusion_density_mode=self.fusion_density_mode,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            self.decoder_stages.append(stage)

        # 构造预测层
        self.final_conv = ConvModule(
            in_channels=self.block_channel_list[-1],
            out_channels=self.prediction_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.conv_seg = nn.Conv2d(self.prediction_channels, self.out_channels, kernel_size=(1, 1), padding=0)

    def forward(self, feature_maps):
        assert len(feature_maps) == self.num_stages
        last_feature_map = feature_maps[-1]
        for i in range(self.num_stages):
            if i == 0:
                out = self.decoder_stages[i](last_feature_map)
            else:
                out = self.decoder_stages[i](out, feature_maps[self.num_stages - 1 - i])
        out = self.final_conv(out)
        out = self.conv_seg(out)
        return out

    @staticmethod
    def tran_sim_loss(seg_logits, batch_data_samples):
        batch_size = seg_logits.size(0) // 2
        real_seg_logits = seg_logits[:batch_size]
        dual_seg_logits = seg_logits[batch_size:]
        all_kl_loss = []
        for i in range(batch_size):
            if len(batch_data_samples[i].overlap_region) == 0:
                continue
            x_loc_orig, y_loc_orig, x_loc_dual, y_loc_dual = batch_data_samples[i].overlap_region
            real_seg_logits_overlap = real_seg_logits[i, :, y_loc_orig, x_loc_orig]
            dual_seg_logits_overlap = dual_seg_logits[i, :, y_loc_dual, x_loc_dual]
            kl = F.kl_div((dual_seg_logits_overlap.softmax(dim=0) + 1e-10).log(),
                          (real_seg_logits_overlap.softmax(dim=0) + 1e-10), reduction='mean')
            all_kl_loss.append(kl)
        if len(all_kl_loss) > 0:
            mean_kl_loss = sum(all_kl_loss) / len(all_kl_loss)
            return mean_kl_loss
        else:
            return 0

    def loss(self, inputs, batch_data_samples, train_cfg):
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        if self.is_dual:
            losses['loss_match'] = self.tran_sim_loss(seg_logits, batch_data_samples) * self.dual_weight
        return losses
