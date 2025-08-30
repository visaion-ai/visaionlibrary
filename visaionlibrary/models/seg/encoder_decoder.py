# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmseg.models.utils import resize
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix, ForwardResults)

from mmseg.models.segmentors.base import BaseSegmentor
from mmengine.registry import MODELS
import cv2, os


@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSampel`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        infer_batch_size: 推理时候的Batch大小，默认是1，推荐设置成与训练相同大小的batch
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 infer_batch_size=1):
        super(EncoderDecoder, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._infer_batch_size = infer_batch_size  # 批量推理计算的时候，一个batch里面有多少小图
        self.trans_stride = 9  # 这个不作为模型外部初始化的参数，只有调用新Loop的时候才改动
        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    @staticmethod
    def get_batch_img_metas(inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        return batch_img_metas

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_img_metas = self.get_batch_img_metas(inputs, data_samples)
        seg_logits = self.inference(inputs, batch_img_metas)

        # add background channel when out channel is 1
        # this operation makes the output channels always greater than 1
        if self.decode_head.out_channels == 1:
            seg_logits = torch.cat((1-seg_logits, seg_logits), dim=1)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def stability_test(self, inputs: Tensor,  data_samples: OptSampleList = None):
        # 稳定性测试现在是中图级别了,只需要变出来一堆平移和旋转后的中图，然后全部输出检测结果就行，返回的都只有指标了，中图信息太多了
        b, _, h_img, w_img = inputs.size()
        assert b == 1  # 稳定性测试和val一样必须是单样本的模式
        label_map = data_samples[0].gt_sem_seg.data
        if label_map.max() == 0:  # 负样本直接跳过
            return []

        area_list = []
        prob_list = []

        h_stride, w_stride = self.test_cfg.stride

        h_grids = h_stride // self.trans_stride  # 首先，单边pad stride之后就可以遍历所有情况了，超过stride就是下一个周期的
        w_grids = w_stride // self.trans_stride

        for h_index in range(h_grids):
            for w_index in range(w_grids):
                padded = torch.zeros((inputs.size(0),
                                      inputs.size(1),
                                      inputs.size(2) + h_stride,
                                      inputs.size(3) + w_stride)).cuda()
                h_start = h_index * self.trans_stride
                w_start = w_index * self.trans_stride
                padded[:, :, h_start:h_start + h_img, w_start:w_start + w_img] = inputs  # 通过填充的方式实现中图的平移
                batch_img_metas = [dict(ori_shape=padded.shape[2:],
                                        img_shape=padded.shape[2:],
                                        pad_shape=padded.shape[2:],
                                        padding_size=[0, 0, 0, 0])]
                prob_map = self.inference(padded, batch_img_metas)[:, :, h_start:h_start + h_img, w_start:w_start + w_img]
                prob_map = prob_map[:, :, 3:-3, 3:-3]  # 干掉边缘过检
                prob_map, seg_map = prob_map.max(dim=1)
                seg_map = (seg_map != 0) * 1.  # 不管类别，全部按照单类算
                area_list.append(seg_map.sum().item())
                prob_list.append(((seg_map * prob_map).sum() / (seg_map.sum() + 1e-8)).item())
                # name = os.path.basename(data_samples[0].img_path[0])
                # cv2.imwrite(f'/root/data/mmcvlab/lck/workdirs/stable_show/{name}_h{h_index}g{w_index}.png',
                #             seg_map.squeeze().cpu().numpy() * 255)

        return np.array(area_list), np.array(prob_list)

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                kwargs: dict = None) -> ForwardResults:
        """The unified entry for a forward process in both training and test.
        The method should accept three modes: "tensor", "predict" and "loss":
        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.
        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.
        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.
            kwargs (dict): 一系列用在deploy过程中的参数，主要包括：
                    1. output_mode: 'raw' / 'argmax'
                        raw: 直接返回原始模型输出，不过带了sigmoid
                        argmax: 会在sigmoid的基础上，进行argmax，返回label+置信度图
                    2. threshold: 二值化阈值
        Returns:
            The return type depends on ``mode``.
            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
            - If ``mode="deploy"``, forward for deployment.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'deploy':
            if kwargs:
                return self.deploy(inputs, **kwargs)
            else:
                return self.deploy(inputs)
        elif mode == 'stability_test':
            return self.stability_test(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def deploy(self,
               inputs: Tensor,
               output_mode='raw'):
        """
        目前输出模式支持两种：
        “raw”：
        “argmax”：
        """
        height, width = inputs.shape[-2:]
        # deploy mode 的 preprocessor
        x = self.data_preprocessor.deploy(inputs)
        # 网络结构的正常 forward
        x = self._forward(x)

        # when height&width of output is not consistent with the input, resize for consistency
        out_height, out_width = x.shape[-2:]
        if out_height != height and out_width != width:
            x = resize(
                input=x,
                size=(height, width),
                mode='bilinear',
                align_corners=self.align_corners
            )

        # when only one channel for prediction, sigmoid_before_loss must be true
        if self.decode_head.out_channels == 1:
            assert self.decode_head.sigmoid_before_loss, 'decode_head.sigmoid_before_loss must be true if prediction channel is 1.'
            x = x.sigmoid()
            x = torch.cat((1-x, x), dim=1)  # add background score to the first channel
        else:
            if self.decode_head.sigmoid_before_loss:
                x = x.sigmoid()
            else:
                x = x.softmax(dim=1)

        if output_mode == 'raw':
            return x
        elif 'argmax' in output_mode:
            scores, indexes = x.max(dim=1, keepdim=True)
            indexes = indexes.to(torch.uint8) if 'int8' in output_mode else indexes.to(torch.float32)
            return scores, indexes
        elif 'onlylabel' in output_mode:
            _, indexes = x.max(dim=1, keepdim=True)
            indexes = indexes.to(torch.uint8) if 'int8' in output_mode else indexes.to(torch.float32)
            return indexes
        else:
            raise ValueError(f'Unsupported output_mode: {output_mode}.')

    @staticmethod
    def _crop_region_in_tensor(img, x, y, region_size, segmap_size):  # 根据随机出来的y和x进行region的裁剪
        """根据随机出来的x和y进行region的裁剪"""
        region_h, region_w = region_size  # 需要切分的小图的分辨率
        segmap_h, segmap_w = segmap_size  # 分割结果segmap的分辨率

        # pad_w 计算的是水平方向上，Crop出来的小图和Gt的尺寸的差异
        pad_w = int((region_w - segmap_w) / 2.)
        # pad_h 计算的是垂直方向上，Crop出来的小图和Gt的尺寸的差异
        pad_h = int((region_h - segmap_h) / 2.)

        # 这个是根据x和pad_w计算图像左边需要pad的宽度，当x > pad_w的时候，意味着左边不用pad,就得到的0，不然就需要pad
        pad_w_l = max(0, pad_w - x)
        # 这个是根据x和pad_w计算图像左边需要pad的宽度，
        pad_w_r = max(0, x - pad_w + region_w - img.size(3))
        # 高度方向的逻辑同上
        pad_h_t = max(0, pad_h - y)
        pad_h_b = max(0, y - pad_h + region_h - img.size(2))

        tmp = F.pad(img, [pad_w_l, pad_w_r, pad_h_t, pad_h_b])

        x_ = x - pad_w + pad_w_l  # x, y 都是segmap分割结果在原图坐标系下的坐标
        y_ = y - pad_h + pad_h_t  # x_, y_ 就是region在pad后的图像下的坐标，因此需要减掉pad再加上在原图左上角pad的大小
        region = tmp[:, :, y_:y_ + region_h, x_:x_ + region_w]
        return region

    def slide_inference(self,
                        inputs: Tensor,
                        batch_img_metas: List[dict],
                        ) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        h_stride, w_stride = self.test_cfg.stride
        label_h, label_w = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        num_head_out_channels = self.decode_head.out_channels
        assert h_img >= self.test_cfg.crop_size[0]
        assert w_img >= self.test_cfg.crop_size[1]
        # # 测试阶段由于前面的preprocessor已经不会有更小的图了
        # pad_h_b = int(max(0, label_h - original_h_img))
        # pad_w_r = int(max(0, label_w - original_w_img))
        # inputs = F.pad(inputs, [0, pad_w_r, 0, pad_h_b])
        # batch_size, _, h_img, w_img = inputs.size()  # 重新计算尺寸

        # 准备preds用来保留预测结果
        preds = inputs.new_zeros((batch_size, num_head_out_channels, h_img, w_img))
        # count_mat是保留每个像素点的计算次数
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        # 计算水平和垂直方向要切分几次
        h_grids = int(max(h_img - label_h + h_stride - 1, 0) // h_stride + 1)
        w_grids = int(max(w_img - label_w + w_stride - 1, 0) // w_stride + 1)

        # 计算一共要切几个小图，方便和前面指定的batch_size进行比较
        grid_num = h_grids * w_grids

        # 把这些小图，按照batsize的大小分开，分批进行推理，这边生成的grid_batches里面存的就是每一次批量推理时候，这批小图的id
        grid_batches = [range(i, min(i+self._infer_batch_size, grid_num))
                        for i in range(0, grid_num, self._infer_batch_size)]

        for grid_batch in grid_batches:  # grid_batch就是这批小图的空间id
            crop_imgs = []  # 用来存储切出来的小图
            crop_locs = []  # 用来存储小图的坐标信息
            for grid_idx in grid_batch:
                h_idx = grid_idx // w_grids
                w_idx = grid_idx % w_grids
                y1 = h_idx * h_stride  # x1 y1 是初步的左上角点
                x1 = w_idx * w_stride
                y2 = int(min(y1 + label_h, h_img))  # x2和y2是计算之后得到的右下角点
                x2 = int(min(x1 + label_w, w_img))
                y1 = int(max(y2 - label_h, 0))  # 再算一次是因为可能x2和y2存在越界的可能
                x1 = int(max(x2 - label_w, 0))  # 所以x1和y1可能比初步计算的结果要小一点
                crop_img = self._crop_region_in_tensor(inputs, x1, y1,
                                                       region_size=self.test_cfg.crop_size,
                                                       segmap_size=(label_h, label_w))
                crop_imgs.append(crop_img)  # 先把小图存起来
                crop_locs.append([x1, y1])  # 对应裁剪的位置也存起来

            crop_imgs = torch.cat(crop_imgs, dim=0)  # 在Batch维度把他拼起来
            # change the image shape to patch shape
            batch_img_metas[0]['img_shape'] = crop_imgs.shape[2:]   # 原版
            batch_img_metas[0]['pad_shape'] = crop_imgs.shape[2:]   # 原版
            crop_seg_logit = self.encode_decode(crop_imgs, batch_img_metas)

            for i in range(len(crop_locs)):
                xi, yi = crop_locs[i]
                if self.decode_head.out_channels == 1:
                    assert self.decode_head.sigmoid_before_loss, \
                        'decode_head.sigmoid_before_loss must be true if prediction channel is 1.'
                    preds[:, :, yi:yi + label_h, xi:xi + label_w] += crop_seg_logit[i].sigmoid()
                else:
                    if self.decode_head.sigmoid_before_loss:
                        preds[:, :, yi:yi + label_h, xi:xi + label_w] += crop_seg_logit[i].sigmoid()
                    else:
                        preds[:, :, yi:yi + label_h, xi:xi + label_w] += crop_seg_logit[i].softmax(dim=0)
                count_mat[:, :, yi:yi + label_h, xi:xi + label_w] += 1

        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def center_slide_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window without overlap, by just save the center result of segmentation
        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image meta info where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logit from model of each
                input image.
        """
        # Need to pad the inputs before inference
        # h_stride, w_stride = self.test_cfg.stride
        label_h, label_w = self.test_cfg.crop_size
        pad_h, pad_w = int(label_h / 8), int(label_w / 8)
        # the stride is deside by pad_out, not manually set
        h_stride, w_stride = label_h - 2 * pad_h, label_w - 2 * pad_w

        batch_size, _, h_img, w_img = inputs.size()
        num_head_out_channels = self.decode_head.out_channels
        assert h_img >= self.test_cfg.crop_size[0]
        assert w_img >= self.test_cfg.crop_size[1]

        inputs = F.pad(inputs, [pad_w, pad_w, pad_h, pad_h], value=-1)
        batch_size, _, h_img, w_img = inputs.size()  # 重新计算尺寸

        # 准备pred用来保留预测结果
        pred = inputs.new_zeros((batch_size, num_head_out_channels, h_img, w_img))
        # count_mat是保留每个像素点的计算次数
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        # 计算水平和垂直方向要切分几次
        h_grids = int(max(h_img - label_h + h_stride - 1, 0) // h_stride + 1)
        w_grids = int(max(w_img - label_w + w_stride - 1, 0) // w_stride + 1)

        # 计算一共要切几个小图，方便和前面指定的batch_size进行比较
        grid_num = h_grids * w_grids

        # 把这些小图，按照batch size的大小分开，分批进行推理，这边生成的grid_batches里面存的就是每一次批量推理时候，这批小图的id
        grid_batches = [range(i, min(i+self._infer_batch_size, grid_num))
                        for i in range(0, grid_num, self._infer_batch_size)]

        for grid_batch in grid_batches:  # grid_batch就是这批小图的空间id
            crop_img_list = []  # 用来存储切出来的小图
            crop_locs = []  # 用来存储小图的坐标信息
            for grid_idx in grid_batch:
                h_idx = grid_idx // w_grids
                w_idx = grid_idx % w_grids
                y1 = h_idx * h_stride  # x1 y1 是初步的左上角点
                x1 = w_idx * w_stride
                y2 = int(min(y1 + label_h, h_img))  # x2和y2是计算之后得到的右下角点
                x2 = int(min(x1 + label_w, w_img))
                y1 = int(max(y2 - label_h, 0))  # 再算一次是因为可能x2和y2存在越界的可能
                x1 = int(max(x2 - label_w, 0))  # 所以x1和y1可能比初步计算的结果要小一点
                crop_img = self._crop_region_in_tensor(inputs, x1, y1,
                                                       region_size=self.test_cfg.crop_size,
                                                       segmap_size=(label_h, label_w))
                crop_img_list.append(crop_img)  # 先把小图存起来
                crop_locs.append([x1, y1])  # 对应裁剪的位置也存起来

            crop_img_list = torch.cat(crop_img_list, dim=0)  # 在Batch维度把他拼起来
            # change the image shape to patch shape
            batch_img_metas[0]['img_shape'] = crop_img_list.shape[2:]   # 原版
            batch_img_metas[0]['pad_shape'] = crop_img_list.shape[2:]   # 原版
            crop_seg_logit = self.encode_decode(crop_img_list, batch_img_metas)[:, :, pad_h:-pad_h, pad_w:-pad_w]

            for i in range(len(crop_locs)):
                xi, yi = crop_locs[i]
                if self.decode_head.out_channels == 1:
                    assert self.decode_head.sigmoid_before_loss, \
                        'decode_head.sigmoid_before_loss must be true if prediction channel is 1.'
                    pred[:, :, yi+pad_h:yi+label_h-pad_h, xi+pad_w:xi+label_w-pad_w] += crop_seg_logit[i].sigmoid()
                else:
                    if self.decode_head.sigmoid_before_loss:
                        pred[:, :, yi+pad_h:yi+label_h-pad_h, xi+pad_w:xi+label_w-pad_w] += crop_seg_logit[i].sigmoid()
                    else:
                        pred[:, :, yi+pad_h:yi+label_h-pad_h, xi+pad_w:xi+label_w-pad_w] += crop_seg_logit[i].softmax(dim=0)
                count_mat[:, :, yi+pad_h:yi+label_h-pad_h, xi+pad_w:xi+label_w-pad_w] += 1
        count_mat = count_mat[:, :, pad_h:-pad_h, pad_w:-pad_w]
        pred = pred[:, :, pad_h:-pad_h, pad_w:-pad_w]
        assert (count_mat == 0).sum() == 0
        seg_logit = pred / count_mat

        return seg_logit

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)
        if self.decode_head.out_channels == 1:
            assert self.decode_head.sigmoid_before_loss, 'decode_head.sigmoid_before_loss must be true if prediction channel is 1.'
            seg_logits = seg_logits.sigmoid()
        else:
            if self.decode_head.sigmoid_before_loss:
                seg_logits = seg_logits.sigmoid()
            else:
                seg_logits = seg_logits.softmax(dim=1)
        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole', 'center_slide']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            # 普通滑窗模式的推理
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        elif self.test_cfg.mode == 'center_slide':
            seg_logit = self.center_slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)
        # 这两个新的inference里边，已经完成了激活的操作，和deploy时候的顺序是一致的
        return seg_logit

    def stability_test_step(self, data: Union[tuple, dict, list], trans_stride: int) -> list:
        self.trans_stride = trans_stride
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='stability_test')
