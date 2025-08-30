"""
RTMDetInsSepBNHead for Visaion
在rtmdet_s_ins模板中使用了RTMDetInsSepBNHead, 在使用过程中发现, 其父类RTMDetInsHead中_bbox_mask_post_process方法的中的
if rescale:
    ori_h, ori_w = img_meta['ori_shape'][:2]
    mask_logits = F.interpolate(
        mask_logits,
        size=[
            math.ceil(mask_logits.shape[-2] * scale_factor[0]),
            math.ceil(mask_logits.shape[-1] * scale_factor[1])
        ],
        mode='bilinear',
        align_corners=False)[..., :ori_h, :ori_w]

有错误, 需要修改
"""
from typing import Optional
import math

import torch
import torch.nn.functional as F

from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from mmengine.registry import MODELS
from mmdet.utils import ConfigType
from mmdet.structures.bbox import (get_box_tensor, get_box_wh, scale_boxes)
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsSepBNHead


@MODELS.register_module()
class VisaionRTMDetInsSepBNHead(RTMDetInsSepBNHead):
    def _bbox_mask_post_process(
            self,
            results: InstanceData,
            mask_feat,
            cfg: ConfigType,
            rescale: bool = False,
            with_nms: bool = True,
            img_meta: Optional[dict] = None) -> InstanceData:
        """bbox and mask post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        stride = self.prior_generator.strides[0][0]
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        assert with_nms, 'with_nms must be True for RTMDet-Ins'
        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

            # process masks
            mask_logits = self._mask_predict_by_feat_single(
                mask_feat, results.kernels, results.priors)

            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0), scale_factor=stride, mode='bilinear')
            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                mask_logits = F.interpolate(
                    mask_logits,
                    size=[
                        math.ceil(mask_logits.shape[-2] * scale_factor[1]),
                        math.ceil(mask_logits.shape[-1] * scale_factor[0])
                    ],
                    mode='bilinear',
                    align_corners=False)[..., :ori_h, :ori_w]
            masks = mask_logits.sigmoid().squeeze(0)
            masks = masks > cfg.mask_thr_binary
            results.masks = masks
        else:
            h, w = img_meta['ori_shape'][:2] if rescale else img_meta[
                'img_shape'][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w),
                dtype=torch.bool,
                device=results.bboxes.device)

        return results