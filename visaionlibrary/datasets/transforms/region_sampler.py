import copy
from typing import Dict, List, Union, Tuple, Optional
import numpy as np
import random
import cv2
import torch

from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms.transforms import RandomCrop
from mmdet.structures.bbox import autocast_box_type
from mmengine.registry import TRANSFORMS
from visaionlibrary.datasets.utils import SampleStatus


@TRANSFORMS.register_module()
class SegRegionSampler(BaseTransform):
    """Region Sampler for Train phase.
        通过在缺陷区域随机裁剪图像进行模型的训练

    
    Required Keys:

    - seg_map_path: Path to the segmentation map file.  TODO:
    
    Modified Keys:

    
    Add Keys:
    - gt_seg_map(np.uint8): Ground truth segmentation map.  TODO:


    Args:
        region_size: 小图的尺寸,(高, 宽)
        patience: 在正样本里面取负样本小图的时候,最多尝试的次数,如果设置成1就是随机裁剪,不管里面是否包含前景,默认为1
        pad_mode: pad填充模式
        centerness_target:目标是否基本在图像中央
        dual_mode:是否额外裁剪出一个样本来改善平移稳定性
    """
    def __init__(self,
                 region_size: int = 256,
                 patience: int = 3,
                 pad_mode: str = 'constant',
                 centerness_target: bool = False,
                 dual_mode: bool = False,
                 defect_ratio: float = 0.0) -> None:
        self._region_h = region_size
        self._region_w = region_size
        assert 1 <= patience <= 5  # 限定patience在[1,5]之间,太大就太慢了
        self._patience = patience
        assert pad_mode in ['constant', 'reflect', 'wrap']  # 这个pad_mode暂时是失效的
        self.centerness_target = centerness_target  # 缺陷和过检区域是否大部分在图像中央
        self.dual_mode = dual_mode
        self.defect_ratio = defect_ratio  # 缺陷区域的最小保留比例, 默认0

    def _crop_region_and_gt(self, img, gt_seg_map, gt_top_left_x, gt_top_left_y):
        """根据随机出来的gt_top_left_y和gt_top_left_x进行region和gt的裁剪"""
        gt_top_left_x = int(gt_top_left_x)
        gt_top_left_y = int(gt_top_left_y)

        # 这个是根据x和pad_w计算图像左边需要pad的宽度,当x > pad_w的时候,意味着左边不用pad,就得到的0,不然就需要pad
        pad_w_l = max(0, -gt_top_left_x)
        # 这个是根据x和pad_w计算图像左边需要pad的宽度,
        pad_w_r = max(0, gt_top_left_x + self._region_w - img.shape[1])
        # 高度方向的逻辑同上
        pad_h_t = max(0, -gt_top_left_y)
        pad_h_b = max(0, gt_top_left_y + self._region_h - img.shape[0])

        # 根据计算得到的需要pad的量处理原始图像
        padded_img = np.pad(img, ((pad_h_t, pad_h_b), (pad_w_l, pad_w_r), (0, 0)))
        padded_gt = np.pad(gt_seg_map, ((pad_h_t, pad_h_b), (pad_w_l, pad_w_r)))

        # region_top_left_x/y 计算的是在pad后的图像坐标系下,裁剪region时候左上角坐标
        region_top_left_x = gt_top_left_x + pad_w_l
        region_top_left_y = gt_top_left_y + pad_h_t
        region = padded_img[region_top_left_y:region_top_left_y + self._region_h,
                            region_top_left_x:region_top_left_x + self._region_w, :]
        gt = padded_gt[region_top_left_y:region_top_left_y + self._region_h,
                       region_top_left_x:region_top_left_x + self._region_w]
        return region, gt

    def transform(self, results: dict) -> dict:
        """裁剪小图,根据其中的crop_mode来指定具体裁哪,是只裁前景位置还是裁背景位置

        Args:
            results(dict): Result dict from
                :class:`BaseTransform`

        Returns:
            dict: 返回裁剪后的字典.

        """
        sample_type = results['sample_type']    # 数据集中第i个数据的类型
        gt_seg_map = results['gt_seg_map']    # 获取标注图片
        img = results['img']                 # 获取图片
        # 获取shape,要直接从img计算而不是直接拿results['img_shape'],因为前面的增广过程可能会修改size
        img_h, img_w = img.shape[:2]

        # 1. 要么是sampler用错了,不是randomNG类的导致没这个字段
        # 2. 要么是dataset初始化读取图像类型的时候错误使用了后续的transform,在sampler之前就调用了本函数,也没这个字段
        if results.get('crop_mode', None) is None:
            # 但是这两种情况,都会有sample_type,因为就算是情况2也刚经过LoadAnn,所以根据type指定mode,就是退化了而已
            crop_mode = "label" if sample_type == SampleStatus.FOREGROUND else "background"
        else:
            crop_mode = results['crop_mode']  # 裁剪小图的模式,label:只裁有标注的地方,background:尽可能只在背景进行裁剪

        if gt_seg_map.max() > 0:  # 确定是缺陷样本
            if crop_mode == 'label':  # 正常裁剪正样本小图
                # 正样本就需要先pad,避免过小的图像出现,或者出现太过靠近边界的缺陷导致过多的pad影响训练
                pad_h = int(max(self._region_h - img_h, 0))
                pad_w = int(max(self._region_w - img_w, 0))
                pad_h_t = random.randint(0, pad_h)
                pad_h_b = pad_h - pad_h_t
                pad_w_l = random.randint(0, pad_w)
                pad_w_r = pad_w - pad_w_l

                img = np.pad(img, ((pad_h_t, pad_h_b), (pad_w_l, pad_w_r), (0, 0)))
                gt_seg_map = np.pad(gt_seg_map, ((pad_h_t, pad_h_b), (pad_w_l, pad_w_r)))  # gt 同样的处理
                img_h, img_w = img.shape[:2]  # 重新计算尺寸
                if self.defect_ratio > 0:
                    contours, _ = cv2.findContours(gt_seg_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    contour = random.choice(contours)
                    tmp_map = np.zeros_like(gt_seg_map)
                    cv2.drawContours(tmp_map, [contour], 0, 1, -1)
                    defect_ys, defect_xs = np.nonzero(tmp_map)
                    defect_area = len(defect_xs)  # 缺陷区域面积
                    max_try = 20  # 最多尝试20次,如果都找不到,就跳出循环
                    for _ in range(max_try):
                        defect_y, defect_x = random.choice(list(zip(defect_ys, defect_xs)))
                        # 随机化该缺陷点在_gt_h*_gt_w范围内的相对坐标 从0波动到self._gt_h - 1这个范围
                        gt_top_left_y = defect_y.item() - random.randint(0, self._region_h - 1)
                        gt_top_left_x = defect_x.item() - random.randint(0, self._region_w - 1)

                        # 如果都在中央,就可以随机裁剪不做限制,因为小图中一定包含大量原始图像信息
                        # 如果目标不都是在中央,那正样本就需要进行过界判定,不能出图像的范围,不然可能导致裁剪质量下降
                        if not self.centerness_target:
                            gt_top_left_y = max(gt_top_left_y, 0)
                            gt_top_left_x = max(gt_top_left_x, 0)
                            gt_top_left_y = min(gt_top_left_y, img_h - self._region_h)  # 这个就是说y不能超过计算得到的最大值
                            gt_top_left_x = min(gt_top_left_x, img_w - self._region_w)  # x方向上的逻辑与y方向相同

                        region, gt = self._crop_region_and_gt(img, gt_seg_map, gt_top_left_x, gt_top_left_y)
                        if len(np.nonzero(gt)[0]) >= defect_area * self.defect_ratio:
                            break

                else:
                    defect_ys, defect_xs = np.nonzero(gt_seg_map)
                    defect_y, defect_x = random.choice(list(zip(defect_ys, defect_xs)))
                    # 随机化该缺陷点在_gt_h*_gt_w范围内的相对坐标 从0波动到self._gt_h - 1这个范围
                    gt_top_left_y = defect_y.item() - random.randint(0, self._region_h - 1)
                    gt_top_left_x = defect_x.item() - random.randint(0, self._region_w - 1)

                    # 如果都在中央,就可以随机裁剪不做限制,因为小图中一定包含大量原始图像信息
                    # 如果目标不都是在中央,那正样本就需要进行过界判定,不能出图像的范围,不然可能导致裁剪质量下降
                    if not self.centerness_target:
                        gt_top_left_y = max(gt_top_left_y, 0)
                        gt_top_left_x = max(gt_top_left_x, 0)
                        gt_top_left_y = min(gt_top_left_y, img_h - self._region_h)  # 这个就是说y不能超过计算得到的最大值
                        gt_top_left_x = min(gt_top_left_x, img_w - self._region_w)  # x方向上的逻辑与y方向相同

                    region, gt = self._crop_region_and_gt(img, gt_seg_map, gt_top_left_x, gt_top_left_y)

            else:  # 从正样本裁剪负样本小图
                gt_top_left_x, gt_top_left_y = None, None  # 事先定义几个None,来确保出了for之后,region和gt还能访问到
                region, gt = None, None
                for _ in range(self._patience):  # 最多尝试patience次,次数到了之后就算还是裁剪到了前景区域也不管了
                    gt_top_left_y = random.randint(0, img_h) - 0.5 * self._region_h  # gt_top_left_x/y是裁剪区域左上角点的坐标
                    gt_top_left_x = random.randint(0, img_w) - 0.5 * self._region_w
                    region, gt = self._crop_region_and_gt(img, gt_seg_map, gt_top_left_x, gt_top_left_y)
                    if gt.max() == 0:  # 如果裁到了负样本,就跳出循环
                        break

        else:
            # 背景样本的话,就是只限制裁剪小图的中心点不出原始图像的范围
            gt_top_left_y = random.randint(0, img_h) - 0.5 * self._region_h  # gt_top_left_x/y是裁剪区域左上角点的坐标
            gt_top_left_x = random.randint(0, img_w) - 0.5 * self._region_w
            region, gt = self._crop_region_and_gt(img, gt_seg_map, gt_top_left_x, gt_top_left_y)

        results['img'] = np.ascontiguousarray(region)   # 将region转换为连续的内存块
        results['img_shape'] = region.shape[:2]
        results['gt_seg_map'] = np.ascontiguousarray(gt)  # 将gt转换为连续的内存块
        results['coord'] = [gt_top_left_x, gt_top_left_y, self._region_h, self._region_w]

        if self.dual_mode:
            # Generating the dual sample, translated within 25% of crop size.
            trans_range = int(self._region_h / 4.)
            dual_trans_x = random.randint(1, trans_range) if random.random() > 0.5 else -random.randint(1, trans_range)
            dual_trans_y = random.randint(1, trans_range) if random.random() > 0.5 else -random.randint(1, trans_range)

            dual_gt_top_left_x = gt_top_left_x + dual_trans_x
            dual_gt_top_left_y = gt_top_left_y + dual_trans_y
            # temporary ignore the out-range problem
            dual_region, dual_gt = self._crop_region_and_gt(img, gt_seg_map, dual_gt_top_left_x, dual_gt_top_left_y)

            results['dual_img'] = dual_region
            results['dual_gt_seg_map'] = dual_gt

            # 对偶样本不进行旋转,确保角度较小情况下统计对应关系
            img_h, img_w = results['img_shape']
            # Find the corresponding area before rotation
            region_orig = [max(0, dual_trans_x), max(0, dual_trans_y),
                           min(img_w, img_w + dual_trans_x), min(img_h, img_h + dual_trans_y)]
            region_dual = [max(0, -dual_trans_x), max(0, -dual_trans_y),
                           min(img_w, img_w - dual_trans_x), min(img_h, img_h - dual_trans_y)]

            region_w = region_orig[2] - region_orig[0]  # 重叠区域的宽高
            region_h = region_orig[3] - region_orig[1]

            x_grid_orig, y_grid_orig = np.meshgrid(np.arange(region_orig[0], region_orig[0] + region_w),
                                                   np.arange(region_orig[1], region_orig[1] + region_h))
            x_grid_dual, y_grid_dual = np.meshgrid(np.arange(region_dual[0], region_dual[0] + region_w),
                                                   np.arange(region_dual[1], region_dual[1] + region_h))
            # 这里存储的就是两张图像里边信息对应的一系列坐标点,都是按相同的顺序存储的
            results['overlap_region'] = (x_grid_orig.flatten(), y_grid_orig.flatten(),
                                         x_grid_dual.flatten(), y_grid_dual.flatten())

        return results

@TRANSFORMS.register_module()
class DetRegionSampler(RandomCrop):
    """Crop small img from oring image
    1. select a bbox from image
    2. bbox random in new img coord, only make sure hole box is inside.
    3. if box is large than crop image, crop image inside box
    4. Pad extra area

    off_rate: close

    """
    def __init__(self,
                 crop_size: tuple,  # (w, h)
                 crop_type: str = 'absolute',
                 allow_negative_crop: bool = True,
                 recompute_bbox: bool = False,
                 bbox_clip_border: bool = True,
                 enlarge_img: bool = True,
                 off_rate: float = 0.0,
                 pad_val: Union[Union[int, float], dict] = 0,
                 padding_mode: str = 'constant'
                 ) -> None:
        super(DetRegionSampler, self).__init__(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border,
        )
        assert self.crop_size[0] > 0 and self.crop_size[1] > 0
        self.pad_val = pad_val
        self.padding_mode = padding_mode
        self.enlarge_img = enlarge_img
        self.off_rate = off_rate

    def _crop_data(self, results: dict, allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        img = results['img']
        margin_h = img.shape[0] - self.crop_size[1]
        margin_w = img.shape[1] - self.crop_size[0]
        offset_h, offset_w = self._rand_offset((margin_h, margin_w), results, img.shape[:2])  # by ldf
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[1]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[0]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop and pad the image
        img = self._crop_and_pad_img(img, crop_x1, crop_y1, crop_x2, crop_y2)  # by ldf
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                gt_masks = results['gt_masks'][valid_inds.nonzero()[0]]
                # translate the masks first, then crop
                gt_masks = gt_masks.translate((gt_masks.height, max(crop_x2, gt_masks.width)-min(crop_x1, 0)), -offset_w, direction="horizontal")  # translate in the horizontal
                gt_masks = gt_masks.translate((max(crop_y2, gt_masks.height)-min(crop_y1, 0), gt_masks.width), -offset_h, direction="vertical")  # translate in the vertical
                gt_masks = gt_masks.crop(np.asarray([0, 0, img_shape[1], img_shape[0]]))
                results['gt_masks'] = gt_masks

                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            raise NotImplementedError("Segmentation map is not supported yet")


        return results

    def _crop_and_pad_img(self, img, x1, y1, x2, y2):
        """

        Args:
            img:
            crop_coords: x1, y1, x2, y2,

        Returns:

        """
        # TODO: more pad mode
        h, w = img.shape[:2]
        pad_size = [self.crop_size[1], self.crop_size[0]]
        if len(img.shape) == 3:
            pad_size.append(img.shape[2])
        if self.pad_val is not None:
            if isinstance(self.pad_val, int):
                pad_img = np.ones(pad_size, dtype=np.uint8) * self.pad_val
            elif isinstance(self.pad_val, dict):
                pad_img = np.ones(pad_size, dtype=np.uint8) * self.pad_val["img"]

            pad_img[max(0-y1, 0): min(y2-y1, h-y1), max(0-x1, 0): min(x2-x1, w-x1), ...] = \
                img[max(y1, 0): min(y2, h), max(x1, 0): min(x2, w), ...]
        else:
            raise ValueError("Plz input pad val, other pad mode not implementation yet")

        return pad_img

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int], results: dict, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin  # img size - crop size
        bboxes = results['gt_bboxes']  # (x1, y1, x2, y2)
        labels = results['gt_bboxes_labels']
        assert len(bboxes) == len(labels)
        if len(bboxes) > 0:
            bbox = bboxes[np.random.randint(0, len(bboxes))]  # random bbox
            x1, y1, x2, y2 = bbox.tensor[0]
            # ensure box in img
            x1, y1 = int(max(x1, 0)), int(max(y1, 0))
            x2, y2 = int(min(x2, img_size[1]-1)), int(min(y2, img_size[0]-1))
            if np.abs(x1 + self.crop_size[0] - x2) <= 1:
                offset_w = x1
            else:
                if self.enlarge_img:
                    w = int(img_size[1] * (1+self.off_rate))
                    if self.crop_size[0] < w:
                        if self.crop_size[0] > x2-x1:
                            offset_w = np.random.randint(max(x2-self.crop_size[0], 0), min(w-self.crop_size[0], x1)+1)
                        else:
                            offset_w = np.random.randint(x1, x2-self.crop_size[0]+1)
                    elif self.crop_size[0] == w:
                        offset_w = 0
                    else:
                        offset_w = np.random.randint(w-self.crop_size[0], 1)
                else:
                    offset_w = np.random.randint(min(x1, x2-self.crop_size[0]), max(x1, x2-self.crop_size[0])+1)
            if np.abs(y1 + self.crop_size[1] - y2) <= 1:
                offset_h = y1
            else:
                if self.enlarge_img:
                    h = int(img_size[0] * (1+self.off_rate))
                    if self.crop_size[1] < h:
                        if self.crop_size[1] > y2-y1:
                            offset_h = np.random.randint(max(y2-self.crop_size[1], 0), min(h-self.crop_size[1], y1)+1)
                        else:
                            offset_h = np.random.randint(y1, y2 - self.crop_size[1]+1)
                    elif self.crop_size[1] == h:
                        offset_h = 0
                    else:
                        offset_h = np.random.randint(h-self.crop_size[1], 1)
                else:
                    offset_h = np.random.randint(min(y1, y2-self.crop_size[1]), max(y1, y2-self.crop_size[1])+1)
        else:
            offset_h = np.random.randint(min(0, margin_h), max(0, margin_h)+1)
            offset_w = np.random.randint(min(0, margin_w), max(0, margin_w)+1)

        return offset_h, offset_w

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        results = self._crop_data(results, self.allow_negative_crop)
        return results