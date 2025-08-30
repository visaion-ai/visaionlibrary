import random
from typing import Optional, Sequence, Tuple, Union
import math
import numpy as np

import mmcv
from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS

Number = Union[int, float]


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rotate_point(x, y, cx, cy, angle_rad):
    x_new = (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad) + cx
    y_new = (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad) + cy
    return int(round(x_new, 0)), int(round(y_new, 0))


def rotate_region(x, y, width, height, cx, cy, angle_deg):
    angle = math.radians(angle_deg)
    # 构造原始矩阵的坐标矩阵
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # 将坐标转换为以旋转中心为原点的坐标系
    X = X - (cx - x)
    Y = Y - (cy - y)

    # 应用旋转矩阵
    X_rotated = X * np.cos(angle) - Y * np.sin(angle)
    Y_rotated = X * np.sin(angle) + Y * np.cos(angle)

    # 将坐标转换回原始坐标系
    X_rotated += cx - x
    Y_rotated += cy - y

    # 将浮点型坐标四舍五入为整数型
    X_rotated = np.round(X_rotated).astype(int).flatten()
    Y_rotated = np.round(Y_rotated).astype(int).flatten()

    # 找到旋转后还在图像范围内的像素点并拉成直线
    condition = np.where(
        (X_rotated >= 0)
        & (X_rotated <= width - 1)
        & (Y_rotated >= 0)
        & (Y_rotated <= height - 1)
    )

    return X_rotated[condition], Y_rotated[condition]


@TRANSFORMS.register_module()
class RandomRescale(BaseTransform):
    """
    合并了resize和ALB的RandomScale的版本

    Required Keys:

    - seg_map_path: Path to the segmentation map file.  TODO:

    Modified Keys:


    Add Keys:
    - gt_seg_map(np.uint8): Ground truth segmentation map.  TODO:

    """

    def __init__(
        self,
        base_scale_factor: float = 1.0,
        scale_range: float = 0.1,
        prob: float = 0.5,
        keep_ratio: bool = True,
        backend: str = "cv2",
        interpolation="bilinear",
        clip_object_border: bool = True,
    ) -> None:
        assert 0 <= prob <= 1
        assert base_scale_factor > scale_range >= 0
        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        self.base_scale_factor = base_scale_factor
        self.scale_range = scale_range
        self.prob = prob

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        if results.get("img", None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results["img"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results["img"].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results["img"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
            #  如果输入图像是灰度图，本来是[h, w, 1]这种size经过mmcv的resize就变成了[h, w]这种size，少了一维后面直接报错, by fufa
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
            results["img"] = img
            results["img_shape"] = img.shape[:2]
            results["scale_factor"] = (w_scale, h_scale)
            results["keep_ratio"] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"].rescale_(results["scale_factor"])
            if self.clip_object_border:
                results["gt_bboxes"].clip_(results["img_shape"])

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is not None:
            if self.keep_ratio:
                results['gt_masks'] = results['gt_masks'].rescale(
                    results['scale'])
            else:
                results['gt_masks'] = results['gt_masks'].resize(
                    results['img_shape'])

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get("gt_seg_map", None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results["gt_seg_map"],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            else:
                gt_seg = mmcv.imresize(
                    results["gt_seg_map"],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            results["gt_seg_map"] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get("gt_keypoints", None) is not None:
            keypoints = results["gt_keypoints"]
            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results["scale_factor"]
            )
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(
                    keypoints[:, :, 0], 0, results["img_shape"][1]
                )
                keypoints[:, :, 1] = np.clip(
                    keypoints[:, :, 1], 0, results["img_shape"][0]
                )
            results["gt_keypoints"] = keypoints

    def transform(self, results: dict) -> dict:
        img_shape = results["img"].shape[:2]
        if random.random() > self.prob:  # 说明直接进行固定比例的缩放
            results["scale"] = _scale_size(img_shape[::-1], self.base_scale_factor)  # type: ignore
        else:  # 需要进行随机比例缩放
            scale_factor = (
                (2 * random.random() * self.scale_range)
                - self.scale_range
                + self.base_scale_factor
            )
            results["scale"] = _scale_size(img_shape[::-1], scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"clip_object_border={self.clip_object_border}), "
        repr_str += f"backend={self.backend}), "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str
