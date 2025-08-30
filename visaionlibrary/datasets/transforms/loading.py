import numpy as np

from mmengine import load
from mmseg.datasets.transforms import LoadAnnotations as MMSeg_LoadAnnotations
from mmengine.registry import TRANSFORMS

from visaionlibrary.datasets.utils import decode_str2mask, SampleStatus


@TRANSFORMS.register_module()
class LoadSegAnnotations(MMSeg_LoadAnnotations):
    def _load_seg_map(self, results: dict) -> None:
        seg_map_path = results['seg_map_path']  # 标注json路径
        height, width = results['img_shape']

        if results['sample_type'] == SampleStatus.RAW:
            gt_semantic_seg = np.ones((height, width), dtype=np.uint8) * 255  # 255 表示为无标注
        elif results['sample_type'] == SampleStatus.BACKGROUND:
            gt_semantic_seg = np.zeros((height, width), dtype=np.uint8)
        elif results['sample_type'] == SampleStatus.FOREGROUND:
            masks = list()  # 用于存储一张图中的多个object的mask
            segmentations = load(seg_map_path)['segmentations']  # 读取json文件, 并获取segmentation字段
            for segmentation in segmentations:
                range_box = segmentation.get('rangeBox')  # 标注子框 x1, y1, w, h
                segmentation_map = segmentation.get('segmentationMap')  # 子框标注图
                label_name = segmentation.get('labelName')  # 获得子框对应的类别
                # for inference class not in metainfo, 推理时, 如果类别不在metainfo中, 则不进行读取, 认为是负样本
                if label_name in results['dataset_metainfo']['classes']:
                    # 获取mask_sub
                    label_index = results['dataset_metainfo']['classes'].index(label_name)  # 通过类别名称获得类别ID
                    range_box_width, range_box_height = range_box[2], range_box[3]  # 获得子框的宽, 高
                    mask_sub = decode_str2mask(segmentation_map, range_box_height, range_box_width)  # 获得子框的mask
                    mask_sub = np.where(mask_sub == 1, label_index, 0)  # 赋值对应labelIndex

                    # 获得mask
                    mask = np.zeros((height, width), dtype=np.uint8)  
                    # check rangebox and mask sub
                    if range_box[1] + range_box[3] >= height:
                        range_box[3] = height - range_box[1]
                    if range_box[0] + range_box[2] >= width:
                        range_box[2] = width - range_box[0]
                    mask_sub = mask_sub[:range_box[3], :range_box[2]]

                    mask[range_box[1]:range_box[1] + range_box[3], range_box[0]:range_box[0] + range_box[2]] = mask_sub
                    masks.append(mask)  # add mask image into mask_channels
            
            # merge masks
            masks = np.stack(masks, axis=0)  # 将多个mask合并

            # mask1  mask2  备注:(1)同一像素点, (2)0表示背景,2表示类别2,3表示类别3
            # 0      3  --> 情况1
            # 3      3  --> 情况2
            # 2      3  --> 情况3 (前端已经过滤掉了)
            gt_semantic_seg = np.max(masks, axis=0)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

