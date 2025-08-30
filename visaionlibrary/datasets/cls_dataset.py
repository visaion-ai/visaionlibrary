import os
import os.path as osp
import random
import copy
import logging
import numpy as np
import json
import psutil
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional, Sequence, Union

from mmengine.dataset import Compose
from mmengine import load
from mmengine.registry import DATASETS
from mmpretrain.datasets import BaseDataset
from mmengine.logging import print_log

from visaionlibrary.datasets.utils import SampleStatus


@DATASETS.register_module()
class VisaionClsDataset(BaseDataset):
    """
    Custom dataset for classification.
    """
    def __init__(self,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 lazy_init: bool = False,
                 **kwargs):
        self.data_root = data_root
        super().__init__(
            # The base class requires string ann_file but this class doesn't
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            # Force to lazy_init for some modification before loading data.
            lazy_init=True,
            **kwargs)
        
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()
    
    def load_data_list(self) -> List[dict]:
        # self.data_root格式校验
        if isinstance(self.data_root, str) and "::" not in self.data_root:
                self.data_root = [self.data_root]
        elif isinstance(self.data_root, str) and "::" in self.data_root:
            self.data_root = self.data_root.split("::")
        elif isinstance(self.data_root, list):
            pass
        else:
            raise ValueError(f"data_root must be a string of dataset path or a list of dataset path, but got {self.data_root}")
        
        # dataset可以是文件夹和json文件
        # - 文件夹是visaion后端格式
        # - json是visaion前端兼容格式
        if self.data_root[0].endswith("json"):
            data_list = self.load_cls_from_json()
        else:
            data_list = self.load_cls_from_folder()
        
        return data_list
    

    def load_cls_from_folder(self) -> List[dict]:
        """
        提取数据集文件夹中的路径信息

        每个数据集情况data_root1
            |-- image1.bmp
            |-- image1.json
            |-- *.bmp
            |-- *.json
            |-- label.json
        
        每个数据集情况data_root2
            |-- image1.bmp
            |-- image1.json
            |-- *.bmp
            |-- *.json
        
        Returns:
            data_list:
                [
                    {
                        "img_path":image_path,
                        "det_map_path":channel1_mask,
                        "label_map":None,
                    },
                    ...
                ]
        """

        # 读取多个数据集
        data_list = []          # 存储最终的结果
        for dataset_dir in self.data_root:
            # 读取正样本
            img_suffix = ['.bmp', '.jpg', '.jpeg', '.png']
            images_path = [osp.join(dataset_dir,t) for t in os.listdir(dataset_dir) if Path(t).suffix in img_suffix]
            for img_p in images_path:
                gt_p = img_p.replace(Path(img_p).suffix, '.json')
                if not osp.exists(gt_p):
                    gt_name = 'background'
                    gt_label = self.metainfo['classes'].index(gt_name)
                else:
                    gt_name = load(gt_p)['classfication']['labelName']
                    if gt_name not in self.metainfo['classes']:
                        raise ValueError(f"gt_name not in metainfo['classes']: {gt_name}")
                    gt_label = self.metainfo['classes'].index(gt_name)
                
                # one_record存放一条记录
                one_record = dict()
                one_record["dataset_metainfo"] = self._metainfo
                one_record["img_path"] = img_p
                one_record["gt_label"] = gt_label
                
                data_list.append(one_record)
        
        return data_list
    
    def load_cls_from_json(self) -> List[dict]:
        """
        提取数据集json文件中的路径信息
        """
        VISAION_DIR = os.environ['VISAION_DIR']
        assert VISAION_DIR is not None, "VISAION_DIR is not set"

        data_list = []
        for dataset_json_path in self.data_root:
            # 读取数据集内容, 即一个json文件中的内容
            dataset_json_dict = load(dataset_json_path)
            all_sample = dataset_json_dict['dataset']
            url_prefix = dataset_json_dict['meta_info']['url_prefix']

            for sample in all_sample:
                one_record = dict()

                # 读取图片
                one_record["dataset_metainfo"] = self._metainfo
                one_record["img_path"] = osp.join(VISAION_DIR, url_prefix, sample['data'][0]['img_path'])
                

                # 读取标注文件
                ann_path = os.path.join(VISAION_DIR, url_prefix, sample['mix_annotation_path'])
                if os.path.exists(ann_path) and sample['mix_annotation_path'] != '':
                    ann_data = load(ann_path)
                    gt_name = ann_data['classfication']['labelName']
                    if gt_name not in self.metainfo['classes']:
                        raise ValueError(f"gt_name not in metainfo['classes']: {gt_name}")
                    one_record["gt_label"] = self.metainfo['classes'].index(gt_name)
                else:
                    gt_name = 'background'
                    one_record["gt_label"] = self.metainfo['classes'].index(gt_name)
                
                data_list.append(one_record)
        return data_list
