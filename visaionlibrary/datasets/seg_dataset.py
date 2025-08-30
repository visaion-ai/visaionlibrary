from typing import List
import tempfile
import os
import os.path as osp
from pathlib import Path

from mmengine.registry import DATASETS
from mmengine.logging import print_log
from mmseg.datasets import BaseSegDataset
from mmengine import load

from visaionlibrary.datasets.utils import SampleStatus

@DATASETS.register_module()
class VisaionSegDataset(BaseSegDataset):
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
            data_list = self.load_seg_from_json()
        else:
            data_list = self.load_seg_from_folder()
        
        # 判断数据集中每个数据是前景还是背景
        print_log(f"collect samples type ...", "current")
        self.samples_type = []
        for data in data_list:
            if data['seg_map_path'] is None:
                self.samples_type.append(SampleStatus.BACKGROUND)
            else:
                self.samples_type.append(SampleStatus.FOREGROUND)

        return data_list
    
    def load_seg_from_folder(self) -> List[dict]:
        """
        提取数据集文件夹中的路径信息

        每个数据集情况
            |-- image1.bmp
            |-- image1.json
            |-- *.bmp
            |-- *.json
            |-- label.json
            |-- background.json
        
        Args:
            None
        Returns:
            data_list: List[dict]   返回数据集中每个数据的dict
                [
                    {
                        "dataset_metainfo":{"classes":[], "color":[]},
                        "img_path": image_path,
                        "seg_map_path": mixed_json_path,
                        "sample_global_id": str,
                        "label_map":None,
                        'reduce_zero_label':True,
                        'seg_fields':[]
                    },
                    ...
                ]
        """
        
        # 读取多个数据集
        data_list = []          # 存储最终的结果
        img_suffix = ['.bmp', '.jpg', '.jpeg', '.png']

        for dataset_dir in self.data_root:
            # 读取背景信息
            background_info_path = osp.join(dataset_dir, 'background.json')
            background_info = load(background_info_path) if osp.exists(background_info_path) else []

            # 读取正样本
            images_path = [osp.join(dataset_dir,t) for t in os.listdir(dataset_dir) if Path(t).suffix in img_suffix]
            for img_p in images_path:
                is_background = True if os.path.basename(img_p) in background_info else False

                mask_p = img_p.replace(Path(img_p).suffix, '.json')
                if not osp.exists(mask_p):
                    mask_p = img_p.replace(Path(img_p).suffix, '.mix.json')
                mask_path = mask_p if osp.exists(mask_p) else None

                # one_record存放一条记录
                one_record = dict()
                one_record["dataset_metainfo"] = self._metainfo
                one_record["img_path"] = img_p
                one_record["seg_map_path"] = mask_path
                one_record['label_map'] = self.label_map
                one_record['reduce_zero_label'] = self.reduce_zero_label
                one_record['seg_fields'] = []
                if mask_path is not None:   # 有json文件，那就是已标注样本
                    one_record['sample_type'] = SampleStatus.FOREGROUND
                elif is_background:         # 没有json但是在background里面，那就是背景样本
                    one_record['sample_type'] = SampleStatus.BACKGROUND
                else:                       # 没有json也没有bg信息，那就是无标注样本
                    one_record['sample_type'] = SampleStatus.RAW
                data_list.append(one_record)

        return data_list
    
    def load_seg_from_json(self) -> List[dict]:
        """
        提取数据集json文件中的路径信息
        """
        VISAION_DIR = os.environ['VISAION_DIR']
        assert VISAION_DIR is not None, "VISAION_DIR is not set"

        # 读取多个数据集
        data_list = []          # 存储最终的结果

        for dataset_json_path in self.data_root:
            # 读取数据集内容, 即一个json文件中的内容
            dataset_json_dict = load(dataset_json_path)
            all_sample = dataset_json_dict['dataset']
            url_prefix = dataset_json_dict['meta_info']['url_prefix']

            for one_sample in all_sample:
                # 获取图片和标注路径
                image_path = osp.join(VISAION_DIR, url_prefix, one_sample['data'][0]['img_path'])
                anno_path = one_sample.get('mix_annotation_path', None)
                if anno_path is not None and anno_path != '':
                    anno_path = osp.join(VISAION_DIR, url_prefix, anno_path)
                anno_path = None if not osp.exists(anno_path) else anno_path

                # one_record存放一条记录
                one_record = dict()
                one_record["dataset_metainfo"] = self._metainfo
                one_record["img_path"] = image_path
                one_record["seg_map_path"] = anno_path
                one_record['label_map'] = self.label_map
                one_record['reduce_zero_label'] = self.reduce_zero_label
                one_record['seg_fields'] = []
                if one_sample.get('status', SampleStatus.RAW.value) == SampleStatus.RAW.value:
                    raise ValueError(f"sample {one_sample['data'][0]['img_path']} is raw")
                one_record['sample_type'] = SampleStatus(one_sample.get('status', SampleStatus.BACKGROUND.value))

                data_list.append(one_record)

        return data_list
