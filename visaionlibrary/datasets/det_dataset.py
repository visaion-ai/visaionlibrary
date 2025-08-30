from typing import List
import os
import os.path as osp
import json
import cv2
import time

from mmengine.registry import DATASETS
from mmyolo.datasets import YOLOv5CocoDataset
from mmengine.logging import print_log
from mmengine import load

from visaionlibrary.datasets.utils import SampleStatus

@DATASETS.register_module()
class VisaionDetDataset(YOLOv5CocoDataset):
    def load_data_list(self) -> List[dict]:
        """
        将visaion格式的数据集转换为coco格式
        """
        json_file = osp.join(f"visaion_det_{time.time()}.json")

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
        self.ann_file = json_file
        print_log(f"convert visaion format dataset to coco format ...", "current")
        if self.data_root[0].endswith("json"):
            self.convert_visaion_json_to_coco_det(self.ann_file)
        else:
            self.convert_visaion_folder_to_coco_det(self.ann_file)

        # 调用父类的load_data_list方法
        data_list = super().load_data_list()

        # 判断数据集中每个数据是前景还是背景
        print_log(f"collect samples type ...", "current")
        self.samples_type = []
        for data in data_list:
            if len(data['instances']) == 0:
                self.samples_type.append(SampleStatus.BACKGROUND)
            else:
                self.samples_type.append(SampleStatus.FOREGROUND)
        
        os.remove(json_file)
        return data_list

    
    def convert_visaion_folder_to_coco_det(self, output_file: str):
        """
        将visaion后端格式转换为coco格式
        Args:
            data_roots: 数据集路径
            output_file: 转换成coco数据集的json文件
        """
        coco_data = {
            "info": {
                "description": "Converted from visaion format",
                "version": "1.0",
                "year": 2025,
                "contributor": "visaion2coco",
                "date_created": "2025/05/11"
            },
            "licenses": [{
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 创建类别映射
        category_name_to_id = {}
        category_id = 0  # 从0开始
        annotation_id = 0  # 从0开始

        for data_root in self.data_root:
            for file_name in os.listdir(data_root):
                if not file_name.endswith(('.png', '.bmp', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(data_root, file_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                height, width = img.shape[:2]
                
                # 添加图片信息
                image_id = len(coco_data['images'])  # 从0开始
                coco_data['images'].append({
                    "id": image_id,
                    "file_name": os.path.join(data_root, file_name),
                    "width": width,
                    "height": height,
                    "date_captured": "2024-05-03"
                })
                
                # 检查是否有对应的json文件
                json_file = file_name.rsplit('.', 1)[0] + '.json'
                json_path = os.path.join(data_root, json_file)
                if not os.path.exists(json_path):
                    json_file = file_name.rsplit('.', 1)[0] + '.mix.json'
                    json_path = os.path.join(data_root, json_file)
                
                if os.path.exists(json_path):
                    # 读取json文件获取标注信息
                    with open(json_path, 'r') as f:
                        ann_data = json.load(f)
                    
                    # 从json中获取类别信息
                    for detection in ann_data.get('detections', []):
                        label_name = detection['labelName']
                        if label_name not in category_name_to_id:
                            category_name_to_id[label_name] = category_id
                            coco_data['categories'].append({
                                "id": category_id,
                                "name": label_name,
                                "supercategory": "none"
                            })
                            category_id += 1
                    
                    # 添加标注信息
                    for detection in ann_data.get('detections', []):
                        bbox = detection['boundingBox']  # [x, y, w, h]
                        label_name = detection['labelName']
                        
                        if label_name in category_name_to_id:
                            coco_data['annotations'].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_name_to_id[label_name],
                                "bbox": bbox,
                                "area": bbox[2] * bbox[3],
                                "segmentation": [],
                                "iscrowd": 0
                            })
                            annotation_id += 1

        # 保存为coco格式的json文件
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

    def convert_visaion_json_to_coco_det(self, output_file: str):
        """
        将json文件转换为coco格式
        Args:
            input_file: 输入的json文件路径
            output_file: 输出的coco格式json文件路径
        
        self.data_root: List[str]  数据集路径, eg. ['/visaion/projects/1/datasets/5617.json', ...]
            1116.json内容如下:
            {
                'meta_info': {
                    'url_prefix': 'projects/1/samples'
                },
                'dataset': [
                    {
                        'height': 429, 
                        'width': 473, 
                        'sample_id': 2882, 
                        'mem_size': 1623336, 
                        'mix_annotation_path': '0000001_CCD31-0_20220604172015_8bb121aa92.json',
                        'data': [
                            {
                                'img_path': '0000001_CCD31-0_20220604172015_a1076a68f4.bmp',
                            }
                        ]
                    },
                    ...,
                ]
            }
        
        """
        VISAION_DIR = os.environ['VISAION_DIR']
        assert VISAION_DIR is not None, "VISAION_DIR is not set"

        coco_data = {
            "info": {
                "description": "Converted from visaion format",
                "version": "1.0",
                "year": 2025,
                "contributor": "visaion2coco",
                "date_created": "2025/05/11"
            },
            "licenses": [{
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 创建类别映射
        category_name_to_id = {}
        category_id = 0  # 从0开始
        annotation_id = 0  # 从0开始

        # 从数据集中获取类别信息, 原来是转coco获取的, 那样会有bug
        for class_name in self.metainfo['classes']:
            if class_name not in category_name_to_id:
                category_name_to_id[class_name] = category_id
                coco_data['categories'].append({
                    "id": category_id,
                    "name": class_name,
                    "supercategory": "none"
                })
                category_id += 1

        for dataset_json_path in self.data_root:
            # 读取数据集内容, 即一个json文件中的内容
            dataset_json_dict = load(dataset_json_path)
            all_sample = dataset_json_dict['dataset']
            url_prefix = dataset_json_dict['meta_info']['url_prefix']

            for sample in all_sample:
                # 添加图片信息
                image_id = len(coco_data['images'])
                img_path = os.path.join(VISAION_DIR, url_prefix, sample['data'][0]['img_path'])
                
                coco_data['images'].append({
                    "id": image_id,
                    "file_name": img_path,
                    "width": sample['width'],
                    "height": sample['height'],
                    "date_captured": "2024-05-03"
                })

                # 读取标注文件
                # 如果sample['mix_annotation_path'] is None or sample['mix_annotation_path'] == "", 则跳过
                if sample['mix_annotation_path'] is None or sample['mix_annotation_path'] == "":
                    continue
                ann_path = os.path.join(VISAION_DIR, url_prefix, sample['mix_annotation_path'])
                if os.path.exists(ann_path):
                    with open(ann_path, 'r') as f:
                        ann_data = json.load(f)
                    
                    # 添加标注信息
                    for detection in ann_data.get('detections', []):
                        bbox = detection['boundingBox']  # [x, y, w, h]
                        label_name = detection['labelName']
                        
                        if label_name in category_name_to_id:
                            coco_data['annotations'].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_name_to_id[label_name],
                                "bbox": bbox,
                                "area": bbox[2] * bbox[3],
                                "segmentation": [],
                                "iscrowd": 0
                            })
                            annotation_id += 1

        # 保存为coco格式的json文件
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)