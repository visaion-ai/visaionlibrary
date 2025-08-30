import json
import os
import shutil
import numpy as np
import re
from pathlib import Path
import argparse
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from visaionlibrary.datasets.utils.str2mask import encode_mask_to_str
SPLIT=".jpg"

def convert_coco_to_visaion(coco_json_path, image_dir, output_dir):
    """将COCO实例分割格式转换为Visaion格式"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用pycocotools加载COCO数据
    coco = COCO(coco_json_path)
    
    # 获取所有图片ID
    img_ids = coco.getImgIds()
    
    # 处理每张图片
    for img_id in img_ids:
        # 获取图片信息
        img_info = coco.loadImgs(img_id)[0]
        image_file = img_info['file_name']
        image_path = os.path.join(image_dir, image_file)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # 获取该图片的所有标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 处理每个标注
        instances = []
        for ann in anns:
            # 获取类别名称
            cat_info = coco.loadCats(ann['category_id'])[0]
            label_name = cat_info['name']
            
            # 获取分割mask
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):  # RLE格式
                    # 处理RLE格式
                    rle = ann['segmentation']
                    if isinstance(rle['counts'], list):
                        # 如果是数字数组格式，转换为字符串格式
                        rle['counts'] = mask_util.frPyObjects([rle], rle['size'][0], rle['size'][1])[0]['counts']
                        mask = mask_util.decode(rle)
                    else:
                        # 如果是二进制字符串格式
                        rle['counts'] = rle['counts'].decode('utf-8')
                        mask = mask_util.decode(rle)
                else:  # polygon格式
                    # 将多个polygon转换为RLE格式
                    rle = mask_util.frPyObjects(ann['segmentation'], 
                                              img_info['height'], 
                                              img_info['width'])
                    if isinstance(rle, list):
                        # 合并多个polygon的RLE
                        rle = mask_util.merge(rle)
                    mask = mask_util.decode(rle)
                
                # 获取边界框
                bbox = ann['bbox']
                bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                
                # 计算range_box（稍微扩大一点边界框以确保包含完整mask）
                # x, y, w, h = bbox
                # range_box = [max(0, x-1), max(0, y-1), min(img_info['width'], w+2), min(img_info['height'], h+2)]
                # 如果稍微扩大容易出现bug, 所以不扩大
                range_box = bbox

                
                # 编码mask为字符串
                mask_str = encode_mask_to_str(mask[range_box[1]:(range_box[1]+range_box[3]), range_box[0]:(range_box[0]+range_box[2])])
                
                # 添加到实例列表
                instances.append({
                    'rangeBox': range_box,
                    'segmentationMap': mask_str,
                    'labelName': label_name,
                    # 'boundingBox': bbox
                })
        
        # 创建标注数据
        annotation_data = {
            'segmentations': instances
        }
        
        # 保存标注文件
        json_path = os.path.join(output_dir, f"{os.path.basename(image_path).split(SPLIT)[0]}.json")
        with open(json_path, 'w') as f:
            json.dump(annotation_data, f)
        
        # 复制并重命名图片文件
        image_output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split(SPLIT)[0]}.jpg")
        shutil.copy2(image_path, image_output_path)
        
        print(f"Processed: {image_file} -> {os.path.basename(image_path).split(SPLIT)[0]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_json_path', type=str, default='/root/data/visaion/visaionlib/data/coco2017/annotations/instances_val2017.json')
    parser.add_argument('--image_dir', type=str, default='/root/data/visaion/visaionlib/data/coco2017/val2017')
    parser.add_argument('--output_dir', type=str, default='/root/data/visaion/visaionlib/data/coco_val2017_ins')
    args = parser.parse_args()
    
    # 执行转换
    convert_coco_to_visaion(args.coco_json_path, args.image_dir, args.output_dir)
