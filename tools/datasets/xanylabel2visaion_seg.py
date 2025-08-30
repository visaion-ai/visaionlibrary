import json
import os
import shutil
import numpy as np
from pathlib import Path
import argparse
from pycocotools import mask as mask_util
from visaionlibrary.datasets.utils.str2mask import encode_mask_to_str

SPLIT = ".jpg"

def polygon_to_mask(polygon, height, width):
    """将polygon转换为二进制mask"""
    # 将polygon点列表展平成一维数组
    polygon_flat = [coord for point in polygon for coord in point]
    # 将polygon转换为COCO RLE格式
    rles = mask_util.frPyObjects([polygon_flat], height, width)
    rle = mask_util.merge(rles)
    mask = mask_util.decode(rle)
    return mask

def get_bbox_from_polygon(polygon):
    """从polygon点获取边界框"""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    width = int(max(x_coords) - x_min)
    height = int(max(y_coords) - y_min)
    return [x_min, y_min, width, height]

def convert_xanylabel_to_visaion(input_dir, output_dir):
    """将xanylabel格式转换为Visaion格式"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录中的所有json文件
    for json_file in Path(input_dir).glob("*.json"):
        # 读取标注文件
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 获取图片信息
        image_height = data['imageHeight']
        image_width = data['imageWidth']
        image_path = os.path.join(os.path.dirname(json_file), data['imagePath'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # 处理每个标注
        instances = []
        for shape in data['shapes']:
            if shape['shape_type'] != 'polygon':
                continue
            
            # 获取标签名称
            label_name = shape['label']
            
            # 获取polygon点
            polygon = shape['points']
            
            # 获取边界框
            bbox = get_bbox_from_polygon(polygon)
            
            # 生成mask
            mask = polygon_to_mask(polygon, image_height, image_width)
            
            # 使用边界框作为range_box
            range_box = bbox
            
            # 裁剪并编码mask
            mask_cropped = mask[range_box[1]:(range_box[1]+range_box[3]), 
                              range_box[0]:(range_box[0]+range_box[2])]
            mask_str = encode_mask_to_str(mask_cropped)
            
            # 添加到实例列表
            instances.append({
                'rangeBox': range_box,
                'segmentationMap': mask_str,
                'labelName': label_name
            })
        
        # 创建标注数据
        annotation_data = {
            'segmentations': instances
        }
        
        # 保存标注文件
        output_json = os.path.join(output_dir, f"{os.path.basename(image_path).split(SPLIT)[0]}.json")
        with open(output_json, 'w') as f:
            json.dump(annotation_data, f)
        
        # 复制并重命名图片文件
        image_output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split(SPLIT)[0]}.jpg")
        shutil.copy2(image_path, image_output_path)
        
        print(f"Processed: {data['imagePath']} -> {os.path.basename(image_path).split(SPLIT)[0]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/data/projects/visaion/visaionlib/data/yaoluchuan/test')
    parser.add_argument('--output_dir', type=str, default='/data/projects/visaion/visaionlib/data/yaoluchuan/test_visaion')
    args = parser.parse_args()
    
    # 执行转换
    convert_xanylabel_to_visaion(args.input_dir, args.output_dir)
