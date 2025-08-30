import json
import os
import shutil
from pathlib import Path
import argparse
SPLIT=".jpg"
def convert_coco_to_visaion(coco_json_path, image_dir, output_dir):
    """将COCO格式转换为Visaion格式"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取COCO标注文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 创建类别ID到名称的映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 按图片ID组织标注
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        
        # 转换边界框格式 [x,y,width,height] -> [x,y,width,height]
        bbox = ann['bbox']
        image_annotations[image_id].append({
            'boundingBox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            'labelName': categories[ann['category_id']]
        })
    
    # 处理每张图片
    for img in coco_data['images']:
        image_id = img['id']
        image_file = img['file_name']
        image_path = os.path.join(image_dir, image_file)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # 创建标注数据
        detections = image_annotations.get(image_id, [])
        if len(detections) == 0:
            continue
        annotation_data = {
            'detections': detections
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
    parser.add_argument('--coco_json_path', type=str, default='/root/data/visaion/visaionlib/.backup/instances_val2017.json')
    parser.add_argument('--image_dir', type=str, default='/root/data/visaion/visaionlib/.backup/val2017')
    parser.add_argument('--output_dir', type=str, default='/root/data/visaion/visaionlib/.backup/val2017_convert')
    args = parser.parse_args()
    
    # 执行转换
    convert_coco_to_visaion(args.coco_json_path, args.image_dir, args.output_dir)
