import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pycocotools.coco import COCO
from pycocotools import mask as mask_util

def create_colormap(num_classes):
    """创建随机颜色映射"""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors

def visualize_instances(image_path, ann_ids, coco, output_path, colors):
    """可视化单个图像的实例"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 获取该图像的所有标注
    anns = coco.loadAnns(ann_ids)
    
    # 绘制每个实例
    for ann in anns:
        category_id = ann['category_id']
        color = colors[category_id].tolist()
        
        # 绘制分割掩码
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                # 多边形格式
                for seg in ann['segmentation']:
                    points = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    # 创建半透明填充
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [points], color)
                    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                    # 绘制轮廓
                    cv2.polylines(img, [points], True, color, 2)
            elif isinstance(ann['segmentation'], dict):
                # RLE格式
                mask = mask_util.decode(ann['segmentation'])
                # 创建半透明填充
                overlay = img.copy()
                overlay[mask > 0] = color
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
                # 绘制轮廓
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, color, 2)
        
        # 绘制边界框
        if 'bbox' in ann:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # 添加类别标签
            cat_info = coco.loadCats([category_id])[0]
            label = f"{cat_info['name']}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

def main():
    parser = argparse.ArgumentParser(description='可视化COCO格式的标注文件')
    parser.add_argument('--json_path', type=str, default='ins.json', help='COCO格式的标注文件路径')
    parser.add_argument('--output_dir', type=str, default='work_dirs/show_coco_ins', help='输出目录')
    args = parser.parse_args()

    # 加载COCO数据集
    coco = COCO(args.json_path)
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建颜色映射
    num_classes = len(coco.getCatIds())
    colors = create_colormap(num_classes)
    
    # 处理每张图像
    for img_id in tqdm(coco.getImgIds()):
        img_info = coco.loadImgs([img_id])[0]
        image_path = img_info['file_name']
        
        # 获取该图像的所有标注ID
        ann_ids = coco.getAnnIds(imgIds=img_id)
        
        if ann_ids:
            output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}")
            visualize_instances(image_path, ann_ids, coco, output_path, colors)

if __name__ == '__main__':
    main()
