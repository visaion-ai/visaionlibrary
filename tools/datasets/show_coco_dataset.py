import json
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse

def visualize_coco_dataset(json_file, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取COCO格式的JSON文件
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 创建类别ID到名称的映射
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 为每个图像创建可视化
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_path = img_info['file_name']
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建图形
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        # 获取该图像的所有标注
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        # 为每个标注绘制边界框
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']
            category_name = category_map[category_id]
            
            # 创建矩形补丁
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            
            # 添加矩形到图像
            ax.add_patch(rect)
            
            # 添加类别标签
            ax.text(
                bbox[0], bbox[1] - 5,
                category_name,
                color='red',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        # 移除坐标轴
        ax.axis('off')
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{Path(img_path).stem}_vis.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"已保存可视化结果: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化COCO数据集')
    parser.add_argument('--json_file', default="tmp0oj4hq0o.json", type=str, help='COCO数据集的JSON文件路径')
    parser.add_argument('--output_dir', default="work_dirs/show_coco_data", type=str, help='输出目录')
    args = parser.parse_args()

    json_file = args.json_file
    output_dir = args.output_dir
    visualize_coco_dataset(json_file, output_dir)
