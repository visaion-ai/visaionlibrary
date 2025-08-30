import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from tqdm import tqdm
import re
import argparse

from visaionlibrary.datasets.utils.str2mask import decode_str2mask

def binary_mask_to_polygon(binary_mask):
    """
    将二进制mask转换为轮廓点列表
    
    Args:
        binary_mask: 二进制mask数组
    
    Returns:
        list: 轮廓点列表，每个轮廓是一个点的列表，格式为[x1,y1,x2,y2,...]
    """
    # 找到所有轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 将轮廓转换为点列表
    polygons = []
    for contour in contours:
        # 简化轮廓，减少点数
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 将轮廓点展平为一维数组
        if len(approx) >= 3:  # 只保留至少3个点的轮廓
            polygon = approx.flatten().tolist()
            polygons.append(polygon)
    
    return polygons

def visualize_coco(image_path, coco_data, image_id, output_path=None):
    """
    可视化COCO格式的标注结果
    
    Args:
        image_path: 图像路径
        coco_data: COCO格式的数据
        image_id: 要可视化的图像ID
        output_path: 可视化结果保存路径，如果为None则显示不保存
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建一个用于显示的图像
    vis_image = image.copy()
    
    # 创建一个只包含掩码的图像
    mask_overlay = np.zeros_like(image)
    
    # 为每个类别生成不同的颜色
    colors = np.array([
        [255, 0, 0],      # 红色
        [0, 255, 0],      # 绿色
        [0, 0, 255],      # 蓝色
        [255, 255, 0],    # 黄色
        [0, 255, 255],    # 青色
        [255, 0, 255],    # 洋红色
        [255, 165, 0],    # 橙色
        [128, 0, 128],    # 紫色
        [255, 192, 203],  # 粉色
        [0, 128, 0]       # 深绿色
    ]).astype(np.float64)
    
    # 获取该图像的所有标注
    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
    
    # 绘制每个实例的掩码
    for i, ann in enumerate(annotations):
        # 获取类别ID和名称
        category_id = ann["category_id"]
        category_name = next((cat["name"] for cat in coco_data["categories"] if cat["id"] == category_id), "unknown")
        
        # 获取掩码
        polygons = ann["segmentation"]
        
        # 创建一个全零的mask
        binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 绘制所有轮廓
        for polygon in polygons:
            # 将点列表转换为轮廓格式
            points = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(binary_mask, [points], 1)
        
        # 获取边界框
        x, y, w, h = ann["bbox"]
        
        # 获取当前实例的颜色
        color = colors[i % len(colors)]
        
        # 创建一个彩色掩码
        color_mask = np.zeros_like(image)
        for c in range(3):
            color_mask[:, :, c] = binary_mask * color[c]
        
        # 将彩色掩码添加到mask_overlay
        mask_overlay = np.maximum(mask_overlay, color_mask)
        
        # 绘制边界框
        cv2.rectangle(vis_image, (int(x), int(y)), (int(x+w), int(y+h)), color.tolist(), 2)
        
        # 添加标签
        label = f"{category_name}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(vis_image, (int(x), int(y)-text_size[1]-5), 
                     (int(x)+text_size[0], int(y)), color.tolist(), -1)
        cv2.putText(vis_image, label, (int(x), int(y)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 创建一个只包含掩码的二进制图像
    any_mask = np.any(mask_overlay > 0, axis=2).astype(np.uint8)
    
    # 将掩码应用到图像上
    for c in range(3):
        vis_image[:, :, c] = np.where(
            any_mask > 0,
            mask_overlay[:, :, c] * 0.7 + vis_image[:, :, c] * 0.3,
            vis_image[:, :, c]
        )
    
    # 显示或保存结果
    if output_path:
        # 转回BGR保存
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_image)
        # print(f"可视化结果已保存至: {output_path}")
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.show()

def visaion2coco(visaion_dir, output_json_path, visualize=False):
    """
    将visaion格式的标注转换为COCO格式
    
    Args:
        visaion_dir: visaion格式标注文件目录
        output_json_path: 输出的COCO格式JSON文件路径
        images_dir: 图像目录，如果提供则会将图像信息添加到COCO格式中
        visualize: 是否生成可视化结果
    """
    # 初始化COCO格式数据结构
    coco_data = {
        "info": {
            "description": "Converted from visaion format",
            "version": "1.0",
            "year": 2024,
            "contributor": "visaion2coco",
            "date_created": "2024/05/03"
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
    category_id = 1
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(visaion_dir) if f.endswith('.json')]
    
    # 处理每个JSON文件
    annotation_id = 1
    for json_file in tqdm(json_files, desc="Converting annotations"):
        # 读取visaion格式的JSON文件
        with open(os.path.join(visaion_dir, json_file), 'r', encoding='utf-8') as f:
            visaion_data = json.load(f)
        
        # 获取图像文件名
        image_filename = json_file.replace('.json', '.jpg')
        
        # 读取图像信息
        img = cv2.imread(os.path.join(visaion_dir, image_filename))
        height, width = img.shape[:2]
    
        # 添加图像信息到COCO格式
        image_id = len(coco_data["images"]) + 1
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "height": height,
            "width": width,
            "date_captured": "2024-05-03"
        })
        
        # 处理每个实例
        for instance in visaion_data.get("instances", []):
            # 获取类别名称
            category_name = instance["labelName"]
            
            # 如果类别不存在，添加到类别列表
            if category_name not in category_name_to_id:
                category_name_to_id[category_name] = category_id
                coco_data["categories"].append({
                    "id": category_id,
                    "name": category_name,
                    "supercategory": "none"
                })
                category_id += 1
            
            # 获取rangeBox
            x, y, w, h = instance["rangeBox"]
            
            # 解码mask
            mask_str = instance["segmentationMap"]
            binary_mask = decode_str2mask(mask_str, h, w)
            
            if binary_mask is None:
                continue
            
            # 创建一个全图大小的mask
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = binary_mask
            
            # 将mask转换为轮廓点
            polygons = binary_mask_to_polygon(full_mask)
            
            if not polygons:  # 如果没有找到有效的轮廓，跳过这个实例
                continue
            
            # 计算mask的面积
            area = int(np.sum(full_mask))
            
            # 添加标注信息到COCO格式
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_name_to_id[category_name],
                "segmentation": polygons,
                "area": area,
                "bbox": instance["boundingBox"],
                "iscrowd": 0
            })
            
            annotation_id += 1
    
    # 保存COCO格式的JSON文件
    if not os.path.exists(os.path.dirname(output_json_path)):
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=4)
    
    print(f"转换完成！COCO格式的标注文件已保存至: {output_json_path}")

    # 如果需要可视化，创建可视化输出目录
    if visualize:
        vis_dir = os.path.join(os.path.dirname(output_json_path), "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        print("\n生成可视化结果...")
        for image_info in tqdm(coco_data["images"], desc="Visualizing"):
            image_id = image_info["id"]
            image_filename = image_info["file_name"]
            image_path = os.path.join(visaion_dir, image_filename)
            vis_output_path = os.path.join(vis_dir, f"vis_{image_filename}")
            visualize_coco(image_path, coco_data, image_id, vis_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visaion_dir", type=str, default="/root/data/visaion/visaionlib/data/coco_val2017_ins")
    parser.add_argument("--output_json_path", type=str, default="/root/data/visaion/visaionlib/data/coco_val2017_ins_vis")
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()

    # 转换训练集
    visaion2coco(
        visaion_dir=args.visaion_dir,
        output_json_path=args.output_json_path,
        visualize=args.visualize  # 启用可视化
    )
