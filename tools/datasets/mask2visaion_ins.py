import numpy as np
import cv2
import argparse
import os
import shutil
import json

from visaionlibrary.datasets.utils.str2mask import encode_mask_to_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="/root/data/visaion/visaionlib/data/demo100/living_room")
    parser.add_argument("--dst", type=str, default="/root/data/visaion/visaionlib/data/WasteBin/")
    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.makedirs(args.dst, exist_ok=True)

    all_images = [img_path for img_path in os.listdir(args.src) if img_path.endswith(".jpg")]
    all_masks = [mask_path[:-4]+"_mask.png" for mask_path in all_images]

    for img_p, mask_p in zip(all_images, all_masks):
        img_path = os.path.join(args.src, img_p)
        mask_path = os.path.join(args.src, mask_p)

        # copy image
        shutil.copy(img_path, os.path.join(args.dst, img_p))

        # copy mask
        label = cv2.imread(mask_path)
        # bbox = label[:, :, 0]
        mask = label[:, :, 1]
        # mask2 = label[:, :, 2]

        # 取mask的非0区域的外接矩形
        mask_nonzero = mask > 0
        if mask_nonzero.sum() == 0:
            continue
        bbox = cv2.boundingRect(mask_nonzero.astype(np.uint8))
        
        # 计算range_box（稍微扩大一点边界框以确保包含完整mask）
        # x, y, w, h = bbox
        # range_box = [max(0, x-1), max(0, y-1), w+2, h+2]
        range_box = bbox


        # 编码mask为字符串
        mask_str = encode_mask_to_str(mask_nonzero.astype(np.uint8)[range_box[1]:(range_box[1]+range_box[3]), range_box[0]:(range_box[0]+range_box[2])])
        
        # 保存mask_str到json
        with open(os.path.join(args.dst, img_p.replace(".jpg", ".json")), "w") as f:
            json.dump(
                {
                    "instances": [
                        {
                            "rangeBox": range_box, 
                            "segmentationMap": mask_str, 
                            "labelName": "wastebin", 
                            "boundingBox": bbox
                        }
                    ]
                }, f, indent=4)

