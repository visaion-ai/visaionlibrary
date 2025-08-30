import os
import cv2
import numpy as np
import shutil
import random
import json

def add_single_pixel_noise(image):
    # Create a copy of the image
    noisy_image = image.copy()
    
    # Get random coordinates
    h, w = image.shape[:2]
    x = random.randint(0, w-1)
    y = random.randint(0, h-1)
    
    # Add random noise to single pixel
    if len(image.shape) == 3:  # Color image
        channel = random.randint(0, 2)
        noisy_image[y, x, channel] = random.randint(0, 255)
    else:  # Grayscale image
        noisy_image[y, x] = random.randint(0, 255)
        
    return noisy_image

def main():
    # Define paths
    base_dir = "/data/projects/visaion/visaionlib"
    input_dir = os.path.join(base_dir, "data/iphone")
    output_dir = os.path.join(base_dir, "data/iphone_expand")
    suffix = ".bmp"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jpg files
    jpg_files = [f for f in os.listdir(input_dir) if f.endswith(suffix)]
    
    for jpg_file in jpg_files:
        # Read the image
        img_path = os.path.join(input_dir, jpg_file)
        img = cv2.imread(img_path)
        
        # Get corresponding JSON file
        json_file = jpg_file.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(input_dir, json_file)
        
        if not os.path.exists(json_path):
            print(f"Warning: No JSON file found for {jpg_file}")
            continue
            
        # Generate 100 variations
        for i in range(1000):
            # Create new filenames
            new_base_name = f"{jpg_file.rsplit('.', 1)[0]}_{i+1}"
            new_jpg_name = f"{new_base_name}{suffix}"
            new_json_name = f"{new_base_name}.json"
            
            # Add noise and save new image
            noisy_img = add_single_pixel_noise(img)
            cv2.imwrite(os.path.join(output_dir, new_jpg_name), noisy_img)
            
            # Copy and rename JSON file
            shutil.copy2(json_path, os.path.join(output_dir, new_json_name))
            
        print(f"Processed {jpg_file}: Generated 100 variations")

if __name__ == "__main__":
    main()
