import cv2
from pathlib import Path
import numpy as np
import os
import re
import glob
from collections import defaultdict
import json
import argparse


def convert_segment_masks_to_yolo_seg(masks_dir, output_dir):
    """
    Converts a dataset of segmentation mask images to the YOLO segmentation format.

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation format.
    The converted masks are saved in the specified output directory.

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        classes (int): Total classes in the dataset i.e for COCO classes=80

    Example:
        ```python
        from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        # for coco dataset, we have 80 classes
        convert_segment_masks_to_yolo_seg('path/to/masks_directory', 'path/to/output/directory', classes=80)
        ```

    Notes:
        The expected directory structure for the masks is:

            - masks
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        After execution, the labels will be organized in the following structure:

            - output_dir
                ├─ mask_yolo_01.txt
                ├─ mask_yolo_02.txt
                ├─ mask_yolo_03.txt
                └─ mask_yolo_04.txt
    """
    import os

    pixel_to_class_mapping = {}
    class_list = []
    for mask_filename in os.listdir(masks_dir):
        if mask_filename.endswith(".png"):
            mask_path = os.path.join(masks_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
            img_height, img_width = mask.shape  # Get image dimensions
            print(f"Processing {mask_path} imgsz = {img_height} x {img_width}")

            unique_values = np.unique(mask)  # Get unique pixel values representing different classes
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue
                else:
                    if value in pixel_to_class_mapping.keys():
                        pass
                    else:
                        if len(class_list) != 0:
                            pixel_to_class_mapping[value] = class_list[-1] + 1
                        else:
                            pixel_to_class_mapping[value] = 0
                        class_list.append(pixel_to_class_mapping[value])

            for value in unique_values:
                if value == 0:
                    continue  # Skip background
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    print(f"Unknown class for pixel value {value} in file {mask_filename}, skipping.")
                    continue

                # Create a binary mask for the current class and find contours
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # Find contours

                for contour in contours:
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        yolo_format = [class_index]
                        for point in contour:
                            # Normalize the coordinates
                            yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # Save Ultralytics YOLO format data to file
            output_path = os.path.join(output_dir, os.path.splitext(mask_filename)[0] + ".txt")
            with open(output_path, "w") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            print(f"Processed and stored at {output_path} imgsz = {img_height} x {img_width}")
    print(f"No. of classes: {len(class_list)}")
    print(f"Pixel Values: {list(pixel_to_class_mapping.keys())}")
    print(f"Class Names: {class_list}")

def main():
    parser = argparse.ArgumentParser(description="将数据集转换为YOLO格式")
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="数据集根目录路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--objs_num", type=int, required=True, help="物体个数")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，默认0.8")
    
    args = parser.parse_args()
    
    masks_dir = f"{args.dataset_dir}/{args.dataset_name}/mask_visib"
    output_dir = f"{args.dataset_dir}/{args.dataset_name}/labels"

    with open(f"{args.dataset_dir}/{args.dataset_name}/scene_gt.json", 'r') as f:
        scene_gt = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # 自动从 scene_gt.json 中提取所有出现的 obj_id，并映射到从 0 开始的连续索引
    all_obj_ids = set()
    for val in scene_gt.values():
        for item in val:
            all_obj_ids.add(str(item['obj_id']))
    
    # 排序以保证映射的确定性
    sorted_obj_ids = sorted(list(all_obj_ids), key=int)
    real_obj_id_mapping = {obj_id: str(i) for i, obj_id in enumerate(sorted_obj_ids)}
    
    print(f"发现物体 ID: {sorted_obj_ids}")
    print(f"映射关系: {real_obj_id_mapping}")

    suffix_to_obj_id = {}
    for key, val in scene_gt.items():
        for item in val:
            obj_id = item['obj_id']
            if key not in suffix_to_obj_id.keys():
                suffix_to_obj_id[key] = []
            suffix_to_obj_id[key].append(real_obj_id_mapping[str(obj_id)])

    convert_segment_masks_to_yolo_seg(masks_dir=masks_dir, output_dir=output_dir)


    # 合并一张图像中的各个物体标注
    directory = output_dir

    # 获取目录中所有的txt文件
    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    # 用于存储文件分组的字典
    file_groups = defaultdict(list)

    # 使用正则表达式提取前缀和后缀
    pattern = re.compile(r'(\d+)_(\d+)\.txt')

    # 首先根据文件名的前缀和后缀对文件的class_index进行修改
    print("修改文件中的类别索引:")
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        match = pattern.match(filename)
        
        if match:
            prefix = match.group(1)  # 前缀
            suffix = match.group(2)  # 后缀
            
            try:
                # 根据前缀和后缀获取正确的class_index
                correct_class_index = suffix_to_obj_id[str(int(prefix))][int(suffix)]
                
                # 读取文件内容
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # 修改每行的class_index
                modified_lines = []
                for line in lines:
                    if line.strip():  # 跳过空行
                        parts = line.strip().split()
                        if parts:
                            parts[0] = correct_class_index  # 替换class_index
                            modified_lines.append(" ".join(parts))
                
                # 写回文件
                with open(file_path, 'w') as f:
                    for i, line in enumerate(modified_lines):
                        f.write(line)
                        if i < len(modified_lines) - 1:
                            f.write("\n")
                
                print(f"  - {filename}: 类别索引已更新为 {correct_class_index}")
                
            except (KeyError, IndexError) as e:
                print(f"  - 警告: 无法为 {filename} 更新类别索引: {e}")

            # 对文件进行分组（用于后续合并）
            file_groups[prefix].append(file_path)
        
    # 合并文件并删除原始文件
    print("\n开始合并文件:")
    for prefix, file_paths in file_groups.items():
        # 创建合并后的文件名
        merged_file = os.path.join(directory, f"{prefix}.txt")
        
        print(f"将以下文件合并到 {merged_file}:")
        for path in file_paths:
            print(f"  - {os.path.basename(path)}")
        
        # 合并文件内容
        all_lines = []
        for file_path in file_paths:
            with open(file_path, 'r') as infile:
                # 读取所有行并去除尾部空白字符
                file_lines = [line.rstrip() for line in infile.readlines()]
                # 过滤掉空行
                file_lines = [line for line in file_lines if line]
                all_lines.extend(file_lines)
        
        # 写入合并后的文件
        with open(merged_file, 'w') as outfile:
            for i, line in enumerate(all_lines):
                outfile.write(line)
                # 如果不是最后一行，添加换行符
                if i < len(all_lines) - 1:
                    outfile.write("\n")
        
        # 删除原始文件
        for file_path in file_paths:
            os.remove(file_path)
            
    print("完成！所有文件已合并并删除原始文件。")

if __name__ == "__main__":
    main()

