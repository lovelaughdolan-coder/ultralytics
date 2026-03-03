#!/usr/bin/env python3
"""
将彩色数据集转换为灰度数据集
用于提高模型对不同颜色旋钮的泛化能力
"""

import cv2
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_dataset_to_grayscale(src_dir: str, dst_dir: str):
    """
    将YOLO格式的数据集转换为灰度图
    
    Args:
        src_dir: 源数据集目录 (包含 images/ 和 labels/ 子目录)
        dst_dir: 目标数据集目录
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # 检查源目录
    if not src_path.exists():
        raise FileNotFoundError(f"源目录不存在: {src_dir}")
    
    # 创建目标目录结构
    for subdir in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (dst_path / subdir).mkdir(parents=True, exist_ok=True)
    
    # 处理 train 和 val 两个分割
    for split in ['train', 'val']:
        src_images_dir = src_path / 'images' / split
        dst_images_dir = dst_path / 'images' / split
        src_labels_dir = src_path / 'labels' / split
        dst_labels_dir = dst_path / 'labels' / split
        
        if not src_images_dir.exists():
            print(f"跳过不存在的目录: {src_images_dir}")
            continue
        
        # 获取所有图像文件
        image_files = list(src_images_dir.glob('*.[jJ][pP][gG]')) + \
                      list(src_images_dir.glob('*.[pP][nN][gG]'))
        
        print(f"\n处理 {split} 集: {len(image_files)} 张图像")
        
        for img_file in tqdm(image_files, desc=f"转换 {split}"):
            # 读取并转换为灰度图
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"警告: 无法读取图像 {img_file}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 转回3通道 (YOLO需要3通道输入)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # 保存灰度图
            dst_img_path = dst_images_dir / img_file.name
            cv2.imwrite(str(dst_img_path), gray_3ch)
            
            # 复制对应的标签文件
            label_name = img_file.stem + '.txt'
            src_label = src_labels_dir / label_name
            dst_label = dst_labels_dir / label_name
            
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: 标签文件不存在 {src_label}")
    
    print(f"\n✅ 转换完成! 灰度数据集保存到: {dst_dir}")


if __name__ == '__main__':
    # 配置路径
    SRC_DATASET = '/home/hxy/ultralytics/data/yolo/knob'
    DST_DATASET = '/home/hxy/ultralytics/data/yolo/knob_gray'
    
    convert_dataset_to_grayscale(SRC_DATASET, DST_DATASET)
