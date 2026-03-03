import os
import shutil
import random
from pathlib import Path
import argparse

def split_dataset(dataset_dir, dataset_name, train_ratio=0.8, shuffle=False):
    """
    将数据集按照指定比例划分为训练集和验证集
    
    Args:
        dataset_dir: 包含images和labels文件夹的数据集根目录
        train_ratio: 训练集比例，默认0.8
        shuffle: 是否随机打乱数据集，默认False
    """
    # 构建路径
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / dataset_name / "images"
    labels_dir = dataset_path / dataset_name / "labels"
    
    # 确保images和labels文件夹存在
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"在 {dataset_dir}/{dataset_name} 下找不到 images 或 labels 文件夹")
    
    # 获取所有图像文件及其完整路径（支持递归查找）
    image_list = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        # 记录 (文件名, 完整路径) 元组
        image_list.extend([(f.name, f) for f in images_dir.glob(ext) if f.is_file()])
        # 如果当前层没有，去下一层找（针对之前运行过的情况，如 train/val 分类目录）
        image_list.extend([(f.name, f) for f in images_dir.glob(f"*/{ext}") if f.is_file()])
    
    # 根据文件名去重，保留第一个找到的路径
    seen_names = set()
    image_files_with_path = []
    for name, path in sorted(image_list, key=lambda x: x[0]):
        if name not in seen_names:
            image_files_with_path.append((name, path))
            seen_names.add(name)
    
    if not image_files_with_path:
        print(f"警告: 在 {images_dir} 中找不到任何图像文件，跳过划分。")
        return

    # 随机打乱
    if shuffle:
        print("正在随机打乱数据集...")
        random.shuffle(image_files_with_path)

    # 计算训练集大小
    train_size = int(len(image_files_with_path) * train_ratio)
    
    # 划分训练集和验证集
    train_data = image_files_with_path[:train_size]
    val_data = image_files_with_path[train_size:]
    
    # 创建目标目录
    train_img_dir = images_dir / "train"
    train_label_dir = labels_dir / "train"
    val_img_dir = images_dir / "val"
    val_label_dir = labels_dir / "val"
    
    # 确保目标目录存在
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # 复制训练集文件
    for img_name, src_img in train_data:
        # 获取不带扩展名的文件名
        file_stem = Path(img_name).stem
        
        # 复制图像文件 (如果已经在目标位置则忽略)
        dst_img = train_img_dir / img_name
        if src_img.resolve() != dst_img.resolve():
            shutil.move(str(src_img), str(dst_img))
        
        # 尝试复制对应的标签文件（如果存在）
        label_file = f"{file_stem}.txt"
        # 标签也可能已经由于之前的失败运行而在 train 目录下
        possible_label_dirs = [labels_dir, labels_dir / "train", labels_dir / "val"]
        for p_dir in possible_label_dirs:
            src_label = p_dir / label_file
            if src_label.exists():
                dst_label = train_label_dir / label_file
                if src_label.resolve() != dst_label.resolve():
                    shutil.move(str(src_label), str(dst_label))
                break
    
    # 复制验证集文件
    for img_name, src_img in val_data:
        # 获取不带扩展名的文件名
        file_stem = Path(img_name).stem
        
        # 复制图像文件
        dst_img = val_img_dir / img_name
        if src_img.resolve() != dst_img.resolve():
            shutil.move(str(src_img), str(dst_img))
        
        # 尝试复制对应的标签文件
        label_file = f"{file_stem}.txt"
        possible_label_dirs = [labels_dir, labels_dir / "train", labels_dir / "val"]
        for p_dir in possible_label_dirs:
            src_label = p_dir / label_file
            if src_label.exists():
                dst_label = val_label_dir / label_file
                if src_label.resolve() != dst_label.resolve():
                    shutil.move(str(src_label), str(dst_label))
                break
    
    print(f"数据集划分完成！")
    print(f"总文件数: {len(image_files_with_path)}")
    print(f"训练集数量: {len(train_data)} ({len(train_data)/len(image_files_with_path):.1%})")
    print(f"验证集数量: {len(val_data)} ({len(val_data)/len(image_files_with_path):.1%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="划分数据集为训练集和验证集")
    parser.add_argument("--dataset_dir", type=str, required=True, help="数据集根目录路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，默认0.8")
    parser.add_argument("--shuffle", action="store_true", help="是否在划分前随机打乱数据集")
    
    args = parser.parse_args()
    
    split_dataset(args.dataset_dir, args.dataset_name, args.train_ratio, args.shuffle)

# python data_split.py --dataset_dir datasets/es6dyolo1