import os
import shutil
from pathlib import Path
import glob
import re

def get_max_index(directory, extension="*.jpg"):
    """
    给定一个目录，查找图片名称中最大的数字ID。
    假设文件名为类似 000001.jpg 或者 knob_big_000001.jpg
    """
    max_idx = -1
    for p in Path(directory).rglob(extension):
        # 使用正则提取出文件名中的所有连续数字，取最后面的那一组作为序号
        numbers = re.findall(r'\d+', p.name)
        if numbers:
            idx = int(numbers[-1])
            if idx > max_idx:
                max_idx = idx
    return max_idx

def merge_datasets(src_dirs, out_dir):
    """
    合并两或多个数据集（如 YOLO 格式）。
    根据上一个数据集输出的最大序号自动递增。
    如果原始数据没有分 train/val 文件夹，则默认归入 train。
    """
    out_dir = Path(out_dir)
    images_out = out_dir / 'images'
    labels_out = out_dir / 'labels'

    # 创建基础输出目录
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0

    # 获取当前输出目录中已经存在的最大序号，以便从那里开始（如果是空的话是从0开始）
    current_global_index = get_max_index(images_out)
    if current_global_index < 0:
        current_global_index = -1  # 这样下一个就是 0

    for src in src_dirs:
        src = Path(src)
        dataset_name = src.name
        print(f"[{dataset_name}] 开始处理... 当前起始序号应大于: {current_global_index}")

        if not (src / 'images').exists():
            print(f"警告：找不到 {src}/images 目录，跳过此数据集。")
            continue

        # 收集所有需要复制的图像和它们对应的 label
        image_extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
        img_paths = []
        for ext in image_extensions:
            # 只找文件，不找目录
            img_paths.extend([p for p in (src / 'images').rglob(ext) if p.is_file()])

        # 按原名排序一下，保证转换成新序号时的顺序性
        img_paths.sort(key=lambda x: x.name)

        dataset_copied_images = 0

        for img_path in img_paths:
            # 分配新的流水号
            current_global_index += 1
            new_id_str = f"{current_global_index:06d}"

            # 决定输出目标路径。如果原图直接在 images 下（未分 train/val），则默认分入 train 文件夹
            try:
                rel_path = img_path.relative_to(src / 'images')
            except ValueError:
                rel_path = Path(img_path.name)
                
            target_subfolder = rel_path.parent
            if str(target_subfolder) == '.':
                target_subfolder = Path('train')

            # 后缀保持不变
            suffix = img_path.suffix
            new_img_name = f"{new_id_str}{suffix}"
            out_img = images_out / target_subfolder / new_img_name
            out_img.parent.mkdir(parents=True, exist_ok=True)

            # 复制图片
            shutil.copy2(img_path, out_img)
            copied_images += 1
            dataset_copied_images += 1

            # 寻找对应的标签文件
            # 注意原始的标签也是相对于 labels 目录存放的，且结构通常和 images 匹配
            # 为了适配不同层级，如果 img 在 images/train/xxx.jpg，我们需要找 labels/train/xxx.txt
            lbl_name = img_path.stem + '.txt'
            lbl_path = src / 'labels' / rel_path.parent / lbl_name

            # 如果按照严格路径找不到，就在整个 dataset_name/labels 目录下尝试找一下
            if not lbl_path.exists():
                lbl_path = src / 'labels' / lbl_name
            
            if lbl_path.exists() and lbl_path.is_file():
                # 写入到新的目标位置，名字改成新分配的序号
                out_lbl = labels_out / target_subfolder / f"{new_id_str}.txt"
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lbl_path, out_lbl)
                copied_labels += 1
            else:
                # 找不到标签可能是正常情况（无目标图）
                pass

        print(f"[{dataset_name}] 处理完毕。处理了 {dataset_copied_images} 张图片，目前最新序号为 {current_global_index}")

        # 如果有 data.yaml，复制第一份遇到的 data.yaml（仅在输出目录还没有 yaml 时）
        yaml_files = list(src.glob('*.yaml'))
        if yaml_files and not (out_dir / 'data.yaml').exists():
            shutil.copy2(yaml_files[0], out_dir / 'data.yaml')
            print(f"[{dataset_name}] 复制了 dataset.yaml/data.yaml。")

        # 尝试复制 class.txt 或 classes.txt
        for cls_file in ['classes.txt', 'class.txt']:
            cls_path = src / cls_file
            if cls_path.exists() and not (out_dir / cls_file).exists():
                shutil.copy2(cls_path, out_dir / cls_file)

    print(f"\n合并完成！共复制图片数：{copied_images}，共复制标签数：{copied_labels}")
    print(f"如果原先有的数据集没有划分 train/val，已默认放入 'train' 文件夹中。")
    print(f"你可以前往 {out_dir} 检查合并后的结果，如有必要，请修改里面的 data.yaml 确保路径正确。")

if __name__ == "__main__":
    # 配置源目录和输出目录
    base_dir = Path("/home/hxy/ultralytics/data/yolo")
    
    source_datasets = [
        base_dir / "knob_switch0214",
        base_dir / "knob_big"
    ]
    
    output_dataset = base_dir / "knob_merged"
    
    merge_datasets(source_datasets, output_dataset)
