"""
验证标签可视化脚本
在图像上用不同颜色绘制各类别的分割轮廓，方便人工检查标注质量。

用法:
    python data_prep/verify_labels.py --dataset_dir data/yolo --dataset_name knob_switch --num 10
"""

import os
import cv2
import numpy as np
import argparse
import random

# 类别颜色映射 (BGR格式)
CLASS_COLORS = {
    0: (0, 255, 0),    # knob  - 绿色
    1: (0, 0, 255),    # switch - 红色
    2: (255, 0, 0),    # 蓝色（备用）
    3: (0, 255, 255),  # 黄色（备用）
    4: (255, 0, 255),  # 紫色（备用）
}

CLASS_NAMES = {
    0: "knob",
    1: "switch",
}


def draw_labels_on_image(img, label_path):
    """在图像上绘制 YOLO 分割标签的轮廓"""
    h, w = img.shape[:2]
    overlay = img.copy()

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # 将归一化坐标转为像素坐标
        points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i + 1] * h)
            points.append([px, py])
        points = np.array(points, dtype=np.int32)

        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        # 绘制半透明填充
        cv2.fillPoly(overlay, [points], color)

        # 绘制轮廓线
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

        # 标注类别名称
        cx, cy = points.mean(axis=0).astype(int)
        cv2.putText(img, name, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 混合半透明填充
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img


def main():
    parser = argparse.ArgumentParser(description="验证 YOLO 分割标签")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", help="train 或 val")
    parser.add_argument("--num", type=int, default=10, help="可视化图像数量")
    parser.add_argument("--shuffle", action="store_true", default=True, help="随机选取")
    args = parser.parse_args()

    img_dir = os.path.join(args.dataset_dir, args.dataset_name, "images", args.split)
    label_dir = os.path.join(args.dataset_dir, args.dataset_name, "labels", args.split)
    out_dir = os.path.join(args.dataset_dir, args.dataset_name, "verify_vis")
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if args.shuffle:
        random.shuffle(img_files)

    count = 0
    for img_name in img_files:
        if count >= args.num:
            break

        stem = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, stem + ".txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is None:
            continue

        vis = draw_labels_on_image(img, label_path)

        # 添加图例
        legend_y = 30
        for cid, cname in CLASS_NAMES.items():
            color = CLASS_COLORS[cid]
            cv2.rectangle(vis, (10, legend_y - 15), (30, legend_y + 5), color, -1)
            cv2.putText(vis, cname, (35, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            legend_y += 30

        save_path = os.path.join(out_dir, f"verify_{stem}.jpg")
        cv2.imwrite(save_path, vis)
        print(f"[{count + 1}/{args.num}] 已保存: {save_path}")
        count += 1

    print(f"\n完成！共生成 {count} 张验证图，保存在: {out_dir}")


if __name__ == "__main__":
    main()
