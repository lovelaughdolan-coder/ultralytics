from ultralytics import YOLO
import os
import torch

# Fix for CUSOLVER_STATUS_INTERNAL_ERROR in PyTorch 2.10+ with CUDA 12.8
try:
    torch.backends.cuda.preferred_linalg_library('magma')
except Exception:
    pass


def train_animal_pose_v3():
    # 权重路径
    model_path = '/home/hxy/ultralytics/model/yolo26s-pose.pt'
    # 数据集 YAML 路径
    data_path = '/home/hxy/ultralytics/data/roboflow/Animal Pose.v3i.yolov8/data.yaml'
    
    # 初始化模型
    model = YOLO(model_path)
    
    # 开始训练
    # 针对 RTX 5060 Ti (16GB) 的优化配置
    results = model.train(
        data=data_path,
        epochs=300,
        imgsz=640,
        batch=64,      # 显存充足，使用大 batch size 提高稳定性
        workers=8,     # 高速数据搬运
        device=0,      # 使用 GPU 0
        name='animal_pose_yolo26s_v3',
        project='runs/pose',
        exist_ok=True,
        pretrained=True,
        # 针对姿态估计的常用调优参数
        patience=50,   # 早停策略
        save=True,
        close_mosaic=10 # 最后 10 个 epoch 关闭 Mosaic 增强以提高精度
    )

if __name__ == '__main__':
    train_animal_pose_v3()
