import os
# Windows 环境通常不需要设置 GPU 相关的环境变量，驱动会自动识别 CUDA

from ultralytics import YOLO

# 加载模型 (请根据您的 Windows 实际路径修改)
# 例如: model = YOLO(r"D:\project\ultralytics\yolo26n-seg.pt")
model = YOLO("yolo26n-seg.pt") 

# 在 Windows RTX 4060 Ti 上的激进训练配置
results = model.train(
    data="ultralytics/cfg/datasets/knob.yaml", # 请确保 knob.yaml 中的 path 也是 Windows 格式
    epochs=300,        # 增加训练轮数以获得更好效果
    imgsz=640,         # 恢复标准 640 分辨率，甚至可以尝试 800
    batch=32,          # RTX 4060 Ti 性能强劲，可以从 32 开始尝试，甚至 64
    workers=8,         # 增加读取线程，取决于您的 CPU 核心数
    device=0,          # 使用第一块显卡
    amp=True,          # NVIDIA 显卡必须开启 AMP (混合精度)，速度显著提升且显存减半
    val=True,          # 重新开启验证，追踪模型精度变化
    patience=50        # 如果 50 轮指标不提升则提前停止
)
