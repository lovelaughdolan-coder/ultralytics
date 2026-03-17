from ultralytics import YOLO

def train_pose_model():
    # 既然您的 RTX 5060 Ti 有 16GB 显存且目前余量极大，建议换用更大、更准确的模型。
    # yolov8n-pose.pt (Nano，最快，准度最低)
    # yolov8s-pose.pt (Small，适合您的显卡，精度提升明显)
    # yolov8m-pose.pt (Medium，精度更高，训练稍慢)
    model = YOLO('/home/hxy/ultralytics/runs/pose/runs/train/animal_pose_v24/weights/last.pt')

    # 开始训练
    results = model.train(
        data='/home/hxy/ultralytics/data/roboflow/Animal Pose.v2i.yolov8/data.yaml',
        resume=True,
        epochs=300,
        imgsz=640,  # 保持 640 即可，因为 Roboflow 数据集默认已经是 640x640
        batch=64,   # 64 或者 128 都可以
        workers=8,  # 增加数据加载线程，缓解 CPU 喂不饱 GPU 的问题
        device=0,  # 使用 GPU 0
        name='animal_pose_v2',
        project='runs/train',
        # 根据显存情况可以调整 batch
    )

if __name__ == '__main__':
    train_pose_model()
