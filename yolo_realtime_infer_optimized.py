#!/usr/bin/env python3
"""
ROS2 + YOLO 实时推理节点
订阅奥比中光 Gemini 330 相机的彩色图像话题，使用 YOLO 模型进行实时检测/分割推理。
在 OpenCV 窗口中显示检测结果和实时 FPS。

用法:
    1. 启动相机: ros2 launch orbbec_camera gemini_330_series.launch.py
    2. 运行推理: python3 yolo_realtime_infer.py [--model yolo26n.pt] [--topic /camera/color/image_raw]
    
按 'q' 退出。
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloRealtimeNode(Node):
    def __init__(self, model_path, topic, imgsz, conf, iou, half, show, show_masks):
        super().__init__('yolo_realtime_infer')
        self.get_logger().info(f'加载模型: {model_path}')
        self.model = YOLO(model_path)

        # 预热模型
        self.get_logger().info(f'模型预热中 (half={half})...')
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model(dummy, imgsz=imgsz, half=half, verbose=False)
        self.get_logger().info('模型预热完成！')

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.show = show
        self.show_masks = show_masks
        self.bridge = CvBridge()

        # FPS 统计
        self.fps_window = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()

        # 订阅相机话题
        self.get_logger().info(f'订阅话题: {topic}')
        self.sub = self.create_subscription(
            Image, topic, self.image_callback, 10
        )

    def image_callback(self, msg):
        t0 = time.time()

        # 将 ROS Image 转为 OpenCV 格式
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        # YOLO 推理 (支持 half 和 iou 参数)
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, half=self.half, verbose=False)

        # 绘制结果 (支持控制是否绘制 masks)
        annotated = results[0].plot(masks=self.show_masks)

        # 计算 FPS
        t1 = time.time()
        dt = t1 - t0
        self.fps_window.append(dt)
        fps_avg = len(self.fps_window) / sum(self.fps_window)
        self.frame_count += 1

        # 在画面上显示状态
        state_text = f'FPS: {fps_avg:.1f} | Half: {self.half} | Masks: {self.show_masks}'
        cv2.putText(annotated, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, f'Infer: {dt*1000:.1f}ms | IoU: {self.iou}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 每秒在终端打印一次统计
        if self.frame_count % 30 == 0:
            elapsed = t1 - self.start_time
            overall_fps = self.frame_count / elapsed
            self.get_logger().info(
                f'帧数: {self.frame_count} | '
                f'平均FPS: {overall_fps:.1f} | '
                f'实时FPS: {fps_avg:.1f} | '
                f'推理时间: {dt*1000:.1f}ms'
            )

        if self.show:
            cv2.imshow('YOLO Realtime Inference', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('用户按下 q，退出...')
                cv2.destroyAllWindows()
                rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='ROS2 YOLO 实时推理')
    parser.add_argument('--model', type=str, default='yolo26n.pt',
                        help='YOLO 模型路径')
    parser.add_argument('--topic', type=str, default='/camera/color/image_raw',
                        help='ROS2 图像话题')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理分辨率')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU 阈值')
    parser.add_argument('--half', action='store_true',
                        help='使用 FP16 半精度推理')
    parser.add_argument('--no-masks', action='store_true',
                        help='显示分割结果时不绘制 mask (仅显示框)')
    parser.add_argument('--no-show', action='store_true',
                        help='不显示 OpenCV 窗口')
    args = parser.parse_args()

    rclpy.init()
    node = YoloRealtimeNode(
        model_path=args.model,
        topic=args.topic,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        show=not args.no_show,
        show_masks=not args.no_masks,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        total_time = time.time() - node.start_time
        avg_fps = node.frame_count / max(total_time, 1e-6)
        node.get_logger().info(f'总计处理 {node.frame_count} 帧, 平均 FPS: {avg_fps:.1f}')
        cv2.destroyAllWindows()
        node.destroy_node()


if __name__ == '__main__':
    main()
