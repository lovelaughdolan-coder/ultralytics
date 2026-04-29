#!/usr/bin/env python3
"""
ROS2 + YOLO 姿态实时推理节点
订阅奥比中光相机的彩色图像话题，使用 YOLO pose 模型进行实时推理。
在 OpenCV 窗口中显示关键点结果和实时 FPS。

用法:
  1. 启动相机: ros2 launch orbbec_camera gemini_330_series.launch.py
  2. 运行推理: python3 yolo_realtime_pose_infer.py [--model yolov8n-pose.pt] [--topic /camera_01/color/image_raw]

按 'q' 退出。
"""

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
# 彻底禁用 cv_bridge，因为它与 Numpy 2.x 二进制库不兼容
CvBridge = None

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
except Exception:
    PILImage = ImageDraw = ImageFont = None

from ultralytics import YOLO


def _load_chinese_font(size=24):
    if ImageFont is None:
        return None
    candidates = [
        os.environ.get('YOLO_CN_FONT'),
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


def put_text_cn(img, text, org, font, color=(255, 0, 0)):
    if PILImage is None or ImageDraw is None or font is None:
        return img
    x, y = org
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def simple_imgmsg_to_cv2(img_msg):
    """
    一种不依赖 cv_bridge 的简单实现，直接从 sensor_msgs/Image 提取 numpy 数组。
    仅支持常见的 'rgb8' 和 'bgr8' 编码。
    """
    if img_msg.encoding not in ['rgb8', 'bgr8']:
        raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
    
    dtype = np.uint8
    n_channels = 3
    
    # 转换为 numpy 数组
    im = np.frombuffer(img_msg.data, dtype=dtype).reshape(img_msg.height, img_msg.width, n_channels)
    
    if img_msg.encoding == 'rgb8':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
    return im.copy()


class YoloPoseRealtimeNode(Node):
    def __init__(self, model_path, topic, imgsz, conf, iou, half, agnostic_nms, classes, show, show_masks, use_track, tracker, pub_topic, target_class):
        super().__init__('yolo_pose_realtime_infer')
        self.get_logger().info(f'加载模型: {model_path}')
        self.model = YOLO(model_path)

        self.cn_font = _load_chinese_font(size=24)
        if self.cn_font is None:
            self.get_logger().warning('未找到中文字体，OpenCV 画面中文可能无法显示。可设置 YOLO_CN_FONT 环境变量指定字体路径。')

        # 预热模型
        self.get_logger().info(f'模型预热中 (half={half})...')
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model(dummy, imgsz=imgsz, half=half, verbose=False)
        self.get_logger().info('模型预热完成！')

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.show = show
        self.show_masks = show_masks
        self.use_track = use_track
        self.tracker = tracker
        self.target_class = target_class
        if CvBridge is None:
            self.get_logger().info('使用内置图像转换逻辑 (Numpy 2.x 兼容模式)。')
            self.bridge = None
        else:
            self.bridge = CvBridge()

        # FPS 统计
        self.fps_window = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()
        self._shutdown_requested = False

        # 目标像素坐标发布器
        self.pub_topic = pub_topic
        self.target_pub = self.create_publisher(PointStamped, pub_topic, 10)
        self.get_logger().info(f'发布目标像素坐标到: {pub_topic}')

        # 订阅相机话题
        self.get_logger().info(f'订阅话题: {topic}')
        self.sub = self.create_subscription(
            Image, topic, self.image_callback, 10
        )

    def image_callback(self, msg):
        t0 = time.time()

        # 将 ROS Image 转为 OpenCV 格式
        try:
            # 强制使用自定义转换函数，避开 cv_bridge 的崩溃风险
            frame = simple_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        # YOLO 推理（track 模式可平滑帧间抖动）
        common_args = dict(
            imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            half=self.half, agnostic_nms=self.agnostic_nms,
            classes=self.classes, verbose=False,
        )
        if self.use_track:
            results = self.model.track(frame, persist=True, tracker=self.tracker, **common_args)
        else:
            results = self.model(frame, **common_args)

        # 绘制结果 (姿态关键点由 plot() 自动绘制)
        annotated = results[0].plot(masks=self.show_masks)

        # 标注关键点索引
        kpts = results[0].keypoints
        if kpts is not None:
            kpt_xy = kpts.xy
            kpt_conf = kpts.conf
            for det_i in range(len(kpt_xy)):
                for kpt_i, (x, y) in enumerate(kpt_xy[det_i]):
                    if kpt_conf is not None and kpt_conf[det_i][kpt_i] < 0.2:
                        continue
                    cv2.putText(annotated, str(kpt_i), (int(x) + 2, int(y) - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # 关键点 6-7 方向向量与左右标注
            for det_i in range(len(kpt_xy)):
                if kpt_xy.shape[1] <= 7:
                    continue
                if kpt_conf is not None:
                    if kpt_conf[det_i][6] < 0.2 or kpt_conf[det_i][7] < 0.2:
                        continue
                x6, y6 = kpt_xy[det_i][6]
                x7, y7 = kpt_xy[det_i][7]
                p6 = (int(x6), int(y6))
                p7 = (int(x7), int(y7))
                cv2.arrowedLine(annotated, p6, p7, (255, 0, 0), 2, tipLength=0.2)
                direction = '左' if (x7 - x6) < 0 else '右'
                mid = (int((p6[0] + p7[0]) / 2), int((p6[1] + p7[1]) / 2))
                annotated = put_text_cn(annotated, direction, (mid[0] + 4, mid[1] - 18),
                                        self.cn_font, color=(255, 0, 0))

        # 提取最小 ID 目标并发布 (u, v)
        target_info = self._publish_target(results, annotated, msg.header.stamp)

        # 计算 FPS
        t1 = time.time()
        dt = t1 - t0
        self.fps_window.append(dt)
        fps_avg = len(self.fps_window) / sum(self.fps_window)
        self.frame_count += 1

        # 在画面上显示状态
        mode = f'Track({self.tracker})' if self.use_track else 'Detect'
        state_text = f'FPS: {fps_avg:.1f} | {mode} | Half: {self.half}'
        cv2.putText(annotated, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        target_str = f'ID:{target_info["id"]} ({target_info["u"]},{target_info["v"]})' if target_info else 'No target'
        cv2.putText(annotated, f'Infer: {dt*1000:.1f}ms | Target: {target_str}', (10, 60),
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
            cv2.imshow('YOLO Pose Realtime Inference', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('用户按下 q，退出...')
                self._shutdown_requested = True
                raise SystemExit

        if self._shutdown_requested:
            return

    def _publish_target(self, results, annotated, stamp):
        """从跟踪结果中找目标类别里 ID 最小的物体，发布其 (u,v) 像素坐标"""
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # 获取跟踪 ID
        if boxes.id is None:
            return None

        ids = boxes.id.cpu().numpy().astype(int)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        # 筛选目标类别
        candidates = []
        for i in range(len(ids)):
            if cls_ids[i] == self.target_class:
                candidates.append((ids[i], i))

        if not candidates:
            return None

        # 找 ID 最小的
        candidates.sort(key=lambda x: x[0])
        min_id, idx = candidates[0]

        # 计算包围框中心 (u, v) 和面积
        x1, y1, x2, y2 = xyxy[idx]
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        area = float((x2 - x1) * (y2 - y1))

        # 发布 PointStamped (x=u, y=v, z=bbox_area)
        pt_msg = PointStamped()
        pt_msg.header.stamp = stamp
        pt_msg.header.frame_id = 'camera_color_optical_frame'
        pt_msg.point.x = float(u)
        pt_msg.point.y = float(v)
        pt_msg.point.z = area
        self.target_pub.publish(pt_msg)

        # 在画面上高亮该目标（蓝色十字 + 圆圈）
        color = (255, 100, 0)  # 蓝色
        cv2.drawMarker(annotated, (u, v), color, cv2.MARKER_CROSS, 30, 2)
        cv2.circle(annotated, (u, v), 20, color, 2)
        cv2.putText(annotated, f'T{min_id} A:{area:.0f}', (u + 25, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return {'id': min_id, 'u': u, 'v': v, 'area': area}


def main():
    parser = argparse.ArgumentParser(description='ROS2 YOLO Pose 实时推理')
    parser.add_argument('--model', type=str, default='runs/pose/runs/train/animal_pose_v24/weights/best.pt',
                        help='YOLO pose 模型路径')
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
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='跨类别 NMS，抑制不同类别间的重叠检测')
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='只检测指定类别，如 --classes 0 或 --classes 0 2')
    parser.add_argument('--no-track', action='store_true',
                        help='关闭跟踪模式（默认启用 ByteTrack 跟踪以减少闪烁）')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                        help='跟踪器配置 (bytetrack.yaml 或 botsort.yaml)')
    parser.add_argument('--no-masks', action='store_true',
                        help='不绘制 masks')
    parser.add_argument('--pub-topic', type=str, default='/yolo/target_pixel',
                        help='目标像素坐标发布话题')
    parser.add_argument('--target-class', type=int, default=0,
                        help='要跟踪的目标类别 ID（发布该类别中 ID 最小的物体的 u,v）')
    parser.add_argument('--no-show', action='store_true',
                        help='不显示 OpenCV 窗口')
    args = parser.parse_args()

    rclpy.init()
    node = YoloPoseRealtimeNode(
        model_path=args.model,
        topic=args.topic,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        agnostic_nms=args.agnostic_nms,
        classes=args.classes,
        show=not args.no_show,
        show_masks=not args.no_masks,
        use_track=not args.no_track,
        tracker=args.tracker,
        pub_topic=args.pub_topic,
        target_class=args.target_class,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        total_time = time.time() - node.start_time
        avg_fps = node.frame_count / max(total_time, 1e-6)
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            node.get_logger().info(f'总计处理 {node.frame_count} 帧, 平均 FPS: {avg_fps:.1f}')
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
