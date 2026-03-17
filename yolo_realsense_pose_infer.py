#!/usr/bin/env python3
"""
RealSense + YOLO 姿态实时推理

用法:
  python3 yolo_realsense_pose_infer.py --model yolov8n-pose.pt
按 'q' 退出。
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


def parse_args():
    parser = argparse.ArgumentParser(description='RealSense YOLO Pose Realtime Inference')
    parser.add_argument('--model', type=str, default='yolov8n-pose.pt', help='YOLO pose 模型路径')
    parser.add_argument('--imgsz', type=int, default=640, help='推理分辨率')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--half', action='store_true', help='使用 FP16 半精度推理')
    parser.add_argument('--agnostic-nms', action='store_true', help='跨类别 NMS')
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='仅检测指定类别')
    parser.add_argument('--no-track', action='store_true', help='关闭跟踪模式（默认启用 ByteTrack）')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='跟踪器配置')
    parser.add_argument('--no-show', action='store_true', help='不显示 OpenCV 窗口')
    parser.add_argument('--width', type=int, default=1280, help='RealSense 宽度')
    parser.add_argument('--height', type=int, default=720, help='RealSense 高度')
    parser.add_argument('--fps', type=int, default=30, help='RealSense 帧率')
    return parser.parse_args()


def main():
    args = parse_args()

    if not HAS_REALSENSE:
        raise RuntimeError('未检测到 pyrealsense2，请先安装 RealSense SDK/Python 包')

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # 预热
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model(dummy, imgsz=args.imgsz, half=args.half, verbose=False)
    print('Model warmup done.')

    # RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    temporal_filter = rs.temporal_filter()

    fps_window = deque(maxlen=30)
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            t0 = time.time()

            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            depth_frame = temporal_filter.process(depth_frame)

            frame = np.asanyarray(color_frame.get_data())

            common_args = dict(
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                half=args.half,
                agnostic_nms=args.agnostic_nms,
                classes=args.classes,
                verbose=False,
            )
            if args.no_track:
                results = model(frame, **common_args)
            else:
                results = model.track(frame, persist=True, tracker=args.tracker, **common_args)

            annotated = results[0].plot()

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
                    cv2.putText(annotated, direction, (mid[0] + 4, mid[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            t1 = time.time()
            dt = t1 - t0
            fps_window.append(dt)
            fps_avg = len(fps_window) / max(sum(fps_window), 1e-6)
            frame_count += 1

            mode = f"Track({args.tracker})" if not args.no_track else "Detect"
            state_text = f"FPS: {fps_avg:.1f} | {mode} | Half: {args.half}"
            cv2.putText(annotated, state_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Infer: {dt*1000:.1f}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if frame_count % 30 == 0:
                elapsed = t1 - start_time
                overall_fps = frame_count / max(elapsed, 1e-6)
                print(f"Frames: {frame_count} | Avg FPS: {overall_fps:.1f} | Live FPS: {fps_avg:.1f} | Infer: {dt*1000:.1f}ms")

            if not args.no_show:
                cv2.imshow('YOLO Pose RealSense', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
