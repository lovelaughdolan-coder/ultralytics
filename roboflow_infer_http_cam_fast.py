import cv2
import time
import queue
import threading
import numpy as np
import pyorbbecsdk as ob
from inference_sdk import InferenceHTTPClient

# ===== Roboflow HTTP Client 配置 =====
client = InferenceHTTPClient.init(
    api_url="https://detect.roboflow.com",
    api_key="nkgoQBCFfdjPAZTo8iRt"
)

# ===== 队列与多线程状态 =====
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)
running = True
latest_predictions = []  # 保存最新的推理结果用于在视频流上连续绘制
latest_infer_time = 0

# ===== 1. 推理线程 =====
def inference_thread():
    global latest_predictions, latest_infer_time
    while running:
        try:
            # 获取最新帧，丢弃积压帧
            bgr = frame_queue.get(timeout=1)
            
            # 清空队列中剩余的帧，保持实时性
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            start_t = time.time()
            # 可以稍微缩小减小网络压力（比如 640），但不至于小到 416
            h, w = bgr.shape[:2]
            scale = 1.0
            if w > 640:
                scale = 640 / w
                bgr_resized = cv2.resize(bgr, (640, int(h * scale)))
            else:
                bgr_resized = bgr
                
            # 发送 HTTP 请求
            result = client.infer_from_workflow(
                workspace_name="claras-workspace-lkmpi",
                workflow_name="custom-workflow-2",
                images={"image": bgr_resized}
            )
            
            infer_time = (time.time() - start_t) * 1000
            
            # 不要打印 result，因为里面是巨大的 Basi64 图片，会导致终端乱码卡死
            
            preds = []
            rendered_image = None
            
            outputs_dict = {}
            if isinstance(result, list) and len(result) > 0:
                outputs_dict = result[0]
            elif isinstance(result, dict):
                outputs_list = result.get('outputs', [])
                if isinstance(outputs_list, list) and len(outputs_list) > 0:
                    outputs_dict = outputs_list[0]
                else:
                    outputs_dict = result
                   
            if outputs_dict:
                # 打印调试信息，避免乱码，只看结构
                print(f"--- API 返回 (耗时 {infer_time:.0f}ms) ---")
                for k, v in outputs_dict.items():
                    if isinstance(v, str):
                        print(f"  Key: {k}, Type: str (length: {len(v)})")
                    elif isinstance(v, list):
                        print(f"  Key: {k}, Type: list (length: {len(v)})")
                    else:
                        print(f"  Key: {k}, Type: {type(v)}")
                        
                # custom-workflow-2 目前看来只返回了 'output_1'，里面是已经用云端节点画好框的 base64 图片
                if 'output_1' in outputs_dict and isinstance(outputs_dict['output_1'], str):
                    try:
                        import base64
                        img_data = base64.b64decode(outputs_dict['output_1'])
                        np_arr = np.frombuffer(img_data, np.uint8)
                        rendered_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    except Exception as e:
                        print(f"Failed to decode base64: {e}")
                else:
                    # 如果不是字符串图片，尝试解析 predictions 坐标
                    target_key = None
                    if 'predictions' in outputs_dict:
                        target_key = 'predictions'
                    elif 'output_1' in outputs_dict and isinstance(outputs_dict['output_1'], list):
                        target_key = 'output_1'
                    elif 'output' in outputs_dict:
                        target_key = 'output'
                        
                    if target_key and isinstance(outputs_dict[target_key], list):
                        for p in outputs_dict[target_key]:
                            if not isinstance(p, dict):
                                continue # 可能只是数值或其他
                            x = int(p.get('x', 0) / scale)
                            y = int(p.get('y', 0) / scale)
                            wt = int(p.get('width', 0) / scale)
                            ht = int(p.get('height', 0) / scale)
                            cls = p.get('class', 'object')
                            conf = p.get('confidence', 0)
                            preds.append({'x': x, 'y': y, 'w': wt, 'h': ht, 'class': cls, 'conf': conf})
                           
            latest_predictions = (preds, rendered_image)
            latest_infer_time = infer_time
            
            # 把结果放队列里通知显示线程（可省）
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Inference thread error: {e}")
            time.sleep(0.5)

# ===== 2. 主程序 (抓帧 + 显示) =====
def main():
    global running, latest_predictions, latest_infer_time
    
    ctx = ob.Context()
    dev_list = ctx.query_devices()
    if len(dev_list) == 0:
        raise RuntimeError("No Orbbec device found!")

    dev = dev_list[0]
    info = dev.get_device_info()
    print(f"Using device: {info.get_name()}, SN: {info.get_serial_number()}")

    pipe = ob.Pipeline(dev)
    ob_config = ob.Config()
    profile_list = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    print(f"Color stream: {color_profile.get_width()}x{color_profile.get_height()} "
          f"@ {color_profile.get_fps()} fps, format={color_profile.get_format()}")
    ob_config.enable_stream(color_profile)

    pipe.start(ob_config)
    
    # 启动推理线程
    infer_t = threading.Thread(target=inference_thread, daemon=True)
    infer_t.start()

    print("Starting fast live inference... Press 'q' to quit.")
    
    try:
        while True:
            # 抓帧
            frameset = pipe.wait_for_frames(100)
            if frameset is None:
                continue
            color_frame = frameset.get_color_frame()
            if color_frame is None:
                continue
            
            # 转 BGR
            data = np.asanyarray(color_frame.get_data())
            fmt = color_frame.get_format()
            
            if fmt == ob.OBFormat.MJPG:
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif fmt == ob.OBFormat.RGB:
                bgr = cv2.cvtColor(data.reshape(
                    color_profile.get_height(), color_profile.get_width(), 3
                ), cv2.COLOR_RGB2BGR)
            elif fmt == ob.OBFormat.YUYV:
                bgr = cv2.cvtColor(data.reshape(
                    color_profile.get_height(), color_profile.get_width(), 2
                ), cv2.COLOR_YUV2BGR_YUYV)
            else:
                bgr = data.reshape(color_profile.get_height(), color_profile.get_width(), 3)
                
            if bgr is None:
                continue
                
            # 发送一帧给推理线程 (非阻塞)
            if not frame_queue.full():
                frame_queue.put_nowait(bgr.copy())
            
            # 绘制最新的推理结果
            if isinstance(latest_predictions, tuple):
                preds, rendered_image = latest_predictions
            else:
                preds, rendered_image = latest_predictions, None
                
            # 如果云端返回了渲染好的图像，我们就展示它。为了清晰和铺满窗口，我们放大回原始画面的尺寸
            if rendered_image is not None:
                # 恢复到原生捕捉视角的尺寸 (如 1280x720) 以铺满全屏
                display_img = cv2.resize(rendered_image, (bgr.shape[1], bgr.shape[0]))
            else:
                display_img = bgr

            cv2.putText(display_img, f"API Latency: {latest_infer_time:.0f}ms", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if rendered_image is None:
                cv2.putText(display_img, f"Detected: {len(preds)}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for p in preds:
                    x, y, w, h = p['x'], p['y'], p['w'], p['h']
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"{p['class']} {p['conf']:.2f}"
                    cv2.putText(display_img, label, (x1, max(y1-10, 0)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 显示
            cv2.imshow("Fast Live Inference", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        pipe.stop()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
