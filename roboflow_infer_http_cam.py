import cv2
import time
import numpy as np
import pyorbbecsdk as ob
from inference_sdk import InferenceHTTPClient

# ===== Roboflow HTTP Client 配置 =====
client = InferenceHTTPClient.init(
    api_url="https://detect.roboflow.com", # HTTP API 用 detect
    api_key="nkgoQBCFfdjPAZTo8iRt"
)

# ===== Orbbec 摄像头配置 =====
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

print("Starting HTTP-based live inference... Press 'q' to quit.")
pipe.start(ob_config)

try:
    while True:
        # 1. 抓取彩色帧
        frameset = pipe.wait_for_frames(100)
        if frameset is None:
            continue
        color_frame = frameset.get_color_frame()
        if color_frame is None:
            continue
        
        # 2. 转换为 BGR
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
            
        # 3. 本地发给 HTTP API 推理（利用 HTTP 规避 WebRTC 的复杂事件循环）
        try:
            # 推理图片字典 {"image": img_array} 送入你设定的输入节点
            result = client.infer_from_workflow(
                workspace_name="claras-workspace-lkmpi",
                workflow_name="custom-workflow-2",
                images={"image": bgr}
            )
            
            # 画一个简单的标记表示处理完成
            cv2.putText(bgr, "Inferencing... (HTTP)", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 简单展示工作流返回的输出（如果有 predictions 数组）
            if isinstance(result, dict) and 'outputs' in result and isinstance(result['outputs'], list):
               if len(result['outputs']) > 0 and 'predictions' in result['outputs'][0]:
                   for p in result['outputs'][0]['predictions']:
                    x = int(p.get('x', 0))
                    y = int(p.get('y', 0))
                    w = int(p.get('width', 0))
                    h = int(p.get('height', 0))
                    cls = p.get('class', 'object')
                    conf = p.get('confidence', 0)
                    
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(bgr, f"{cls} {conf:.2f}", (x1, max(y1-10, 0)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(bgr, f"API Error", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Inference error: {e}")

        # 4. 显示画面
        cv2.imshow("Roboflow Live HTTP", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    pipe.stop()
    cv2.destroyAllWindows()
    print("Cleaned up devices and windows.")
