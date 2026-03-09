import cv2
import time
import numpy as np
import pyorbbecsdk as ob
from inference_sdk import InferenceHTTPClient

# ===== Roboflow HTTP Client 配置 =====
client = InferenceHTTPClient.init(
    api_url="https://detect.roboflow.com",
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

pipe.start(ob_config)

print("Capturing ONE frame for debugging...")

bgr = None
# 循环几次为了等相机自动曝光稳定
for i in range(10):
    frameset = pipe.wait_for_frames(100)
    if frameset is None: continue
    color_frame = frameset.get_color_frame()
    if color_frame is None: continue
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
pipe.stop()

if bgr is None:
    print("Failed to capture frame.")
    exit(1)

cv2.imwrite("debug_input.jpg", bgr)
print(f"Captured debug_input.jpg, shape {bgr.shape}. Sending to Roboflow HTTP API...")

try:
    # 缩小发给官方以提速
    h, w = bgr.shape[:2]
    scale = 1.0
    if w > 416:
        scale = 416 / w
        bgr = cv2.resize(bgr, (416, int(h * scale)))
        
    start_t = time.time()
    result = client.infer_from_workflow(
        workspace_name="claras-workspace-lkmpi",
        workflow_name="custom-workflow-2",
        images={"image": bgr}
    )
    print(f"HTTP Inference took {(time.time() - start_t)*1000:.0f} ms")
    
    print("\n====== REAL API RESULT ======")
    import json
    # 我们拦截 base64 打印以免卡死
    safe_result = {}
    if isinstance(result, dict) and 'outputs' in result:
        for i, opt in enumerate(result['outputs']):
            safe_opt = {}
            for k, v in opt.items():
                if isinstance(v, str) and len(v) > 200:
                    safe_opt[k] = f"<Base64 String, len {len(v)}>"
                else:
                    safe_opt[k] = v
            safe_result[f'outputs[{i}]'] = safe_opt
    else:
        safe_result = result
        
    print(json.dumps(safe_result, indent=2, ensure_ascii=False))

except Exception as e:
    print(f"Error: {e}")
