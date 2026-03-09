import cv2
import time
import asyncio
import threading
import numpy as np
import pyorbbecsdk as ob
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import ManualSource, StreamConfig, VideoMetadata

# ===== Roboflow 配置 =====
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="nkgoQBCFfdjPAZTo8iRt"
)

source = ManualSource()

config = StreamConfig(
    stream_output=["output_1"],
    data_output=[],
    requested_plan="webrtc-gpu-medium",
    requested_region="us",
)

session = client.webrtc.stream(
    source=source,
    workflow="custom-workflow-2",
    workspace="claras-workspace-lkmpi",
    image_input="image",
    config=config
)

# ===== Orbbec 摄像头配置 =====
ctx = ob.Context()
dev_list = ctx.query_devices()
if len(dev_list) == 0:
    raise RuntimeError("No Orbbec device found!")

dev = dev_list[0]
info = dev.get_device_info()
print(f"Using device: {info.get_name()}")

pipe = ob.Pipeline(dev)
ob_config = ob.Config()
profile_list = pipe.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
color_profile = profile_list.get_default_video_stream_profile()
ob_config.enable_stream(color_profile)

running = True

def orbbec_capture_thread():
    pipe.start(ob_config)
    print("Orbbec capture started!")
    
    while running:
        try:
            frameset = pipe.wait_for_frames(100)
            if not frameset: continue
            
            color_frame = frameset.get_color_frame()
            if not color_frame: continue
            
            data = np.asanyarray(color_frame.get_data())
            fmt = color_frame.get_format()
            
            if fmt in [ob.OBFormat.MJPG, ob.OBFormat.MJPEG]:
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            else:
                bgr = data.reshape(color_profile.get_height(), color_profile.get_width(), 3)
                if fmt == ob.OBFormat.RGB:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            
            if bgr is not None:
                source.send(bgr)
        except Exception as e:
            time.sleep(0.01)
    
    pipe.stop()

# 由于 cv2.waitKey 会阻塞主线程（进而阻塞 asyncio 事件循环，导致 WebRTC 心跳断开），
# 官方的 session.run() 是一个阻塞的快捷方式，但在这种自定义循环中，我们需要手工管理 asyncio。
# 比较安全的做法是用一个后台线程显示 cv2，或者在主线程显示但频繁调用 asyncio.sleep(0)。
# 为了避免引入太多复杂性，由于 session.on_frame 也是异步回调包装器，我们用子线程显示：

frame_queue = []

@session.on_frame
def receive_frame(frame, metadata):
    # 只保存最新一帧显示
    if len(frame_queue) == 0:
        frame_queue.append(frame)
    else:
        frame_queue[0] = frame

def display_thread():
    while running:
        if frame_queue:
            cv2.imshow("Roboflow Live", frame_queue[0])
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            session.close() # triggers run() to exit
            break
        time.sleep(0.01)

print("Starting pipeline...")
capture_t = threading.Thread(target=orbbec_capture_thread, daemon=True)
capture_t.start()
display_t = threading.Thread(target=display_thread, daemon=True)
display_t.start()

try:
    session.run() # blocks event loop to keep WebRTC alive
except KeyboardInterrupt:
    print("Interrupted")
finally:
    running = False
    cv2.destroyAllWindows()
