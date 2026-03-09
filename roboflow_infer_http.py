
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient

video_path = '/home/hxy/Videos/7172e18bc4369fab55c2aa4b1e79c703.mp4'
output_path = '/home/hxy/ultralytics/output.mp4'

client = InferenceHTTPClient.init(
    api_url='https://detect.roboflow.com',
    api_key='nkgoQBCFfdjPAZTo8iRt'
)

# Initialize Video Generators
video_info = sv.VideoInfo.from_video_path(video_path)
generator = sv.get_video_frames_generator(video_path)

# Initialize Drawers (Assuming Keypoint extraction or general BBox)
# Note: For custom keypoints, we may write custom draw logic, 
# here we just try to get the inference going and show raw JSON print for simplicity.

print(f'{video_info.total_frames} frames to process.')
fps = video_info.fps
h, w = video_info.resolution_wh[1], video_info.resolution_wh[0]

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

for i, frame in enumerate(generator):
    try:
        # Run inference on current frame
        result = client.run_workflow(
            workspace_name='claras-workspace-lkmpi',
            workflow_id='custom-workflow-2',
            images={'image': frame}
        )
        
        # We can implement visualization based on what 'predictions' contains, 
        # but since WebRTC failed to get 'output_1', we focus on running processing.
        # Adding simple text to show it's processed
        cv2.putText(frame, f"Frame {i} Processed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        
        if i % 10 == 0:
            print(f'Processed frame {i}/{video_info.total_frames}')
    except Exception as e:
        print(f'Error processing frame {i}:', e)

out.release()
print(f'Video saved to {output_path}')

