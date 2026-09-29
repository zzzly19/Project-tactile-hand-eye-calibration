import pyrealsense2 as rs
import numpy as np
import cv2

pipeline=rs.pipeline()
#config=rs.config
#config.enable_stream(rs.stream.color,0,640,480,rs.format.bgr8,30)
print("start camera")
pipeline.start()

try:
    frames=pipeline.wait_for_frames(10000)
    color_frame = frames.get_color_frame()
    if not color_frame: 
        print("no color frame received")
    else:
        img = np.asanyarray(color_frame.get_data())
        print("frame shape:",img.shape)
finally:
    pipeline.stop()
    print("stopped")
