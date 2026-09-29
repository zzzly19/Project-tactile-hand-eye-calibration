import pyrealsense2 as rs
import numpy as np
import cv2
import time

MARKER_SIZE = 0.1  # 5x5 cm marker (must match actual printed size)


calib_data = np.load("camera_calibration.npz", allow_pickle=True)
mtx, dist = calib_data["mtx"], calib_data["dist"]

aruco=cv2.aruco
aruco_dict=aruco.Dictionary_get(aruco.DICT_5X5_1000)
aruco_params=aruco.DetectorParameters_create()

'''img=cv2.imread("aruco5x5_1000_id0_REAL.png")
if img is None:
    print("img not found")
    exit()'''
pipeline=rs.pipeline()

#config=rs.config
#config.enable_stream(rs.stream.color,0,640,480,rs.format.bgr8,30)

print("start camera")
pipeline.start()
time.sleep(2)


# Marker corner points (3D)
obj_points = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],  # Top-left
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],  # Top-right
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],  # Bottom-right
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]   # Bottom-left
], dtype=np.float32)


try:
    '''frames=pipeline.wait_for_frames(10000)
    color_frame = frames.get_color_frame()'''
    while True:
        color_frame=None
        for i in range(10):
            frames = pipeline.wait_for_frames(10000)###none
            color_frame = frames.get_color_frame()
            if color_frame:
                break
                time.sleep(0.2)
        if not color_frame: 
            raise RuntimeError("no color frame received")
    

        img = np.asanyarray(color_frame.get_data())
        print("frame shape:",img.shape)

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(
                gray,
                aruco_dict,
                parameters=aruco_params
            )  
    
        print("ids:",ids)
        print("number of detected markers:",len(corners))

        if ids is None:
        
           print("no marker detected")
        else:
            for i in range(len(ids)):

                print("current marker id:",ids[i][0])
                print("corners[i]:",corners[i])
                print("corners[i] shape:",corners[i].shape)


                # Estimate pose
                success, rvec, tvec = cv2.solvePnP(obj_points, corners[i], mtx, dist)#success->_

                print("solvePnP success:",success)
                print("rvec:",rvec)
                print("tvec:",tvec)


                # Draw marker and axes
                cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, MARKER_SIZE/2)
                aruco.drawDetectedMarkers(img, corners, ids)

                # Display coordinates (convert to cm)
                x, y, z = tvec.flatten() * 100  # Convert to cm
                cv2.putText(img, f"ID {ids[i][0]}", (10, 30 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(img, f"X: {x:+.1f} cm", (10, 60 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(img, f"Y: {y:+.1f} cm", (10, 90 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(img, f"Z: {z:+.1f} cm", (10, 120 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        
        
            print("marker ids:",ids.flatten())
            aruco.drawDetectedMarkers(img,corners,ids)
    
        cv2.imshow("ArUco Marker Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #cv2.imwrite("arucotestresult.png",img)
        #print("saved img")


finally:
    pipeline.stop()
    print("stopped")
