
# This script is used to verify the calibration matrix computed by TactiCal.py
# You have to manually plug in the calibration matrix at the main function to apply the calibration.


import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

class CalibrationTester:
    def __init__(self, T_base_cam, marker_size=0.1):
        # Initialize camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # Load camera calibration
        calibration_data = np.load('camera_calibration.npz')
        self.camera_matrix = calibration_data['mtx']
        self.dist_coeffs = calibration_data['dist']

        # Aruco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self.parameters = cv2.aruco.DetectorParameters()
        self.marker_size = marker_size  # meters

        # Hand-eye calibration matrix
        self.T_base_cam = T_base_cam

    def get_marker_pose(self):
        """Detect Aruco marker and return its pose in camera frame"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # 1) Detect coarse markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters)

        if ids is None:
            print("Error: No markers detected!")
            return None, None
            

        # 2) Refine corners to sub-pixel accuracy
        #   Flatten to Nx1x2 for cornerSubPix
        corners_subpix = []
        for c in corners:
            # c is 1×4×2 → 4×2 float32
            pts = c.reshape(-1, 2).astype(np.float32)
            # termination: either 30 iters or 0.01px
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv2.cornerSubPix(gray, pts, winSize=(5,5), zeroZone=(-1,-1), criteria=term)
            corners_subpix.append(pts.reshape(1, -1, 2))
        corners = corners_subpix

        # 3) Estimate pose using refined corners
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[0], self.marker_size, self.camera_matrix, self.dist_coeffs)
        
        T_marker_cam = np.eye(4)
        T_marker_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_marker_cam[:3, 3] = tvec[0][0]


        T_cam_marker = np.linalg.inv(T_marker_cam)

        # Visualize detection
        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        cv2.drawFrameAxes(color_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size)
        cv2.imshow('Marker Detection', cv2.resize(color_image, (960, 540)))
        
        return T_cam_marker, color_image

    def run_test(self):
        """Continuous test of the calibration matrix"""
        print("Starting calibration test...")
        print("Press 'q' to quit")
        
        while True:
            T_cam_marker, img = self.get_marker_pose()
            
            
            if T_cam_marker is not None:
                # Transform to robot base coordinates
                T_base_marker = self.T_base_cam @ np.linalg.inv(T_cam_marker)
                
                pos_m = T_base_marker[:3, 3]
                ori_rad = R.from_matrix(T_base_marker[:3, :3]).as_euler('xyz')
                
                # Overlay coordinates on image
                text = f"Base Frame: X={pos_m[0]* 1000:.3f}mm, Y={pos_m[1]* 1000:.3f}mm, Z={pos_m[2]* 1000:.3f}mm"
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                text = f"Orientation: R={np.degrees(ori_rad[0]):.1f}°, P={np.degrees(ori_rad[1]):.1f}°, Y={np.degrees(ori_rad[2]):.1f}°"
                cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Marker Detection', cv2.resize(img, (960, 540)))
                cv2.waitKey(0)
            
            key = cv2.waitKey(10)
            if key == ord('q'):
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Plug in your calibration matrix here
    T_base_cam = np.array([
                [ 0.002678, -0.999981,  0.005488,  0.336701],
                [-0.999641, -0.00253,   0.026673, -0.16973 ],
                [-0.026659, -0.005558, -0.999629,  0.762252],
                [ 0.,        0.,        0.,        1.      ]
    ])
    
    tester = CalibrationTester(T_base_cam)
    tester.run_test()
