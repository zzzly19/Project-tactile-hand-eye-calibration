# collects data points using april grid to calibrate the intrinsic parameters of the camera
# press c to capture frames showing the april grid from various angles, press q to quit capturing and get the calibration file. >5 frames required fpr calibration
import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # Enable depth

# Start pipeline
pipeline.start(config)

# Align depth to color frame
align = rs.align(rs.stream.color)

# Initialize AprilTag detector
detector = apriltag.Detector(apriltag.DetectorOptions(families="tag16h5"))

# Lists to store calibration points
image_points = []
object_points = []

# Define tag size in meters
tag_size = 0.032  

captured_frame_count = 0
capture_frame = False

try:
    print("Press 'c' to capture frames, 'q' to quit.")

    while True:
        # Wait for frames and align depth to color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get depth and color frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue  # Skip if frames are missing

        # Convert color frame to grayscale
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        results = detector.detect(gray_image)

        num_points = 0  # Track how many 3D-2D correspondences we have in this frame
        obj_pts = []
        img_pts = []

        if results:
            for r in results:
                corners = r.corners.astype(np.int32)
                cv2.polylines(color_image, [corners], True, (0, 255, 0), 2)

                # Get camera intrinsics from RealSense
                intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                fx, fy = intrinsics.fx, intrinsics.fy
                cx, cy = intrinsics.ppx, intrinsics.ppy
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                for corner in r.corners:
                    x, y = int(corner[0]), int(corner[1])
                    depth = depth_frame.get_distance(x, y)

                    if depth > 0:
                        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                        obj_pts.append(point_3d)
                        img_pts.append([x, y])
                        num_points += 1

        if capture_frame and num_points >= 6:
            captured_frame_count += 1
            print(f"Captured frame {captured_frame_count}")

            object_points.append(np.array(obj_pts, dtype=np.float32))
            image_points.append(np.array(img_pts, dtype=np.float32))

            capture_frame = False

        cv2.putText(color_image, f"Captured Frames: {captured_frame_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("AprilTag Detection", color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            capture_frame = True
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    if len(image_points) > 5:
        print("Performing calibration...")

        dist_coeffs = np.zeros((5, 1))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, gray_image.shape[::-1], camera_matrix, dist_coeffs,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        np.savez("camera_calibration_new.npz", 
         mtx=mtx, dist=dist, 
         rvecs=rvecs, tvecs=tvecs, 
         object_points=np.array(object_points, dtype=object), 
         image_points=np.array(image_points, dtype=object))
        print("Calibration complete. Parameters saved to 'camera_calibration.npz'.")
    else:
        print("Not enough valid data for calibration.")
