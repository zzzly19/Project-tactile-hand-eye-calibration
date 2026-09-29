# to verify the accuracy the camera intrinsic parameter calibration
import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

def resize_image(image, scale=0.7):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def compare_intrinsics(file_matrix, file_dist):
    print("\n[1/3] Comparing with RealSense intrinsics...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        realsense_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        realsense_dist = np.array(intrinsics.coeffs)

        print("=RealSense intrinsics obtained")

        # Comparison
        print("\n=== Camera Matrix Comparison ===")
        print(f"From file:\n{np.round(file_matrix, 2)}")
        print(f"\nFrom RealSense:\n{np.round(realsense_matrix, 2)}")

        print("\n=== Distortion Coefficients Comparison ===")
        print(f"From file: {np.round(file_dist, 5)}")
        print(f"From RealSense: {np.round(realsense_dist, 5)}")

        diff_matrix = np.abs(file_matrix - realsense_matrix)
        diff_dist = np.abs(file_dist - realsense_dist)

        print("\n=== Absolute Differences ===")
        print(f"Matrix diff:\n{diff_matrix}")
        print(f"Distortion diff:\n{diff_dist[:5]}")

        diff_dist = np.abs(file_dist - realsense_dist).flatten()

        print(f"Distortion diff:\n{diff_dist[:5]}")

        # Visual comparison of distortion
        coeff_names = ['k1', 'k2', 'p1', 'p2', 'k3']
        num_coeffs = min(len(diff_dist), len(coeff_names))

        if num_coeffs > 0:
            plt.figure(figsize=(10, 4))
            plt.title("Distortion Coefficient Differences")
            color = ['red' if x > 0.01 else 'blue' for x in diff_dist[:num_coeffs]]
            bars = plt.bar(coeff_names[:num_coeffs], diff_dist[:num_coeffs], color=color)
            plt.axhline(0.01, color='gray', linestyle='--')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
            plt.ylabel("Absolute Difference")
            plt.tight_layout()
            # plt.show()
            plt.savefig("distortion_diff_plot.png")
            print("📸 Saved plot to 'distortion_diff_plot.png'")

        else:
            print("Not enough distortion coefficients to visualize.")

    except Exception as e:
        print(f"RealSense intrinsics error: {e}")
    finally:
        pipeline.stop()

def live_undistort_display(mtx, dist):
    print("\n[2/3] Live Undistortion Preview (Press 'q' to quit)...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            undistorted_image = cv2.undistort(color_image, mtx, dist, None, new_camera_matrix)

            combined = np.hstack((
                resize_image(color_image),
                resize_image(undistorted_image)
            ))
            cv2.imshow("Original (Left) vs Undistorted (Right)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def compute_reprojection_error(mtx, dist):
    print("\n[3/3] Reprojection Error Analysis...")
    try:
        calib_data = np.load("camera_calibration.npz", allow_pickle=True)
        rvecs = calib_data["rvecs"]
        tvecs = calib_data["tvecs"]
        object_points = calib_data["object_points"]
        image_points = calib_data["image_points"]

        total_error = 0
        valid_count = 0

        for i in range(len(object_points)):
            imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints2 = imgpoints2.reshape(-1, 2)
            image_points[i] = image_points[i].reshape(-1, 2)

            if imgpoints2.shape != image_points[i].shape:
                print(f"⚠️ Shape mismatch at index {i}: Skipping")
                continue

            error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            valid_count += 1

        mean_error = total_error / valid_count if valid_count > 0 else float('nan')
        print(f"✅ Mean Reprojection Error: {mean_error:.6f} pixels")
    except Exception as e:
        print(f"❌ Error computing reprojection error: {e}")

def main():
    print("=== Camera Calibration Accuracy Tester ===")
    
    try:
        calib_data = np.load("camera_calibration.npz", allow_pickle=True)
        mtx = calib_data["mtx"]
        dist = calib_data["dist"]
        print("✅ Calibration file loaded")
    except Exception as e:
        print(f"❌ Failed to load calibration file: {e}")
        return

    compare_intrinsics(mtx, dist)
    live_undistort_display(mtx, dist)
    compute_reprojection_error(mtx, dist)

    print("\n✅ All tests complete. Exit safely.")

if __name__ == "__main__":
    main()
