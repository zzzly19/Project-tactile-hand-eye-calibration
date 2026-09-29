# it has been noticed that the first 5 frames are dominated by green light and it is only until we reach the 6th frame, we get the actual rgb 
# it needs a warmup capture of around 5 6 frames before it gets to the actual rgb color

import numpy as np
import cv2
from digit_interface import Digit
import time

last_detection_time = 0
last_touch_time = 0  # Track when the last touch was detected
is_touch_detected = False  # Track if the current touch has been registered

# Connect to Digit sensor
def connect_digit_sensor(serial_number):
    digit = Digit(serial_number)
    digit.connect()
    return digit

# Capture and average multiple reference frames
def capture_average_reference(digit, num_frames=5):
    print("Please ensure no touch on the gel area before proceeding.")
    input("Press Enter when the gel area is untouched and ready to capture the reference frame...")

    frames = []
    for _ in range(num_frames):
        frame = digit.get_frame()
        if frame is not None:
            frames.append(frame)
        time.sleep(0.1)  # Small delay between captures to stabilize lighting

    if len(frames) == 0:
        print("Failed to capture reference frames.")
        exit()

    # Compute the average reference frame
    reference_frame = np.mean(frames, axis=0).astype(np.uint8)

    # Convert from BGR to RGB to maintain consistency
    reference_frame = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB)

    # Apply histogram normalization for consistency
    reference_frame = normalize_histogram(reference_frame)

    print("Reference frame captured successfully!")
    cv2.imshow("Initial Reference Frame", reference_frame)
    return cv2.cvtColor(reference_frame, cv2.COLOR_RGB2GRAY), reference_frame


# Function to normalize color histograms
def normalize_histogram(image):
    for i in range(3):  # Apply to each color channel
        image[:, :, i] = cv2.equalizeHist(image[:, :, i])
    return image


# Function to detect touch by comparing current frame with reference frame
def touch_detection(digit, reference_frame_gray, detection_cooldown=0):
    global last_detection_time, last_touch_time, is_touch_detected
    current_time = time.time()

    # # Enforce cooldown to avoid rapid triggering
    # if current_time - last_detection_time < detection_cooldown:
    #     return None, None, None

    frame = digit.get_frame()
    if frame is None:
        return None, None, None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = normalize_histogram(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    reference_frame_gray = cv2.GaussianBlur(reference_frame_gray, (5, 5), 0)

    diff = cv2.absdiff(reference_frame_gray, frame_gray)
    threshold_diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    max_diff = np.max(diff)

    touch_threshold = 30  # Sensitivity of touch detection

    # Detect if a new touch event occurs (only when max_diff exceeds threshold)
    if max_diff > touch_threshold and not is_touch_detected:
        last_detection_time = current_time  # Update the last detection timestamp
        last_touch_time = current_time  # Track the time of the last touch
        is_touch_detected = True  # Mark that a touch event has been registered
        return frame, threshold_diff, diff

    # If touch is still being held, do not trigger it again until it is released
    if max_diff <= touch_threshold and is_touch_detected:
        # If the touch has been released (max_diff is below threshold), wait 5 second before detecting again
        if current_time - last_touch_time > 5:
            is_touch_detected = False  # Allow touch detection again
            last_detection_time = current_time  # Allow new detection after 5 second of no touch
        return None, None, diff

    return None, None, diff
