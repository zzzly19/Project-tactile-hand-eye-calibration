import numpy as np
import cv2
from digit_interface import Digit
import time

# Import the touch detection functions
from touchDetection import connect_digit_sensor, capture_average_reference, touch_detection

def main():
    # Connect to the Digit sensor
    digit = connect_digit_sensor("D20994")  # Replace with your Digit sensor's serial number

    # Capture the reference frame
    reference_frame_gray, reference_frame = capture_average_reference(digit)
    i=0
    # Main loop for touch detection
    try:
        while True:
            # Perform touch detection
            detected_frame, threshold_diff, diff = touch_detection(digit, reference_frame_gray, detection_cooldown=0.1)

            # Display the results
            if detected_frame is not None:
                print("Touch detected!")
                cv2.imshow("Detected Frame", detected_frame)
                cv2.imshow("Threshold Difference", threshold_diff)
                cv2.imshow("Difference", diff)
            else:
                cv2.imshow("Detected Frame", reference_frame)  # Show reference frame if no touch is detected

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        # Clean up
        digit.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()