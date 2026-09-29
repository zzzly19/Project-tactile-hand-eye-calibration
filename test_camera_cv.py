import cv2

for i in range(6):
    cap=cv2.VideoCapture(i)
    ret,frame=cap.read()
    print(i,ret,None if frame is None else frame.shape)

    if ret:
        cv2.imwrite(f"cam{i}.png",frame)
    cap.release()
