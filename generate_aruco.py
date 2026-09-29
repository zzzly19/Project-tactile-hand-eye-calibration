import cv2

aruco = cv2.aruco
aruco_dict=aruco.Dictionary_get(aruco.DICT_5X5_1000)

marker_id=0
marker_size_px=1000
img=aruco.drawMarker(aruco_dict,marker_id,marker_size_px)

cv2.imwrite("aruco5x5_1000_id0.png",img)
print("saved")
