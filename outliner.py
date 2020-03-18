import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    lower_blue = np.array([38,86,0])
    upper_blue = np.array([121,255,255])
    mask = cv2.inRange(hsv , lower_blue , upper_blue)
    cv2.imshow("Frame" , frame)
    cv2.imshow("Mask" , mask)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()