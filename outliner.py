import cv2
import numpy

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow("Frame" , frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()