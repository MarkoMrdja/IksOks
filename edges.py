import itertools
import cv2
import numpy as np

cap = cv2.VideoCapture('xo2c.avi')
while 1:
    ret, frame = cap.read()
    if not ret:
        break

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayImage, 100, 150, apertureSize=3)

    cv2.imshow('Edges Video', edges)

        # Check for the 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()