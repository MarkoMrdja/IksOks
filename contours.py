import cv2
import numpy as np

cap = cv2.VideoCapture('xo2c.avi')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(grayImage, 100, 150, apertureSize=3)
    ret, thresh = cv2.threshold(grayImage, 127, 255, 0)
    im2, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(frame)

    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Original Video', frame)
    cv2.imshow('Contours Video', frame)

        # Check for the 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()