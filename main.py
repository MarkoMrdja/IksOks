import cv2
import numpy as np
from tictactoe_detection import *


displayed_horizontal_lines = []
displayed_vertical_lines = []

game_state = [[-1, -1, -1], 
              [-1, -1, -1], 
              [-1, -1, -1]]

cell_positions = []
hand_in_frame = False


cap = cv2.VideoCapture('xo1c.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    displayed_horizontal_lines, displayed_vertical_lines, cell_positions = detect_grid_lines(frame, 
                                                                                             displayed_horizontal_lines, 
                                                                                             displayed_vertical_lines, 
                                                                                             hand_in_frame, 
                                                                                             cell_positions)
    detected_lines_canvas = np.zeros_like(frame)
    hand_in_frame = detect_hand(frame)

    if not hand_in_frame and len(cell_positions) > 0:
        game_state = detect_signs(frame, game_state, cell_positions)
        
    draw_detected_lines(detected_lines_canvas, displayed_horizontal_lines + displayed_vertical_lines)
    draw_shapes(detected_lines_canvas, game_state, cell_positions)

    # Display the stacked frames
    stacked_frames = np.hstack((frame, detected_lines_canvas))
    cv2.imshow('Tic tac toe', stacked_frames)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

