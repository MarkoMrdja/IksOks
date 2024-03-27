import itertools
import cv2
import numpy as np


horizontal_lines = []
vertical_lines = []
frame_counter = 0

game_state = [[-1, -1, -1], 
              [-1, -1, -1], 
              [-1, -1, -1]]

intersection_points = []
cell_height = 0
cell_width = 0
cell_positions = []

previous_frame = None

def detect_grid_lines(img):
    global horizontal_lines, vertical_lines, frame_counter

    bottom_ignore_pixels = 10
    right_ignore_pixels = 10
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=70, minLineLength=115, maxLineGap=45)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if min(y1, y2) < height - bottom_ignore_pixels and max(x1, x2) < width - right_ignore_pixels:
                if is_nearly_horizontal(x1, y1, x2, y2):
                    add_line = True
                    for h_line in horizontal_lines:
                        x1_h, y1_h, x2_h, y2_h = h_line[0]
                        if abs(y1 - y1_h) < 20 or abs(y2 - y2_h) < 20:
                            add_line = False
                            break
                    if add_line:
                        horizontal_lines.append(line)
                elif is_nearly_vertical(x1, y1, x2, y2):
                    add_line = True
                    for v_line in vertical_lines:
                        x1_v, y1_v, x2_v, y2_v = v_line[0]
                        if abs(x1 - x1_v) < 20 or abs(x2 - x2_v) < 20:
                            add_line = False
                            break
                    if add_line:
                        vertical_lines.append(line)
    
    if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            find_cell_positions()

    frame_counter += 1
    if frame_counter >= 30:
        horizontal_lines.clear()
        vertical_lines.clear()
        frame_counter = 0


def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))

    return Px, Py


def find_cell_positions():
    global intersection_points

    intersection_points = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            px, py = find_intersection(h_line, v_line)
            intersection_points.append((px, py))
    
    if len(intersection_points) > 0:
        calculate_surrounding_cell_positions(intersection_points)

    
def is_nearly_horizontal(x1, y1, x2, y2, threshold=5):
    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
    return abs(angle - 0) < threshold or abs(angle - 180) < threshold


def is_nearly_vertical(x1, y1, x2, y2, threshold=5):
    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
    return abs(angle - 90) < threshold


def draw_grid_lines_on_img(img, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_detected_lines_bottom_corner(frames, lines, draw_red=True):
    num_lines = len(lines)
    color = (0, 0, 255) if draw_red else (0, 255, 0)  # Red if draw_red is True, green otherwise
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frames, (x1, y1), (x2, y2), color, 2)

    cv2.putText(frames, f"Detected Lines: {num_lines}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def calculate_surrounding_cell_positions(intersected_points):
    global cell_positions, cell_height, cell_width
    
    cell_positions = []

    # Extract points of the middle cell
    x1, y1 = intersected_points[0]
    x2, y2 = intersected_points[1]
    x3, y3 = intersected_points[2]
    x4, y4 = intersected_points[3]

    # Calculate width and height of the middle cell
    cell_width = int(abs(x2 - x1))
    cell_height = int(abs(y3 - y1))

    # Calculate center of the middle cell
    middle_cell_center_x = (x1 + x2 + x3 + x4) / 4
    middle_cell_center_y = (y1 + y2 + y3 + y4) / 4

    # Define offsets for surrounding cells relative to the middle cell
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1), (1, 0), (1, 1)]

    cell_positions.append((int(middle_cell_center_x), int(middle_cell_center_y)))

    # Calculate position of each surrounding cell based on the middle cell
    for dx, dy in offsets:
        cell_x = middle_cell_center_x + dx * cell_width
        cell_y = middle_cell_center_y + dy * cell_height
        cell_positions.append((int(cell_x), int(cell_y)))

    cell_positions = sorted(cell_positions, key=lambda pos: (pos[1], pos[0]))



def draw_cells(image):
    global cell_positions, cell_height, cell_width

    for cell_center in cell_positions:
        x = int(cell_center[0] - cell_width / 2)
        y = int(cell_center[1] - cell_height / 2)
        top_left = (x, y)
        bottom_right = (x + cell_width, y + cell_height)
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)


def detect_signs(frame):
    global cell_positions, cell_height, cell_width
    
    for cell_center in cell_positions:
        x = int(cell_center[0] - cell_width / 2)
        y = int(cell_center[1] - cell_height / 2)
        
        # Draw rectangle around the cell in the original frame
        #cv2.rectangle(frame, (x, y), (x + cell_width, y + cell_height), (0, 255, 0), 2)
        
        # Crop the cell area from the frame
        cell_image = frame[y:y+cell_height, x:x+cell_width]
        
        # Convert the cell image to grayscale
        cell_gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to isolate signs
        _, thresh = cv2.threshold(cell_gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the cell image
        cv2.drawContours(cell_image, contours, -1, (0, 0, 255), 2)


def detect_hand(frame):
    global previous_frame

    # Define region of interest (ROI) in the top center part of the frame
    height, width = frame.shape[:2]
    roi_top = 5
    roi_bottom = height // 5  # Adjust this value as needed
    roi_left = width // 5
    roi_right = width - width // 5
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    #cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if previous_frame is None:
        previous_frame = gray_roi
        return
    else:
        frame_diff = cv2.absdiff(gray_roi, previous_frame)
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(motion_mask) > 100:
            return True
        else:
            return False




cap = cv2.VideoCapture('xo2c.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect grid lines
    detect_grid_lines(frame)

    detected_lines_canvas = np.zeros_like(frame)
    draw_detected_lines_bottom_corner(detected_lines_canvas, horizontal_lines + vertical_lines, True)

    if not detect_hand(frame) and len(cell_positions) > 0:
        detect_signs(frame)
        

    # Display the frame with drawn grid lines
    cv2.imshow('Original Video', frame)
    cv2.imshow('Detected Grid Lines', detected_lines_canvas)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
