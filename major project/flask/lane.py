# lane.py
import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    
    if lines is not None and len(lines) > 0:  # Check if any lines are detected
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        
        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(image, left_fit_average)
        else:
            left_line = None

        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(image, right_fit_average)
        else:
            right_line = None

        return np.array([left_line, right_line])

    else:
        return None

def canny(image):
    if len(image.shape) == 3:  # Check if it's a color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_image = cv2.Canny(blur, 50, 150)
        return canny_image
    else:
        print("Error: The image is not a color image.")

def display_lines(image, lines, color=(0, 255, 0), thickness=5):
    line_image = np.zeros_like(image)
    if lines is not None:
        # Extract coordinates of left and right lines
        left_line = lines[0]
        right_line = lines[1]

        # Create polygon points for filling the gap between lanes
        polygon = np.array([[
            (left_line[0], left_line[1]),
            (left_line[2], left_line[3]),
            (right_line[2], right_line[3]),
            (right_line[0], right_line[1]),
        ]], dtype=np.int32)

        # Fill the polygon
        cv2.fillPoly(line_image, polygon, color)

        # Draw the left and right lines on the image
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
