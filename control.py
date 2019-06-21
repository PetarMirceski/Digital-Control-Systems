import cv2
import matplotlib.pyplot as plt
import numpy as np
import serial
from drawnow import *
from simple_pid import PID

lower_red1 = np.array([0, 120, 70])  # Threshold values lower boundary of lower red levels
upper_red1 = np.array([10, 255, 255])  # Threshold values upper boundary of lower red levels
plt_counts = 0
lower_red2 = np.array([170, 120, 70])  # Threshold values lower boundary of upper red levels
upper_red2 = np.array([180, 255, 255])  # Threshold values upper boundary of upper red levels

kernel_open = np.ones((5, 5))  # Kernel for opening of the image
kernel_close = np.ones((15, 15))  # Kernel for closing of the image
list_of_x_positions = np.array([])  # Numpy array for holding the last 50 readings from the x position
list_of_errors = np.array([])  # Numpy array for holding the last 50 readings from the error
distance_points = []
real_distance = 40.5
servo_low = 0
servo_high = 50
once = 1
zero_balance = 32
measurement_list = []
control_list = []


def mouse_drawing(event, x_click, y_click, params, flags):
    if event == cv2.EVENT_LBUTTONDOWN:
        distance_points.append((x_click, y_click))


def image_processing(processing_frame):
    blurred = cv2.GaussianBlur(processing_frame, (11, 11), 0)  # Blur the image to remove noise
    img_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Transfer the image to hsv image space

    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)  # Threshold the lower red boundary
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)  # Threshold the upper red boundary
    mask = mask1 + mask2  # Combine the two masks, you can remove mask2 but not mask1

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)  # Erosion followed by a Dilation

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)  # Erosion followed by a Dilation
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)  # Erosion
    mask_close = cv2.morphologyEx(mask_close, cv2.MORPH_DILATE, kernel_open)  # Dilation
    return mask_close  # Return the final mask


def draw_contours(processing_frame, processing_countour):
    processing_countour = max(processing_countour, key=cv2.contourArea)  # Find the maximum contour
    (x_position, y_position), radius = cv2.minEnclosingCircle(processing_countour)  # Positions of the contour
    center = (int(x_position), int(y_position))  # Center of the minimum enclosing circle
    radius = int(radius)  # Radius of the minimum enclosing circle
    cv2.circle(processing_frame, center, radius, (0, 255, 0), 2)  # Drawing the center of the circle
    return x_position


def connect_to_arduino():
    try:  # Try to connect with the arduino if connection is not established run with processing mode only
        try_ard = serial.Serial('COM3', 9600)  # Establish serial communication
        try_serial_flag = True  # Flag if serial is connected
    except IOError:  # Camera mode only
        try_serial_flag = False
        try_ard = None
        print('Arduino not Connected will not transfer')
    return try_serial_flag, try_ard


def get_points():
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_drawing)

    while True:
        _, get_points_frame = cap.read()
        cv2.imshow("Frame", get_points_frame)
        key = cv2.waitKey(1)
        if len(distance_points) == 2:
            break
    cv2.destroyWindow("Frame")


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Start video feed

    serial_flag, ard = connect_to_arduino()  # Try to connect to the arduino

    get_points()  # Get the start and end points of the image

    x_end, _ = distance_points.pop()
    x_start, _ = distance_points.pop()
    pixel_distance = x_end - x_start
    pixel_width = real_distance / pixel_distance
    control_signal = (real_distance / 2)
    control_pixel_distance = control_signal / pixel_width

    pid = PID(1.2, 0.4, 1.2, setpoint=control_signal)
    while True:
        ret, frame = cap.read()  # Read image
        frame = frame[50:450, :]  # Crop image vertically
        maskFinal = image_processing(frame)  # Process the image with morphological operations
        conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours
        if len(conts) != 0:  # If there are any contours
            x = draw_contours(frame, conts)  # Draw the found contour
            measurement = (x_end - x) * pixel_width
            measurement_list.append(measurement)
            if serial_flag:  # if serial is connected
                control = pid(measurement)
                control_list.append(control)
                control = int(zero_balance - control)
                if control < 0:
                    control = 0
                elif control > 55:
                    control = 55
                control = str(control) + "\n"  # query servo position

                ard.write(bytes(control.encode('utf-8')))  # Encoding and writing to the serial port
        else:
            print('not found')
        frame[:, int(x_start + control_pixel_distance), :] = 0
        cv2.imshow('frame', frame)  # Show the image with the drawn contour

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Stop the video feed
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Destroy the window
    if serial_flag:  # If the arduino is connected close the serial communication
        ard.close()
    from kalman import Kalman

    m = 0.02
    R = 0.05
    L = 0.6
    J = 2 / 5 * m * R ** 2
    d = 0.21
    g = 10
    control_constant = - (m * g * d / L) / ((J / R) + m)
    B = np.array([[0], [control_constant]])
    process_variance = np.array([[100, 0], [0, 100]])
    measurement_variance = np.array([[0.03, 0], [0, 0.03]])
    process_noise = np.array([[1, 0], [0, 1]])
    kalman = Kalman(initial_condition=np.array([[20], [0]]), control_matrix=B, process_variance=process_variance,
                    process_noise=process_noise,
                    measurement_variance=measurement_variance)
    for measurement, control in zip(measurement_list, control_list):
        kalman.make_prediction_and_update(control, measurement)
    plt.plot(control_list)
    plt.plot(measurement_list)
