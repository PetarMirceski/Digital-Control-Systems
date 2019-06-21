import cv2
import matplotlib.pyplot as plt
import numpy as np
import serial
from simple_pid import PID
import time
import math


class Kalman:

    def __init__(self, initial_condition, control_matrix, process_variance, process_noise,
                 measurement_variance=None):
        """
        :param control_matrix: The B matrix of the model of the process x = F*x + Bu
        :param process_variance: The process certainty matrix of the process P
        :param process_noise: The process noise of the system matrix Q
        :param measurement_variance: The measurement certainty matrix for the sensor R
        :param initial_condition: The initial condition of the process X0
        """
        self.control_matrix = control_matrix  # B
        self.process_variance = process_variance  # P
        self.measurement_variance = measurement_variance  # R
        self.process_noise = process_noise  # Q
        self._last_time = time.time()  # last timed moment
        self.x = initial_condition  # the initial condition of the process
        self.h = np.eye(2, dtype=int)  # the transform matrix of the measurement

    @staticmethod
    def _time_step():
        # Static method for calculating time intervals
        # Returns the next timed moment
        test_time = time.time()
        return test_time if test_time != 0 else 1e-16

    @staticmethod
    def process_model(dt):
        # The process model for the ball beam
        # Returns the F matrix from the time model x = F*x + B*u
        process_model = np.array([[1, dt], [0, 1]])
        return process_model

    def make_prediction(self, control_signal):
        """
        Predict next state using the Kalman filter state propagation
        equations.
        :param control_signal : the size of the control signal
        Calculates the equations:
        X_= Fx+Bu
        P_ = FPF^T + Q
        """
        current_time = time.time()  # Timing for the state equation x_ = Fx+Bu
        time_step = current_time - self._last_time  # Calculating the time interval
        process_model_timed = self.process_model(time_step)  # Getting the state equation
        self.x = (np.dot(self.process_model(time_step), self.x) + np.dot(self.control_matrix,
                                                                         control_signal))  # X_= Fx+Bu

        self.process_variance = np.dot(np.dot(process_model_timed, self.process_variance),
                                       np.transpose(process_model_timed)) + self.process_noise  # P_ = FPF^T + Q

        self._last_time = current_time  # The previous last time is the current one
        self.x[0][0] = -20 if self.x[0][0] < -20 else self.x[0][0]  # Limiting the output of the kalman filter
        self.x[0][0] = 60 if self.x[0][0] > 60 else self.x[0][0]  # Limiting the output of the kalman filter
        return self.x

    def make_prediction_and_update(self, control_signal, measurement):
        """
                Predict next state using the Kalman filter state propagation
                equations and add the measurement.
                :param control_signal : The size of the control signal
                :param measurement : The measurement of the control signal
                Calculates the equations:
                 y = z - H*X_
                 K = P_H^T (HP_H^T + R)^-1
                 x = X_ + Ky
                 P = (I - KX)P_
        """
        self.make_prediction(control_signal)

        y = measurement - self.x

        kalman_constant = np.dot(np.dot(self.process_noise, np.transpose(self.h)), np.linalg.inv(
            np.dot(np.dot(self.h, self.process_variance), np.transpose(self.h)) + self.measurement_variance))

        self.x = self.x + np.dot(kalman_constant, y)
        self.process_variance = np.dot(
            np.eye(self.process_variance.shape[0]) - np.dot(kalman_constant, self.process_variance),
            self.process_variance)

        self.x[0][0] = -20 if self.x[0][0] < -20 else self.x[0][0]
        self.x[0][0] = 60 if self.x[0][0] > 60 else self.x[0][0]
        return self.x


lower_red1 = np.array([0, 120, 70])  # Threshold values lower boundary of lower red levels
upper_red1 = np.array([10, 255, 255])  # Threshold values upper boundary of lower red levels
plt_counts = 0
lower_red2 = np.array([170, 120, 70])  # Threshold values lower boundary of upper red levels
upper_red2 = np.array([180, 255, 255])  # Threshold values upper boundary of upper red levels

kernel_open = np.ones((5, 5))  # Kernel for opening of the image
kernel_close = np.ones((15, 15))  # Kernel for closing of the image
distance_points = []  # List for storing the start and end pixel position of the beam
real_distance = 40.5  # The real distance of the beam
servo_low = 0  # Constant for limiting the lower bound of the servo output
servo_high = 45  # Constant for limiting the upper bound of the servo output
zero_balance = 25  # The angle of the servo at which the beam displacement angle is 0 degrees
kalman_blind = []  # List for storing kalman measurements
measurement_blind = []  # List for measurements


def mouse_drawing(event, x_click, y_click, params, flags):
    """
    Function for defining the start and end pixel position of the beam
    It's a cv2 GUI that you click on
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        distance_points.append((x_click, y_click))


def image_processing(processing_frame):
    """
            Process the image in order to find the
            :param processing_frame : single image frame from the camera
            Returns the processed mask of the ball.
    """

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
    """
             Draw the position of the ball on the image and find the ball from the mask
             Returns the x position of the ball
             :param processing_frame : Single image frame from the camera
             :param processing_frame : Processed mask of the ball
             Returns the mask of the ball.
     """
    processing_countour = max(processing_countour, key=cv2.contourArea)  # Find the maximum contour
    (x_position, y_position), radius = cv2.minEnclosingCircle(processing_countour)  # Positions of the contour
    center = (int(x_position), int(y_position))  # Center of the minimum enclosing circle
    radius = int(radius)  # Radius of the minimum enclosing circle
    cv2.circle(processing_frame, center, radius, (0, 255, 0), 2)  # Drawing the center of the circle
    return x_position


def connect_to_arduino():
    """
        Function that tries to establish serial connection to the arduino
        The code works without a connected arduino
        Returns the serial object for communication and a flag if the arduino is connected
    """
    try:  # Try to connect with the arduino if connection is not established run with processing mode only
        try_ard = serial.Serial('COM3', 9600)  # Establish serial communication
        try_serial_flag = True  # Flag if serial is connected
    except IOError:  # Camera mode only
        try_serial_flag = False
        try_ard = None
        print('Arduino not Connected will not transfer')

    return try_serial_flag, try_ard


def get_points():
    """
    Function for defining the start and end pixel position of the beam
    It's a cv2 GUI that you click on
    """
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
    # Constants for the process model
    m = 0.02  # Mass of the ball
    R = 0.05  # Radius of the ball
    L = 0.6  # Length of the beam
    J = 2 / 5 * m * R ** 2  # Momentum of the ball
    d = 0.21  # Length of the servo arm
    g = -10  # Earth's acceleration
    control_constant = - (m * g * d / L) / ((J / R) + m)  # Defining the member of the B marix
    B = np.array([[0], [control_constant / 10]])  # The B matrix
    process_variance = np.array(
        [[100, 0], [0, 100]])  # Defining the process_variance the ball offset in the beginning is around 10 cm

    measurement_variance = np.array(
        [[0.03, 0], [0, 0.03]])  # Definining the measurement variance the camera has a error of arround 3cm
    process_noise = np.array([[1, 0.0002], [0.0002, 1]])  # Defining random process_noise
    kalman = Kalman(initial_condition=np.array([[20], [0]]), control_matrix=B, process_variance=process_variance,
                    process_noise=process_noise,
                    measurement_variance=measurement_variance)  # Initialising the kalman filter

    x_end, _ = distance_points.pop()  # Getting the end position of the beam
    x_start, _ = distance_points.pop() # Getting the start position of the beam
    pixel_distance = x_end - x_start  # Calculating the pixel distance of the beam
    pixel_width = real_distance / pixel_distance  # Getting the width of an individual pixel
    control_signal = (real_distance / 2)  # defining the control signal "The middle of the beam"
    control_pixel_distance = control_signal / pixel_width  # getting the pixel constant of the control_signal
    control_orig = zero_balance  # The angle of the servo at the starting moment
    pid = PID(1.2, 0.4, 1.2, setpoint=control_signal)  # Defininf the Pid object
    while True:
        ret, frame = cap.read()  # Read image
        frame = frame[50:450, :]  # Crop image vertically
        maskFinal = image_processing(frame)  # Process the image with morphological operations
        conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours
        if len(conts) != 0:  # If there are any contours
            x = draw_contours(frame, conts)  # Draw the found contour
            measurement = (x_end - x) * pixel_width
            measurement_blind.append(measurement)
            measurement = kalman.make_prediction_and_update(int(control_orig), measurement)[0][0]
            kalman_blind.append(measurement)
            if serial_flag:  # if serial is connected
                control = pid(measurement)
                control_orig = control
                control = int(zero_balance - control)
                if control < 0:
                    control = 0
                elif control > 55:
                    control = 55
                control = str(control) + "\n"  # query servo position
                ard.write(bytes(control.encode('utf-8')))  # Encoding and writing to the serial port
        else:
            measurement = kalman.make_prediction(int(control_orig))[0][0]
            kalman_blind.append(measurement)
            measurement_blind.append(0)
            if serial_flag:  # if serial is connected
                control = pid(measurement)
                control_orig = control
                control = int(zero_balance - control)
                if control < 0:
                    control = 0
                elif control > 55:
                    control = 55
                control = str(control) + "\n"  # query servo position
                ard.write(bytes(control.encode('utf-8')))  # Encoding and writing to the serial port
        frame[:, int(x_start + control_pixel_distance), :] = 0
        cv2.imshow('frame', frame)  # Show the image with the drawn contour

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Stop the video feed
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Destroy the window

    if serial_flag:  # If the arduino is connected close the serial communication
        ard.close()
    kalman_blind = [i for i in kalman_blind if i < 100]
    print(kalman_blind)
    print(measurement_blind)
    plt.plot(np.array(kalman_blind))
    plt.plot(np.array(measurement_blind))
    plt.legend(['y = kalman', 'y = measure'], loc='upper left')
    plt.savefig('kalman_process')
    print(kalman.process_variance)
    plt.show()
