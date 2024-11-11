import cv2
import numpy as np
import pickle
from cv2 import aruco

# Load the calibration parameters from pickle files
with open('location_of_this_folder/cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)

with open('location_of_this_folder/dist.pkl', 'rb') as f:
    dist_coeffs = pickle.load(f)

# Dictionary of available ArUco dictionaries
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

# Specify the type of ArUco dictionary to use
aruco_type = "DICT_5X5_100"
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

# Create the detector parameters
arucoParams = cv2.aruco.DetectorParameters()

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the frame height

marker_size = 18  # in mm

def poseEstimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the image
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            # Estimate pose of each marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, matrix_coefficients, distortion_coefficients)

            # Calculate the distance to the marker
            distance = np.linalg.norm(tvec) / 10
            print(f"Marker ID: {ids[i][0]}, Distance: {distance:.2f} cm")

            # Draw the detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw the pose axes
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, marker_size * 0.5)

    return frame

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    output = poseEstimation(frame, ARUCO_DICT[aruco_type], camera_matrix, dist_coeffs)

    cv2.imshow('Estimated Pose and Distance', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
