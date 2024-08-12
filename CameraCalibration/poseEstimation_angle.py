import cv2
import numpy as np
import pickle
from cv2 import aruco

# Load the calibration parameters from pickle files
with open('D:/GMISHRA/7th_Sem/Final_Year_Project/ArUco_codes/CameraCalibration/cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)

with open('D:/GMISHRA/7th_Sem/Final_Year_Project/ArUco_codes/CameraCalibration/dist.pkl', 'rb') as f:
    dist_coeffs = pickle.load(f)

# Define the ArUco dictionary and marker size
marker_size = 26  # in mm
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the markers in the image (no cameraMatrix or distCoeff arguments here)
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estimate pose of each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        # Check if two markers are detected
        if len(ids) == 2:
            # Calculate rotation matrices and Euler angles
            rot_matrix_1, _ = cv2.Rodrigues(rvecs[0])
            rot_matrix_2, _ = cv2.Rodrigues(rvecs[1])
            angle_object = cv2.RQDecomp3x3(rot_matrix_1)[0]
            angle_gripper = cv2.RQDecomp3x3(rot_matrix_2)[0]
            beta = np.subtract(angle_object, angle_gripper)
            print(f"beta = {beta[0]:.2f}")

        # Draw axes for each marker
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size * 0.5)

    # Display the result
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
