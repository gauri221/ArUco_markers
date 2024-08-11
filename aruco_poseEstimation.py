import numpy as np
import cv2

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

# Create the ArUco detector
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

def poseEstimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the image
    corners, ids, rejected_img_points = arucoDetector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            marker_corners = corners[i].reshape((4, 2))

            # Object points in 3D space
            obj_points = np.array([
                [-0.01, 0.01, 0], [0.01, 0.01, 0],
                [0.01, -0.01, 0], [-0.01, -0.01, 0]
            ])

            # SolvePnP to obtain rotation and translation vectors
            success, rvec, tvec = cv2.solvePnP(
                obj_points, marker_corners, matrix_coefficients, distortion_coefficients
            )

            if success:
                # Draw the detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Draw the pose axes
                cv2.drawFrameAxes(
                    frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01
                )

    return frame

# Camera intrinsic parameters (example values)
intrinsic_camera = np.array([
    [703.28279886, 0.0, 326.95125833],
    [0.0, 705.20669566, 225.94024129],
    [0.0, 0.0, 1.0]
])

# Distortion coefficients (example values)
distortion = np.array([3.61102799e-03, 3.44479121e-01, -1.06468933e-03, 5.11085895e-03,
                       -2.10463184e+00])

# Open a connection to the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the frame height

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    output = poseEstimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)

    cv2.imshow('Estimated Pose', output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
