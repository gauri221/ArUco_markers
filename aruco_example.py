import cv2
import numpy as np

# Camera calibration parameters (you can keep this for future use if needed)
camera_matrix = np.array([[949.77780879, 0, 335.79814822],
                          [0, 952.37184593, 261.97156517],
                          [0, 0, 1]])

dist_coeffs = np.array([1.61255133e-01, -1.07629996e+00, 1.67134289e-04, -3.97862981e-03, 5.28851595e+00])

# Marker side length in meters (you can keep this for future use if needed)
marker_side_length = 0.018

# Loading ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Creating ArUco parameters
aruco_params = cv2.aruco.DetectorParameters()

# Function to undistort image (you can keep this for future use if needed)
def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

# Capture video from external webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Specify backend for Windows

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Couldn't open camera.")
else:
    print("Camera opened successfully.")

# Display loop to show camera feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame")
        break

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Check for user input to quit (press 'q' key)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
