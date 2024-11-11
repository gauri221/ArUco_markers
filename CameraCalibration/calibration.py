import numpy as np
import cv2 as cv
import glob
import pickle
import os

# Chessboard configuration
chessboardSize = (8, 5)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= 30  # Scale for your chessboard squares size in mm

# Arrays to store object and image points
objpoints = []
imgpoints = []

# Define the folder for saving files
save_folder = 'CameraCalibration'
os.makedirs(save_folder, exist_ok=True)

# Load images
images_path = r'path_of_the_file_where_you_want_to_save_your_photos*.png'
images = glob.glob(images_path)
print(f"Found {len(images)} images.")

# Read the first image to determine frame size
img = cv.imread(images[0])
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
frameSize = gray.shape[::-1]

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners2)
    cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
    cv.imshow('Chessboard Corners', img)
    cv.waitKey(500)

cv.destroyAllWindows()

# Camera calibration to get the camera matrix and distortion coefficients
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the camera calibration result
with open(os.path.join(save_folder, "calibration.pkl"), "wb") as f:
    pickle.dump((cameraMatrix, dist), f)

# Save the intrinsic camera matrix and distortion coefficients separately
with open(os.path.join(save_folder, "cameraMatrix.pkl"), "wb") as f:
    pickle.dump(cameraMatrix, f)

with open(os.path.join(save_folder, "dist.pkl"), "wb") as f:
    pickle.dump(dist, f)

# Load an image for undistortion
img = cv.imread(images[0])
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, frameSize, 1, frameSize)

# Undistort the image
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite(os.path.join(save_folder, 'caliResult1.png'), dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, frameSize, 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
dst = dst[y:y+h, x:x+w]
cv.imwrite(os.path.join(save_folder, 'caliResult2.png'), dst)

# Reprojection error calculation
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Total error: {mean_error / len(objpoints)}")
