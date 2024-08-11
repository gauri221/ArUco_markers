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

def aruco_display(corners, ids, rejected, image):
    """
    Display detected ArUco markers on the image.

    Parameters:
    - corners: Detected marker corners
    - ids: Marker IDs
    - rejected: Rejected marker candidates
    - image: Image to draw markers on
    """
    if len(corners) > 0:
        ids = ids.flatten()  # Flatten the ID array
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))  # Reshape the corners array
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Convert each corner coordinate to integer
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Draw the marker borders
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # Calculate and draw the center of the marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            # Draw the marker ID
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))

    return image

# Specify the type of ArUco dictionary to use
aruco_type = "DICT_5X5_100"
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

# Create the detector parameters
arucoParams = cv2.aruco.DetectorParameters()  # Updated to correct parameter creation

# Create the ArUco detector
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# Open a connection to the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the frame height

while cap.isOpened():
    ret, img = cap.read()  # Read a frame from the camera
    if not ret:
        print("Failed to grab frame")
        break

    h, w, _ = img.shape  # Get the image dimensions
    width = 1000  # Set the desired width
    height = int(width * (h / w))  # Calculate the height while maintaining the aspect ratio
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)  # Resize the image

    # Detect ArUco markers in the image
    corners, ids, rejected = arucoDetector.detectMarkers(img)
    
    # Draw the detected markers on the image
    detected_markers = aruco_display(corners, ids, rejected, img)

    # Display the image with detected markers
    cv2.imshow("Image", detected_markers)
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press
    if key == ord("q"):  # If 'q' is pressed, exit the loop
        break

cv2.destroyAllWindows()  # Close all OpenCV windows
cap.release()  # Release the camera
