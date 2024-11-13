"""
Created on October 09 2024

code for grasping of simple objects  using a robotic arm
Look at the documentation for more details.
"""

import cv2
import numpy as np
import socket
import pickle
import time
from motoman_nx100_control import *
from end_effector_rposc import *

# NX100 IP and port 
nx100_port = 80
nx100_ip = "192.168.100.11"
nx100_subnetMask = "255.255.255.0"
nx100_gateway = "192.168.100.100"
nx100_preferred = "172.31.100.7"

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250) #Load the aruco marker dictionary
aruco_params = cv2.aruco.DetectorParameters() # Create detector parameters

home_position = [430, 0, 160]

def move_robot_to_home():
    MOVL(home_position[0], home_position[1], home_position[2])  # Use MOVL to move to home position
    # MOVL_orient()
    # Wait for the movement to complete

# Loading the camera matrix and distortion coefficients from pickle files
with open('D:/GMISHRA/7th_Sem/Final_Year_Project/NX100_ethernet-main/CameraCalibration/cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)

with open('D:/GMISHRA/7th_Sem/Final_Year_Project/NX100_ethernet-main/CameraCalibration/dist.pkl', 'rb') as f:
    dist_coeffs = pickle.load(f)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Open the camera

X=0
Y=0
Z=0
transformation_matrix=np.array([[0, -1, 0, X+50],  
                                [-1, 0, 0, Y],
                                [0, 0, -1, Z+40],
                                [0, 0, 0, 1]])

marker_size = 19  # size of the ArUco marker in milimeters

# Function to detect the ArUco marker and estimate pose
def detect_marker_and_distance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if ids is not None:
        # Estimate pose of the marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        pos_wrt_camera=np.array([tvecs[0][0][0], tvecs[0][0][1], tvecs[0][0][2], 1])
        pos_wrt_robot = np.matmul(transformation_matrix, pos_wrt_camera)

        marker_orient=np.array([rvecs[0][0][0], rvecs[0][0][1], rvecs[0][0][2], 1])
        
        # Extracting the x,y,z coordinates of the robot
        x_robot = pos_wrt_robot[0]
        y_robot = pos_wrt_robot[1]
        z_robot = pos_wrt_robot[2]

        return x_robot, y_robot, z_robot, pos_wrt_robot, corners, ids, tvecs  # Return distance, translation vector, and rotation vector
    return None, None, None, None, None, None, None

# def change_gripper_orient():
#     # Change the orientation of the end effector (gripper) to match the orientation of the
#     if (marker_orient[0]=180)

# Function to change end effector orientation
# def change_gripper_orientation(rvecs):
#     # Convert rotation vector to rotation matrix
#     rotation_matrix, _ = cv2.Rodrigues(rvecs[0][0])
    
#     # Extract orientation angles (e.g., pitch, roll, yaw) from the rotation matrix
#     # These angles will depend on your robotic arm's expected input format for orientation
    
#     pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
#     roll = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
#     yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
#     # Move end effector to align with the marker's orientation
#     # MOVL_orient(pitch, roll, yaw)  # Assuming MOVL_orient can accept orientation parameters
#     wait_for_robot()

# Function to move the robot to the marker position and grab the object
def move_robot_to_marker():
    ret, frame = cap.read()  # Capture a new frame from the camera
    if ret:
        x_robot, y_robot, z_robot, _, _, _, _ = detect_marker_and_distance(frame)  # Detect the marker and update coordinates
    if x_robot is not None and y_robot is not None and z_robot is not None:
        
        MOVL(x_robot, y_robot, z_robot+200)
        wait_for_robot()  

        ret, frame = cap.read()
        if ret:
            x_robot, y_robot, z_robot, _, _, _, _ = detect_marker_and_distance(frame)

            MOVL(x_robot, y_robot, z_robot+30) # Move to the final position to pick the object
            wait_for_robot()
        
        gripper_close() # Close the gripper to grab the object
        wait_for_robot()
        
        MOVL(x_robot, y_robot, z_robot+200)  # Move up by 10 cm
        wait_for_robot()

        MOVL(x_robot, y_robot+100, z_robot+200)  # Move left by 10 cm
        wait_for_robot()

        MOVL(x_robot, y_robot+100, -230)  # Move down by 10 cm
        wait_for_robot()

        gripper_open()  # Safely drop the object
        wait_for_robot()
    else:
        print("Error: Marker position not detected.")

    move_robot_to_home() # function ends

def wait_for_robot():
    """
    Loop will run till MOVL is running

    Returns
    -------
    None.

    """
    while True:
        status = read_status()
        #print("status", status)
        #print("Process running")
        if status == 194:
            break

# Main loop to detect marker and control robot
def main():
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    gripper_open()
    global x_robot, y_robot, z_robot  # Make coordinates accessible

    x_eff, y_eff, z_eff = get_end_effector_coordinates(1, 0)

    # Initialize x_robot, y_robot, z_robot
    x_robot = 0
    y_robot = 0
    z_robot = 0

    global transformation_matrix
    transformation_matrix = np.array([[0, -1, 0, x_eff+65],
                                      [-1, 0, 0, y_eff],
                                      [0, 0, -1, z_eff+25],
                                      [0, 0, 0, 1]])
    
    # Print end effector coordinates
    # print(f"x_eff: {x_eff} mm, y_eff: {y_eff} mm, z_eff: {z_eff} mm")
    
    last_time = time.time()  # Record the start time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x_robot, y_robot, z_robot, pos_wrt_robot, corners, ids, tvecs = detect_marker_and_distance(frame)
        
        current_time = time.time()
        if (current_time - last_time) >= 5:  # Update every 5 seconds
            last_time = current_time  # Reset last_time for the next interval

            # Ensure tvecs is valid before printing
            if tvecs is not None:
                # Print the tvecs every 5 seconds
                print(f"tvecs (every 5 sec): {tvecs[0][0]} mm")
            
            # Update robot coordinates based on transformation
            x_robot = pos_wrt_robot[0]
            y_robot = pos_wrt_robot[1]
            z_robot = pos_wrt_robot[2]
            move_robot_to_marker()  # Call the function to move the robot to the marker's position

            print(f"x_robot: {x_robot} mm, y_robot: {y_robot} mm, z_robot: {z_robot} mm")

        # Display the frame with ArUco marker detection
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow('Frame', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    servo_on() 
    main() 
    move_robot_to_home()
    servo_off()  

