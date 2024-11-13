"""
Created on October 24 2024

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

home_position = [430, 0, 160]

def wait_for_robot():
    """
    Loop will run till MOL is running

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

def move_robot_to_home():
    MOVL(home_position[0], home_position[1], home_position[2])  # Use MOVL to move to home position
    # MOVL_orient()
    # time.sleep(10)  # Wait for the movement to complete
    wait_for_robot()
    
servo_on() 
move_robot_to_home() # function ends
servo_off()
