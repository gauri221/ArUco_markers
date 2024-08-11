import cv2
import os

# Create the 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():

    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    k = cv2.waitKey(5) & 0xFF

    if k == 27:  # ESC key to break
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite(f'images/img{num}.png', img)
        print("image saved!")
        num += 1
    elif k == ord('q'):  # If 'q' is pressed, exit the loop
        break

    cv2.imshow('Img', img)

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()
