import cv2
import numpy as np

# TODO: Load calibration parameters (intrinsics, distortion coefficients, rectification parameters)
# For now, assuming identity matrices for rectification

# Example camera matrices and distortion coefficients (to be replaced with actual values)
left_camera_matrix = np.eye(3)
right_camera_matrix = np.eye(3)
left_dist_coeffs = np.zeros((4,1))
right_dist_coeffs = np.zeros((4,1))

# Stereo rectification parameters (assuming identity for simplicity)
R = np.eye(3)
T = np.array([[1], [0], [0]])  # Example translation vector
R1 = np.eye(3)
R2 = np.eye(3)
P1 = np.eye(3)
P2 = np.eye(3)
Q = np.eye(4)

# Initialize stereo cameras
cap_left = cv2.VideoCapture(0)  # Index may vary based on your setup
cap_right = cv2.VideoCapture(1)  # Index may vary based on your setup

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: Could not open cameras.")
    exit()

# TODO: Set camera properties if needed

def nothing(x):
    pass

# Create a window for the disparity adjustment slider
cv2.namedWindow('Disparity')
cv2.createTrackbar('Num Disparities', 'Disparity', 16, 256, nothing)

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        print("Error: Unable to capture frames.")
        break
    
    # TODO: Rectify images using calibration parameters
    # rectified_left = cv2.remap(frame_left, left_map_x, left_map_y, cv2.INTER_LINEAR)
    # rectified_right = cv2.remap(frame_right, right_map_x, right_map_y, cv2.INTER_LINEAR)
    # For now, using the raw frames
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    
    # Stereo matching
    num_disparities = cv2.getTrackbarPos('Num Disparities', 'Disparity')
    stereo = cv2.StereoBM_create(numDisparities=num_disparities*16, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right)
    
    # Normalize the disparity map for visualization
    disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Display the disparity map
    cv2.imshow('Disparity Map', disparity)
    
    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
