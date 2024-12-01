import cv2
import numpy as np
import argparse
import os

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

def main(use_cameras=True, left_images_dir='left', right_images_dir='right'):
    if use_cameras:
        # Initialize stereo cameras
        cap_left = cv2.VideoCapture(0)  # Index may vary based on your setup
        cap_right = cv2.VideoCapture(1)  # Index may vary based on your setup

        if not cap_left.isOpened() or not cap_right.isOpened():
            print("Error: Could not open cameras.")
            exit()
    else:
        # List image files
        left_images = sorted([os.path.join(left_images_dir, img) for img in os.listdir(left_images_dir)])
        right_images = sorted([os.path.join(right_images_dir, img) for img in os.listdir(right_images_dir)])

        if len(left_images) != len(right_images):
            print("Error: Number of left and right images do not match.")
            exit()

        img_index = 0

    # Create trackbar for disparity adjustment
    cv2.namedWindow('Disparity')
    cv2.createTrackbar('Num Disparities', 'Disparity', 16, 256, nothing)

    while True:
        if use_cameras:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            
            if not ret_left or not ret_right:
                print("Error: Unable to capture frames.")
                break
        else:
            if img_index >= len(left_images):
                break  # End of image sequence
            
            frame_left = cv2.imread(left_images[img_index])
            frame_right = cv2.imread(right_images[img_index])
            img_index += 1

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
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    # Release resources
    if use_cameras:
        cap_left.release()
        cap_right.release()
    cv2.destroyAllWindows()

def nothing(x):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Depth Estimation")
    parser.add_argument("--images", action="store_true", help="Use images from 'left' and 'right' directories instead of cameras")
    args = parser.parse_args()

    use_cameras = not args.images
    main(use_cameras=use_cameras)
