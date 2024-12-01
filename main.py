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
    cv2.createTrackbar('Min Disparity', 'Disparity', 0, 32, nothing)  # New trackbar for minDisparity

    # Create WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)

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
        rectified_left = frame_left
        rectified_right = frame_right
        
        # Convert to grayscale
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        
        # Get trackbar positions
        num_disparities = cv2.getTrackbarPos('Num Disparities', 'Disparity') * 16
        min_disparity = cv2.getTrackbarPos('Min Disparity', 'Disparity') - 16  # Adjust range to -16 to 16
        
        # Stereo matching
        stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=15)
        stereo.setMinDisparity(min_disparity)
        disparity = stereo.compute(gray_left, gray_right)
        
        # Apply WLS filter
        filtered_disparity = wls_filter.filter(disparity, gray_left, rectified_left, rectified_right)
        
        # Normalize the filtered disparity map for visualization
        filtered_disparity_normalized = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Concatenate input images horizontally
        image_pair = np.hstack((rectified_left, rectified_right))
        
        # Resize disparity map to match the height of the image pair
        disparity_resized = cv2.resize(filtered_disparity_normalized, (filtered_disparity_normalized.shape[1], image_pair.shape[0]))
        
        # Concatenate image pair and disparity map horizontally
        combined_output = np.hstack((image_pair, cv2.cvtColor(disparity_resized, cv2.COLOR_GRAY2BGR)))
        
        # Display the combined output
        cv2.imshow('Combined Output', combined_output)
        
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
