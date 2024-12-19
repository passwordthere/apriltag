import cv2
import apriltag
import numpy as np

# Camera intrinsic parameters (example; you should calibrate your camera to get these)
fx, fy = 800, 800  # focal length in pixels
cx, cy = 320, 240  # principal point in pixels (image center)

# Camera intrinsic matrix (3x3)
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Distortion coefficients (assuming no distortion)
dist_coeffs = np.zeros((4, 1))  # for simplicity, assuming no distortion

# Load an image with AprilTags
image = cv2.imread('april_tag_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the AprilTag detector
detector = apriltag.Detector()

# Detect tags in the image
tags = detector.detect(gray)

# Process each detected tag
for tag in tags:
    print(f"Detected tag ID: {tag.tag_id}")

    # Get the corners of the tag (in image coordinates)
    corners = tag.corners  # 4 points in (x, y) format

    # Define the 3D coordinates of the tag's corners in the world coordinate system
    # Assume a square tag with size 1 (in some arbitrary unit, e.g., meters)
    tag_size = 1.0
    object_points = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    # Image points (2D corner points of the tag detected in the image)
    image_points = np.array(corners, dtype=np.float32)

    # Use solvePnP to estimate the pose (rotation and translation)
    ret, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs)

    if ret:
        # rvec (rotation vector) and tvec (translation vector) give the pose of the tag relative to the camera
        print("Rotation Vector:\n", rvec)
        print("Translation Vector:\n", tvec)

        # Convert rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Draw the pose on the image
        axis = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1, 3)
        img_points, _ = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)

        img_points = np.int32(img_points).reshape(-1, 2)
        image = cv2.polylines(image, [np.int32(corners)], isClosed=True, color=(0, 255, 0), thickness=2)
        image = cv2.line(image, tuple(img_points[0]), tuple(img_points[1]), (255, 0, 0), 5)
        image = cv2.line(image, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 5)
        image = cv2.line(image, tuple(img_points[0]), tuple(img_points[3]), (0, 0, 255), 5)

        # Display the image with pose
        cv2.imshow('Pose Estimation', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

