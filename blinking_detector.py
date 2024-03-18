import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
model_path = "pose_landmarker_heavy.task"

num_poses = 5
min_pose_detection_confidence = 0
min_pose_presence_confidence = 0
min_tracking_confidence = 0


def detect_blinking(image, landmarks):
    """Detects blinking in an image based on eye landmarks.
  
    Args:
        image: The input image in RGB format.
        landmarks: The detected pose landmarks.

    Returns:
        A boolean indicating whether blinking is detected.
    """
    
    left_eye = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE]

    # Calculate vertical eye openness (approximate)
    left_eye_openness = np.linalg.norm(
        np.array([left_eye.x, left_eye.y]) - np.array([left_eye.x + left_eye.z, left_eye.y])
    )
    right_eye_openness = np.linalg.norm(
        np.array([right_eye.x, right_eye.y]) - np.array([right_eye.x + right_eye.z, right_eye.y])
    )

    # Define a threshold for blinking (adjust as needed)
    blink_threshold = 0.2  # 20% of eye openness

    # Check if both eyes are below the threshold
    is_blinking = left_eye_openness < blink_threshold and right_eye_openness < blink_threshold

    print(left_eye_openness)
    return is_blinking


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws pose landmarks on an image.

    Args:
        rgb_image: The input image in RGB format.
        detection_result: The pose detection result from MediaPipe.

    Returns:
        The image with landmarks drawn on it.
    """

    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def process_image(image_path):
    """Processes an image using MediaPipe Pose Landmarker and detects blinking for each person.

    Args:
        image_path: Path to the image file.

    Returns:
        The image with landmarks drawn on it, or None if detection fails.
    """

    image = cv2.resize(cv2.imread(image_path), (1500, 1500))

    # Convert the image to MediaPipe's Image format
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    # Create the PoseLandmarker object with appropriate options
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=True,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Process the image
        results = landmarker.detect(mp_image)

        annotated_image = None
        if results.pose_landmarks:
            # Draw landmarks on the image and display it
            annotated_image = draw_landmarks_on_image(image, results)
            cv2.imshow("Image without pos landmarks", image.copy())

            # Loop through each detected pose (person)
            for idx, pose_landmarks in enumerate(results.pose_landmarks):
                # Check for blinking for this person
                is_blinking = detect_blinking(image, pose_landmarks)
                if is_blinking:
                    text_position = (50, 50 + 50 * idx)  # Adjust position based on multiple people
                    cv2.putText(
                        annotated_image, f"Person {idx+1} Blinking!", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3
                    )

    return annotated_image

import os

for path in os.listdir('images'):
  # Process the image and display the results
  annotated_image = process_image(os.path.join('images', path))
  if annotated_image is not None:
      cv2.imshow("Image with Pose Landmarks", annotated_image)
      cv2.waitKey(1)  # Wait for keypress

cv2.destroyAllWindows()
