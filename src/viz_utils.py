import math
from matplotlib import pyplot as plt
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def plot_images(images, titles=None, figsize=(12, 8)):
    """
    Plot images in dynamic subplots grid.

    Args:
        images: List of image arrays or paths
        titles: Optional list of titles
        figsize: Figure size tuple
    """
    n = len(images)
    if n == 0:
        return

    # Calculate grid size
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            # Load image if it's a path
            img = images[i]
            img = img.permute(1, 2, 0) if img.shape[0] == 3 else img
            ax.imshow(img)
            if titles:
                ax.set_title(titles[i] if i < len(titles) else f'Image {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image