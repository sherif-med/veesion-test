import os, glob, json
from src.models_setup import get_model_path
from src import VIDEOS_DIR, MEDIA_PIPE_SKELETONS_DIR
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision.io import read_video
import tqdm

def extract_skeleton_from_frames(frames: list[np.ndarray], detector)->list[dict]:
    """
    Extract the skeleton from a list of frames using openmp
    """
    # Pass each frame to the detector
    frames_detection_result = [
        detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.numpy()))
        for frame in frames
    ]
    # Keep only the pose landmarks
    frames_landmarks = [result.pose_landmarks for result in frames_detection_result]
    # Convert the landmarks to a dictionary
    frames_poses_dict = [frame_landmarks_to_dict(landmarks) for landmarks in frames_landmarks]
    return frames_poses_dict

def frame_landmarks_to_dict(landmarks):
    """
    Function to convert the landmarks to a dictionary
    Where the key is the person id and the value is a list of landmarks as dicts
    """
    persons_dict = {}
    for person_id, person_landmarks in enumerate(landmarks):
        persons_dict[f"p_{person_id}"] = [vars(l) for l in person_landmarks]
    return persons_dict

def main():
    # Setup openmp detector
    base_options = python.BaseOptions(model_asset_path=get_model_path())
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,)
    detector = vision.PoseLandmarker.create_from_options(options)

    os.makedirs(MEDIA_PIPE_SKELETONS_DIR, exist_ok=True)

    videos_list = glob.glob(f"{VIDEOS_DIR}/*.mp4")
    for video_path in tqdm.tqdm(videos_list, desc="Extracting skeletons from videos"):
        frames, _, _ = read_video(video_path)
        frames = torch.unbind(frames, dim=0)
        skeleton = extract_skeleton_from_frames(frames, detector)
        output_skeleton_path = MEDIA_PIPE_SKELETONS_DIR / os.path.basename(video_path)
        output_skeleton_path = output_skeleton_path.with_suffix(".json")
        with open(output_skeleton_path, "w") as f:
            json.dump(skeleton, f)

if __name__ == "__main__":
    main()