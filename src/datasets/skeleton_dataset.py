import torch
from torch.utils.data import Dataset
import numpy as np
import os, json
from src import MEDIA_PIPE_SKELETONS_DIR

class SkeletonDataset(Dataset):
    def __init__(self, keypoints_dir, file_list=None, labels=None, max_len=100):
        """
        keypoints_dir: path to .npy files
        file_list: list of filenames (e.g., ["video_001.json", ...])
        labels: list of labels
        max_len: sequence length (truncate or pad)
        """
        self.keypoints_dir = keypoints_dir
        self.file_list = file_list if file_list else os.listdir(keypoints_dir)
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        path = os.path.join(self.keypoints_dir, filename)
        with open(path) as f:
            frames_skeleton = json.load(f)

        processed_frames_skeletons = []

        for frame_skeleton in frames_skeleton:
            if frame_skeleton:
                person_skeleton = frame_skeleton["p_0"]
                keypoints_array = []
                for keypoint in person_skeleton:
                    keypoints_array.extend([keypoint["x"], keypoint["y"]])
                processed_frames_skeletons.append(keypoints_array)

        seq = np.array(processed_frames_skeletons)

        # Pad or truncate to max_len
        T = seq.shape[0]
        if T > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = np.pad(seq, ((0, self.max_len - T), (0, 0)))


        label = self.labels[idx] if self.labels else None
        item = {"seq": torch.tensor(seq, dtype=torch.float32), "filename": filename}
        if label is not None:
            item["label"] = torch.tensor(label, dtype=torch.long)

        return item

if __name__ == "__main__":

    sample_dataset = SkeletonDataset(
        keypoints_dir=MEDIA_PIPE_SKELETONS_DIR,
        file_list=["0003.json", "1301.json"],
        labels=[0, 1],
    )

    sample_item = sample_dataset[0]

    print(sample_item["seq"].shape)
    print(sample_item["label"])