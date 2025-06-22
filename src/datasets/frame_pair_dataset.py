import torchvision.transforms.v2 as T
from torchvision.io import read_video
from torch.utils.data import Dataset
import os
import torch
import glob

# -- 1. Data Augmentation Pipeline --
simclr_aug = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.5, 0.5, 0.5, 0.2),
    T.RandomGrayscale(p=0.2),
    T.ToTensor()
])

# -- 2. Dataset that returns two augmented views per frame --
class FramePairDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = glob.glob(os.path.join(root_dir, "*.mp4"))
        self.video_frames = {}
        for path in self.paths:
            frames, _, _ = read_video(path)
            frames = torch.unbind(frames.permute(0, 3, 1, 2), dim=0)
            self.video_frames[path] = frames

    def __len__(self):
        return sum([len(frames) for frames in self.video_frames.values()])

    def __getitem__(self, idx, video_idx=0):
        idx = idx % len(self)
        current_video_len = len(self.video_frames[self.paths[video_idx]])
        if idx <= current_video_len:
            current_frame = self.video_frames[self.paths[video_idx]][idx]
            return simclr_aug(current_frame), simclr_aug(current_frame)
        else:
            video_idx += 1
            return self.__getitem__(idx-current_video_len, video_idx)

if __name__ == "__main__":
    from src import VIDEOS_DIR
    dataset = FramePairDataset(VIDEOS_DIR)
    print(dataset[90][0].shape)
    print(dataset[-1][0].shape)
