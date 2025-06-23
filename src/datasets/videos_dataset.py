from torchvision.io import read_video
from torch.utils.data import Dataset
import os
import glob


class VideosDataset(Dataset):
    def __init__(self, root_dir, file_list=None, labels=None):
        if file_list:
            self.paths = [os.path.join(root_dir, path) for path in file_list]
            assert all([os.path.exists(path) for path in self.paths]), "Missing files in file list"
        else:
            self.paths = glob.glob(os.path.join(root_dir, "*.mp4"))
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        frames, _, _ = read_video(path)
        frames = frames.permute(0, 3, 1, 2).float()
        item = {"video": frames, "file_path": path}
        if self.labels:
            item["label"] = self.labels[idx]
        return item

if __name__ == "__main__":
    from src import VIDEOS_DIR
    dataset = VideosDataset(VIDEOS_DIR)
    print(dataset[0]["video"].shape)
