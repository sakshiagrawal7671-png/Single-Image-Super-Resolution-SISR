import cv2
import torch
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, image_paths, scale=4):
        self.image_paths = image_paths
        self.scale = scale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read HR image
        hr = cv2.imread(self.image_paths[idx])
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        # FORCE SAME SIZE (fixes batching error)
        hr = cv2.resize(hr, (256, 256))

        # Create LR image
        h, w, _ = hr.shape
        lr = cv2.resize(hr, (w // self.scale, h // self.scale))
        lr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

        # Normalize
        hr = hr / 255.0
        lr = lr / 255.0

        # Convert to tensors
        hr = torch.tensor(hr).permute(2, 0, 1).float()
        lr = torch.tensor(lr).permute(2, 0, 1).float()

        return lr, hr
