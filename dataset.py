import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Mapping from color name to label index
COLOR_TO_INDEX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "yellow": 3,
    "magenta": 4
}

class ColorPolygonDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None, mask_transform=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.file_names = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.file_names[idx])
        label_image_path = os.path.join(self.label_dir, self.file_names[idx])

        input_image = Image.open(input_image_path).convert("RGB")
        label_image = Image.open(label_image_path).convert("L")  # grayscale mask

        if self.transform:
            input_image = self.transform(input_image)
        if self.mask_transform:
            label_image = self.mask_transform(label_image)

        return input_image, label_image.squeeze().long()
