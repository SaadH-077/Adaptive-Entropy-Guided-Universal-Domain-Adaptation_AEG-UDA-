import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms
import torch


class NegativeDataset(Dataset):
    """
    Dataset class for loading negative dataset images.
    """

    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths (list): List of file paths for the dataset images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        """Returns the total number of images."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Fetch a single item from the dataset."""
        # Load image
        img_path = self.file_paths[idx]
        image = io.imread(img_path)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Extract label from filename
        label = int(os.path.basename(img_path).split("_")[1])  # Extract category_number

        # Return the data
        return {
            "image": image,
            "label": label,
            "filename": img_path,
        }
