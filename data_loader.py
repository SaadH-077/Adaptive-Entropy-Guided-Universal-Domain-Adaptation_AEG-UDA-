import os
import glob
from torch.utils.data import Dataset, DataLoader
from skimage import io
import torch
from torchvision import transforms
import re

class Office31Dataset(Dataset):
    """
    Custom Dataset for Office31 Dataset with train/val splits.
    """

    def __init__(self, root_dir, split='train', domain='source', transform=None):
        """
        Args:
            root_dir (str): Root directory containing the dataset.
            split (str): 'train' or 'val' to load respective data.
            domain (str): 'source' or 'target' domain.
            transform (callable, optional): Optional transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.split = split
        self.domain = domain
        self.transform = transform

        # Build path for the images
        self.data_path = os.path.join(root_dir, f"{domain}_images", split)
        
        # Fetch all image file paths
        self.image_paths = glob.glob(os.path.join(self.data_path, "*.png"))  # Adjust for your image extensions
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.data_path}")
        
        print(f"Loaded {len(self.image_paths)} images from {self.data_path}")

    def __len__(self):
        """Returns the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = io.imread(img_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Extract label using regex
        filename = os.path.basename(img_path)  # Extract just the filename
        match = re.search(r'category_number_(\d+)', filename)  # Find number after "category_number_"

        if match:
            label = int(match.group(1))  # Extract the matched number
        else:
            raise ValueError(f"Could not find label in filename: {filename}")

        return {
            'image': image,
            'label': label,
            'filename': img_path
        }

# # Data transformations
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Example Usage
# if __name__ == "__main__":
#     root_dir = "usfda_office_31_DtoA"

#     # Source Train Loader
#     source_train_dataset = Office31Dataset(root_dir=root_dir, split='train', domain='source', transform=transform)
#     source_train_loader = DataLoader(source_train_dataset, batch_size=32, shuffle=True)

#     # Source Val Loader
#     source_val_dataset = Office31Dataset(root_dir=root_dir, split='val', domain='source', transform=transform)
#     source_val_loader = DataLoader(source_val_dataset, batch_size=32, shuffle=True)

#     # Target Train Loader
#     target_train_dataset = Office31Dataset(root_dir=root_dir, split='train', domain='target', transform=transform)
#     target_train_loader = DataLoader(target_train_dataset, batch_size=32, shuffle=True)

#     # Target Val Loader
#     target_val_dataset = Office31Dataset(root_dir=root_dir, split='val', domain='target', transform=transform)
#     target_val_loader = DataLoader(target_train_dataset, batch_size=32, shuffle=True)

#     # Test the DataLoader
#     for batch in source_train_loader:
#         images = batch['image']   # Shape: (32, 3, 224, 224)
#         labels = batch['label']   # Shape: (32,)
#         filenames = batch['filename']
#         print(f"Batch Image Shape: {images.shape}")
#         print(f"Labels Shape: {labels.shape}")
#         print(f"Labels: {labels}")
#         break
    
#     for batch in source_val_loader:
#         images = batch['image']   # Shape: (32, 3, 224, 224)
#         labels = batch['label']   # Shape: (32,)
#         filenames = batch['filename']
#         print(f"Batch Image Shape: {images.shape}")
#         print(f"Labels Shape: {labels.shape}")
#         print(f"Labels: {labels}")
#         break

#     for batch in target_train_loader:
#         images = batch['image']   # Shape: (32, 3, 224, 224)
#         labels = batch['label']   # Shape: (32,)
#         filenames = batch['filename']
#         print(f"Batch Image Shape: {images.shape}")
#         print(f"Labels Shape: {labels.shape}")
#         print(f"Labels: {labels}")
#         break

#     for batch in target_val_loader:
#         images = batch['image']   # Shape: (32, 3, 224, 224)
#         labels = batch['label']   # Shape: (32,)
#         filenames = batch['filename']
#         print(f"Batch Image Shape: {images.shape}")
#         print(f"Labels Shape: {labels.shape}")
#         print(f"Labels: {labels}")
#         break
    


