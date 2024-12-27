import os
import glob
from torch.utils.data import Dataset
from skimage import io
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
        
        # Calculate the unique labels during initialization
        self.labels = self._extract_labels()
        self.num_classes = len(set(self.labels))

        print(f"Loaded {len(self.image_paths)} images from {self.data_path}")
        print(f"Number of classes: {self.num_classes}")

    def __len__(self):
        """Returns the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Gets a single data item."""
        img_path = self.image_paths[idx]
        image = io.imread(img_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Extract label
        label = self.labels[idx]

        return {
            'image': image,
            'label': label,
            'filename': img_path
        }

    def _extract_labels(self):
        """Extracts labels from filenames."""
        labels = []
        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            match = re.search(r'category_number_(\d+)', filename)
            if match:
                labels.append(int(match.group(1)))
            else:
                raise ValueError(f"Could not find label in filename: {filename}")
        return labels

    def get_num_classes(self):
        """Returns the number of unique classes."""
        return self.num_classes
