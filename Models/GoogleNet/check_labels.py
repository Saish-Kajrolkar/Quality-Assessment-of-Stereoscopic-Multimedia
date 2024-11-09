import os
import re
from torch.utils.data import Dataset
from collections import defaultdict

# Custom dataset class to handle your specific filename format
class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.bmp')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Extract the label from the filename using regex
        label_match = re.search(r'image_(\d+)_', img_name)
        label = int(label_match.group(1)) - 1  # Adjust for 0-based indexing

        # Return filename and label
        return img_name, label

# Directory where your images are stored
data_dir = '/home/saish/Desktop/TWOTBHD/Cyclopean Images'

# Create the dataset
dataset = CustomImageDataset(root_dir=data_dir)

# Dictionary to store the number of images per label
label_distribution = defaultdict(int)

# Iterate over the dataset and print labels
for img_name, label in dataset:
    print(f"Filename: {img_name}, Label: {label}")
    label_distribution[label] += 1

# Print the label distribution
print("\nLabel Distribution:")
for label, count in label_distribution.items():
    print(f"Class {label}: {count} images")
