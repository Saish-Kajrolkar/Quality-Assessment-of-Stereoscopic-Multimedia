import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset class to handle your specific filename format
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.bmp')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Extract the label from the filename using regex
        label_match = re.search(r'image_(\d+)_', img_name)
        label = int(label_match.group(1)) - 1  # Assuming labels are 1-based, so subtract 1 for 0-based indexing

        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data augmentation and normalization
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Directory where your images are stored
data_dir = '/home/saish/Desktop/TWOTBHD/Cyclopean Images'

# Create the dataset
dataset = CustomImageDataset(root_dir=data_dir, transform=data_transforms)

# Print the total number of images in the dataset
print(f"Total number of images in the dataset: {len(dataset)}")

# Split into train and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True)

# Freeze the feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False

# Modify the classifier to match the number of classes (5 classes)
model.classifier[6] = nn.Linear(4096, 5)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Training loop with plotting
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100):
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.item())

        print(f"Epoch {epoch+1}/{num_epochs} - Training accuracy: {epoch_acc:.4f}, Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_corrects = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())

        print(f"Epoch {epoch+1}/{num_epochs} - Validation accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

    # Plot the accuracy and loss graphs after every epoch
    plot_metrics(train_acc_history, val_acc_history, train_loss_history, val_loss_history)

    return model

def plot_metrics(train_acc, val_acc, train_loss, val_loss):
    epochs = range(1, len(train_acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Start training
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100)
