import torch
import os
import torch.nn as nn
from torchvision import models

# Load the trained GoogLeNet model architecture
def load_trained_model(model_architecture=models.googlenet, num_classes=5):
    model = model_architecture(pretrained=False)  # Do not load pretrained weights
    num_ftrs = model.fc.in_features  # Adjust the final layer
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Save the trained model
def save_model(model, model_name="googlenet_model.pth"):
    save_path = os.path.join("/home/saish/Desktop", model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Load the trained model from GPU memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_trained_model()
model = model.to(device)

# Assuming the model is already trained in memory
save_model(model, "googlenet_model.pth")
