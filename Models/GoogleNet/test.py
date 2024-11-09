import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load the trained GoogLeNet model
def load_model(model_path, model_architecture=models.googlenet, num_classes=5):
    model = model_architecture(pretrained=False)  # Do not load pretrained weights
    num_ftrs = model.fc.in_features  # Adjust the final layer
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))  # Load the saved weights
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# Function to predict on a new image
def predict(model, image_path):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/saish/Desktop/googlenet_model.pth"
loaded_model = load_model(model_path)

# Predict on a new image
predicted_class = predict(loaded_model, "/home/saish/Desktop/FF.png")
print(f"Predicted class: {predicted_class}")
