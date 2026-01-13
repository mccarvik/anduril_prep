import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

from module import helper_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

# Define the transformation pipeline
emnist_transformation = transforms.Compose([
    # Convert grayscale image to 3 channels to match MobileNetV2's input
    transforms.Grayscale(num_output_channels=3),
    # Resize the image to 224x224, the standard input size for MobileNetV2
    transforms.Resize((224, 224)),
    # Apply the 90-degree rotation augmentation
    transforms.RandomRotation(degrees=(90, 90)),
    # Apply the vertical flip augmentation
    transforms.RandomVerticalFlip(p=1.0),
    # Convert the image to a PyTorch Tensor
    transforms.ToTensor(),
    # Normalize the tensor using ImageNet's mean and standard deviation
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

