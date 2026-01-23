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


# Create the training and validation DataLoaders.
train_loader, val_loader = helper_utils.create_emnist_dataloaders(
    batch_size=32,
    transform=emnist_transformation  # Apply the defined transformations
)
# Load the pre-trained MobileNetV3 model and set it to evaluation mode for inference
mobilenet_model = tv_models.mobilenet_v3_small(weights='IMAGENET1K_V1').eval()
# Load the mapping of class indices to human-readable names from the JSON file
class_names = helper_utils.load_imagenet_classes('./imagenet_class_index.json')
# Visualize the model's predictions on the validation images
helper_utils.show_predictions(mobilenet_model, val_loader, device, class_names)

# Instantiate the ResNet18 model architecture and load the selected weights
resnet18_model = tv_models.resnet18(weights='IMAGENET1K_V1')

# ### Uncomment and execute the line below if you wish print the model's architecture.
print(resnet18_model)

# Iterate over each parameter in the resnet18_model
for param in resnet18_model.parameters():
    # Set the requires_grad attribute of each parameter to False to freeze it
    param.requires_grad = False

# Access the fully connected layer (fc) of the resnet18_model
original_fc_layer = resnet18_model.fc
print("Model's Original Fully Connected Layer:")
print(original_fc_layer)

# Get the number of input features to the original fully connected layer
# This is stored in the 'in_features' attribute of the linear layer
num_features = original_fc_layer.in_features
# Define the number of output classes for the new classification task
num_classes = 5
# Create a new fully connected layer (Linear layer)
new_fc_layer = nn.Linear(in_features=num_features, out_features=num_classes)
# Replace the original fully connected layer of resnet18_model with the new_fc_layer
resnet18_model.fc = new_fc_layer
print("Model's New Fully Connected Layer:")
print(resnet18_model.fc)

# Instantiate the MobileNetV3 Small architecture and load the selected weights
mobilenet_model = tv_models.mobilenet_v3_small(weights='IMAGENET1K_V1')
# ### Uncomment and execute the line below if you wish to print the model's architecture.
print(mobilenet_model)

# Iterate through each parameter in model.features.parameters()
for feature_parameter in mobilenet_model.features.parameters():
    # Set the requires_grad attribute of each feature_parameter to False
    feature_parameter.requires_grad = False

# Access the final classification layer of the model
last_classifier_layer = mobilenet_model.classifier[-1]

print("Model's Original Output Layer:")
print(last_classifier_layer)

# Access the in_features attribute of last_classifier_layer
num_features = last_classifier_layer.in_features

# Define the number of output classes for the new classification task
num_classes = 10

# Create a new Linear layer for classification
new_classifier = nn.Linear(in_features=num_features, out_features=num_classes)

# Replace the original last classification layer with the newly created layer
mobilenet_model.classifier[-1] = new_classifier

print("Model's New Output Layer:")
print(mobilenet_model.classifier[-1])

