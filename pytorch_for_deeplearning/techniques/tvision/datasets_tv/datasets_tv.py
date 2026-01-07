import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from module_2_2_2 import helper_utils

# Set dataset directory
root_dir = './module_2_2_2/pytorch_datasets'
# Initialize the CIFAR-10 training dataset
cifar_dataset = datasets.CIFAR10(
    root=root_dir,      # Path to the directory where the data is/will be stored
    train=True,         # Specify that you want the training split of the dataset
    download=True       # Download the data if it's not found in the root directory
)


# Get the first sample (at index 0), which is a (image, label) tuple
image, label = cifar_dataset[0]
print(f"Image Type:        {type(image)}")
# Since `image` a PIL Image object, its dimensions are accessed using the .size attribute.
print(f"Image Dimensions:  {image.size}")
print(f"Label Type:        {type(label)}")

# Define a transformations pipeline
cifar_transformation = transforms.Compose([
    transforms.ToTensor(),
    # The mean and std values are standard for the CIFAR-10 dataset
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)
                        )
])
# Assign the entire transformation pipeline to the dataset's .transform attribute
cifar_dataset.transform = cifar_transformation

# Access the first item again
image, label = cifar_dataset[0]
print(f"Image Type:                   {type(image)}")
# Since the `image` is now a PyTorch Tensor, its dimensions are accessed using the .shape attribute.
print(f"Image Shape After Transform:  {image.shape}")
print(f"Label Type:                   {type(label)}")