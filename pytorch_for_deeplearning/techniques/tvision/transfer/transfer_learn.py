import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

from module import helper_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")