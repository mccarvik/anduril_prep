import glob
import os
import random
from collections import defaultdict

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from module_3_1_1 import helper_utils
from module_3_1_1 import training_functions