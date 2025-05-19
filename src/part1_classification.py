# AI4EO - Part 1: Land Cover Classification

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Custom Dataset Class for Sentinel-2 Data
class LandCoverDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.imag...
        }
    ]
}
