""" 
This module will: define convolutional layers, residual blocks, flatten, and dense layer
- Input: board tensors
- Output: 5000 raw logits 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ResNet - deep learning architecture, layers learn residual functions wrt inputs"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessPolicyNet(nn.Module):
    def __init__(self, in_channels=13, num_planes=73, num_squares=64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(128)
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )
        
        # flatten
        self.flatten = nn.Flatten()
        
        # fully connected layer to output policy
        self.fc = nn.Linear(128 * 8 * 8, num_squares * num_planes)
        
    def forward(self, x):        
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x