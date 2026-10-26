"""
EEGNet: Compact convolutional neural network for EEG-based BCIs.
Based on: Lawhern et al. (2018) J. Neural Eng.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet model for EEG classification.
    Uses depthwise and separable convolutions.
    """
    
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 n_temporal_filters: int = 8,
                 depth_multiplier: int = 2,
                 kernel_length: int = 64,
                 pool_length: int = 8,
                 dropout_rate: float = 0.5):
        """
        Initialize EEGNet.
        
        Args:
            n_channels: Number of EEG channels
            n_times: Number of time points
            n_classes: Number of output classes
            n_temporal_filters: Number of temporal filters (F1)
            depth_multiplier: Depth multiplier for depthwise conv (D)
            kernel_length: Length of temporal convolution kernel
            pool_length: Length of pooling window
            dropout_rate: Dropout rate
        """
        super(EEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        
        F1 = n_temporal_filters
        F2 = F1 * depth_multiplier
        D = depth_multiplier
        
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise convolution (spatial filtering)
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool_length))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 3: Separable convolution
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool_length))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Calculate output size after convolutions and pooling
        self.feature_size = self._get_feature_size(n_channels, n_times)
        
        # Classification layer
        self.fc = nn.Linear(self.feature_size, n_classes)
    
    def _get_feature_size(self, n_channels, n_times):
        """Calculate the size of flattened features."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool2(x)
            return x.numel()
    
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened activations before the classifier, shape (batch, feature_size)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = self.dropout1(self.pool1(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn3(self.conv4(self.conv3(x))))))
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits of shape (batch, n_classes) for input (batch, n_channels, n_times)."""
        return self.fc(self.features(x))


class EEGNetV2(nn.Module):
    """
    EEGNet with some modifications for better performance.
    """
    
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 dropout_rate: float = 0.5):
        """
        Initialize EEGNet V2.
        
        Args:
            n_channels: Number of EEG channels
            n_times: Number of time points
            n_classes: Number of output classes
            dropout_rate: Dropout rate
        """
        super(EEGNetV2, self).__init__()
        
        F1 = 16
        F2 = 32
        
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * 2, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * 2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 3: Separable convolution
        self.conv3 = nn.Conv2d(F1 * 2, F2, (1, 16), padding='same', groups=F1 * 2, bias=False)
        self.conv4 = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(n_channels, n_times)
        
        # Classification
        self.fc = nn.Linear(self.feature_size, n_classes)
    
    def _get_feature_size(self, n_channels, n_times):
        """Calculate the size of flattened features."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool2(x)
            return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, n_channels, n_times)
            
        Returns:
            Logits of shape (batch, n_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

