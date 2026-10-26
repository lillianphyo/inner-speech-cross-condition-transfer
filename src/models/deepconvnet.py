"""
DeepConvNet: Deep convolutional network for EEG decoding.
Based on: Schirrmeister et al. (2017) Hum Brain Mapp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    """
    Deep Convolutional Network for EEG classification.
    """
    
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 n_filters_time: int = 25,
                 filter_time_length: int = 10,
                 n_filters_spat: int = 25,
                 pool_time_length: int = 3,
                 pool_time_stride: int = 3,
                 n_filters_2: int = 50,
                 filter_length_2: int = 10,
                 n_filters_3: int = 100,
                 filter_length_3: int = 10,
                 n_filters_4: int = 200,
                 filter_length_4: int = 10,
                 dropout_rate: float = 0.5):
        """
        Initialize DeepConvNet.
        
        Args:
            n_channels: Number of EEG channels
            n_times: Number of time points
            n_classes: Number of output classes
            n_filters_time: Number of temporal filters
            filter_time_length: Length of temporal filters
            n_filters_spat: Number of spatial filters
            pool_time_length: Length of pooling window
            pool_time_stride: Stride of pooling
            n_filters_2/3/4: Number of filters in subsequent layers
            filter_length_2/3/4: Filter lengths in subsequent layers
            dropout_rate: Dropout rate
        """
        super(DeepConvNet, self).__init__()
        
        # Auto-adjust filter sizes for small inputs
        # Each pooling reduces size by ~3x, need to ensure filters fit
        if n_times < 200:  # Preprocessed data has smaller temporal dimension
            filter_time_length = min(filter_time_length, max(3, n_times // 10))
            filter_length_2 = min(filter_length_2, max(3, filter_time_length))
            filter_length_3 = min(filter_length_3, max(3, filter_time_length))
            filter_length_4 = min(filter_length_4, max(3, filter_time_length))
            print(f"  Adjusted filter sizes for small input (n_times={n_times}): {filter_time_length}")
        
        self.n_times = n_times
        self.filter_time_length = filter_time_length
        self.filter_length_2 = filter_length_2
        self.filter_length_3 = filter_length_3
        self.filter_length_4 = filter_length_4
        
        # Block 1: Temporal + Spatial convolution
        self.conv1_time = nn.Conv2d(
            1, n_filters_time, (1, self.filter_time_length), padding='same', bias=False
        )
        self.conv1_spat = nn.Conv2d(n_filters_time, n_filters_spat, (n_channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters_spat)
        self.pool1 = nn.MaxPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 2
        self.conv2 = nn.Conv2d(
            n_filters_spat, n_filters_2, (1, self.filter_length_2), padding='same', bias=False
        )
        self.bn2 = nn.BatchNorm2d(n_filters_2)
        self.pool2 = nn.MaxPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Block 3
        self.conv3 = nn.Conv2d(
            n_filters_2, n_filters_3, (1, self.filter_length_3), padding='same', bias=False
        )
        self.bn3 = nn.BatchNorm2d(n_filters_3)
        self.pool3 = nn.MaxPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Block 4
        self.conv4 = nn.Conv2d(
            n_filters_3, n_filters_4, (1, self.filter_length_4), padding='same', bias=False
        )
        self.bn4 = nn.BatchNorm2d(n_filters_4)
        self.pool4 = nn.MaxPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(n_channels, n_times)
        
        # Classification layer
        self.fc = nn.Linear(self.feature_size, n_classes)
    
    def _get_feature_size(self, n_channels, n_times):
        """Calculate the size of flattened features."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.conv1_time(x)
            x = self.conv1_spat(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            x = self.conv4(x)
            x = self.pool4(x)
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
            x = x.unsqueeze(1)  # (batch, 1, n_channels, n_times)
        
        # Block 1
        x = self.conv1_time(x)
        x = self.conv1_spat(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ShallowConvNet(nn.Module):
    """
    Shallow Convolutional Network - simpler variant.
    """
    
    def __init__(self, n_channels: int, n_times: int, n_classes: int,
                 n_filters: int = 40,
                 filter_time_length: int = 25,
                 pool_time_length: int = 75,
                 pool_time_stride: int = 15,
                 dropout_rate: float = 0.5):
        """
        Initialize ShallowConvNet.
        
        Args:
            n_channels: Number of EEG channels
            n_times: Number of time points
            n_classes: Number of output classes
            n_filters: Number of filters
            filter_time_length: Length of temporal filters
            pool_time_length: Length of pooling window
            pool_time_stride: Stride of pooling
            dropout_rate: Dropout rate
        """
        super(ShallowConvNet, self).__init__()
        
        # Temporal convolution
        self.conv_time = nn.Conv2d(1, n_filters, (1, filter_time_length), bias=False)
        
        # Spatial convolution
        self.conv_spat = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        
        # Pooling
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(n_channels, n_times)
        
        # Classification
        self.fc = nn.Linear(self.feature_size, n_classes)
    
    def _get_feature_size(self, n_channels, n_times):
        """Calculate the size of flattened features."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.conv_time(x)
            x = self.conv_spat(x)
            x = self.pool(x)
            return x.numel()
    
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened log-power features before the classifier, shape (batch, feature_size)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.bn(self.conv_spat(self.conv_time(x)))
        x = self.dropout(torch.log(torch.clamp(self.pool(torch.square(x)), min=1e-6)))
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits of shape (batch, n_classes) for input (batch, n_channels, n_times)."""
        return self.fc(self.features(x))

