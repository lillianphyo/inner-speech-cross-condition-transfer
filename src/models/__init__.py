"""The two networks of the study."""
from .deepconvnet import ShallowConvNet
from .eegnet import EEGNet

__all__ = ["EEGNet", "ShallowConvNet"]
