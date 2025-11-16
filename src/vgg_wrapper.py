import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        
        # Load a pre-trained VGG19 model
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        
        # We want the features from the 36th layer (conv3_5, before the 4th maxpool)
        # You can also use layer 9 (conv2_2) or 18 (conv3_3)
        # In the SRGAN paper, they use conv_5_4 (layer 36), let's use that.
        self.features = nn.Sequential(*list(vgg19.features.children())[:36]).eval()
        
        # Freeze the VGG model
        for param in self.features.parameters():
            param.requires_grad = False
            
        # VGG networks are trained on ImageNet, which has mean=[0.485, 0.456, 0.406]
        # and std=[0.229, 0.224, 0.225]. We need to normalize our images
        # before passing them to VGG.
        self.normalize = nn.functional.normalize

    def forward(self, x):
        # Normalize the input image to match VGG's training data
        # Note: This assumes your input images are in the range [0, 1].
        # If your images are in [-1, 1], you'll need to adjust.
        # Let's assume input is [0, 1] for simplicity.
        x_normalized = x.clone()
        x_normalized[:, 0] = (x[:, 0] - 0.485) / 0.229
        x_normalized[:, 1] = (x[:, 1] - 0.456) / 0.224
        x_normalized[:, 2] = (x[:, 2] - 0.406) / 0.225
        
        return self.features(x_normalized)