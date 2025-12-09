import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Layers used in this loss calculation:
        # Layer 0: Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 1: ReLU(inplace=True)
        # Layer 2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 3: ReLU(inplace=True)
        # Layer 4: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # Layer 5: Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 6: ReLU(inplace=True)
        # Layer 7: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 8: ReLU(inplace=True)
        # Layer 9: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # Layer 10: Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 11: ReLU(inplace=True)
        # Layer 12: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 13: ReLU(inplace=True)
        # Layer 14: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 15: ReLU(inplace=True)
        # Layer 16: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Layer 17: ReLU(inplace=True)
        self.vgg = nn.Sequential(*list(vgg[:18])).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.criterion = nn.MSELoss()

        # register mean/std as buffers so they move with .to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, sr, hr):
        # sr, hr in [0,1]
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std
        return self.criterion(self.vgg(sr_norm), self.vgg(hr_norm))

# class VGGFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(VGGFeatureExtractor, self).__init__()
        
#         # Load a pre-trained VGG19 model
#         vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        
#         # We want the features from the 36th layer (conv3_5, before the 4th maxpool)
#         # You can also use layer 9 (conv2_2) or 18 (conv3_3)
#         # In the SRGAN paper, they use conv_5_4 (layer 36), let's use that.
#         self.features = nn.Sequential(*list(vgg19.features.children())[:36]).eval()
        
#         # Freeze the VGG model
#         for param in self.features.parameters():
#             param.requires_grad = False
            
#         # VGG networks are trained on ImageNet, which has mean=[0.485, 0.456, 0.406]
#         # and std=[0.229, 0.224, 0.225]. We need to normalize our images
#         # before passing them to VGG.
#         self.normalize = nn.functional.normalize

#     def forward(self, x):
#         # Normalize the input image to match VGG's training data
#         # Note: This assumes your input images are in the range [0, 1].
#         # If your images are in [-1, 1], you'll need to adjust.
#         # Let's assume input is [0, 1] for simplicity.
#         x_normalized = x.clone()
#         x_normalized[:, 0] = (x[:, 0] - 0.485) / 0.229
#         x_normalized[:, 1] = (x[:, 1] - 0.456) / 0.224
#         x_normalized[:, 2] = (x[:, 2] - 0.406) / 0.225
        
#         return self.features(x_normalized)