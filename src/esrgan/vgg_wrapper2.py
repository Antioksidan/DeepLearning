import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class VGGLoss2(nn.Module):
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
        self.vgg = nn.Sequential(*list(vgg[:17])).eval() # using less than in paper (not sure how many was used there but here less is used for faster training)
        
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.criterion = nn.L1Loss() # ESRGAN often uses L1 for VGG loss as well, or MSE.
        
        # register mean/std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std
        return self.criterion(self.vgg(sr_norm), self.vgg(hr_norm))

if __name__ == "__main__":
    # simple test
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    vgg_list = list(vgg.children())
    print("VGG19 feature layers:")
    for i, layer in enumerate(vgg_list):
        print(f"Layer {i}: {layer}")
