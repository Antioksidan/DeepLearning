import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        """
        Implements the Discriminator network from the diagram.
        
        Args:
            input_shape (tuple): The shape of the input HR/SR images,
                                 e.g., (3, 96, 96) for (channels, height, width).
                                 This is needed to calculate the final dense layer size.
        """
        super(Discriminator, self).__init__()

        in_channels, in_height, in_width = input_shape

        # Helper function for a single discriminator block
        def discriminator_block(in_filters, out_filters, stride=1, bn=True):
            """
            Creates a block: Conv(k3s_) -> BN -> LeakyReLU
            """
            layers = []
            # k3n_s_
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1))
            if bn:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # === Convolutional Blocks ===
        self.conv_blocks = nn.Sequential(
            # k3n64s1
            discriminator_block(in_channels, 64, stride=1, bn=False),
            # k3n64s2
            discriminator_block(64, 64, stride=2, bn=True),
            # k3n128s1
            discriminator_block(64, 128, stride=1, bn=True),
            # k3n128s2
            discriminator_block(128, 128, stride=2, bn=True),
            # k3n256s1
            discriminator_block(128, 256, stride=1, bn=True),
            # k3n256s2
            discriminator_block(256, 256, stride=2, bn=True),
            # k3n512s1
            discriminator_block(256, 512, stride=1, bn=True),
            # k3n512s2
            discriminator_block(512, 512, stride=2, bn=True),
        )

        # === Dense Layers ===
        # Calculate the flattened feature size after the conv blocks
        # There are 4 strided convolutions (s2), so the spatial dimension is reduced by 2^4 = 16
        downsampled_height = in_height // (2**4)
        downsampled_width = in_width // (2**4)
        in_features_dense = 512 * downsampled_height * downsampled_width

        self.dense_layers = nn.Sequential(
            # Dense (1024)
            nn.Linear(in_features_dense, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            # Dense (1)
            nn.Linear(1024, 1),
            # Sigmoid
            # nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Pass through conv blocks
        out = self.conv_blocks(x)
        
        # Flatten the output for the dense layers
        # view(batch_size, -1) automatically calculates the flattened size
        out = out.view(batch_size, -1)
        
        # Pass through dense layers
        out = self.dense_layers(out)
        
        return out