import torch
import torch.nn as nn
from torchvision import models
import math

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, channels=64, scale=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, num_res_blocks=8, scale_factor=4):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        upsample_blocks = []
        n_upsample = int(math.log2(scale_factor))
        for _ in range(n_upsample):
            upsample_blocks.append(UpsampleBlock(64, scale=2))
        self.upsample = nn.Sequential(*upsample_blocks)

        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.res_blocks(initial)
        x = self.mid(x) + initial
        x = self.upsample(x)
        x = self.final(x)
        return x

# # -----------------------------------------------
# # 1. Residual Block (for the Generator)
# # -----------------------------------------------
# class ResidualBlock(nn.Module):
#     def __init__(self, in_features=64):
#         """
#         Implements the residual block (k3n64s1 -> BN -> PReLU -> k3n64s1 -> BN)
#         followed by an element-wise sum with the input.
#         """
#         super(ResidualBlock, self).__init__()
        
#         self.conv_block = nn.Sequential(
#             # k3n64s1
#             nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_features),
#             nn.PReLU(),
#             # k3n64s1
#             nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(in_features),
#         )

#     def forward(self, x):
#         """
#         The input 'x' is added to the output of the conv_block,
#         implementing the 'Elementwise Sum' skip connection.
#         """
#         return x + self.conv_block(x)

# # -----------------------------------------------
# # 2. Generator Network
# # -----------------------------------------------
# class Generator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16, upscale_factor=4):
#         """
#         Implements the Generator network from the diagram.
        
#         Args:
#             in_channels (int): Number of input image channels (e.g., 3 for RGB).
#             out_channels (int): Number of output image channels (e.g., 3 for RGB).
#             n_residual_blocks (int): Number of residual blocks (B in the diagram).
#             upscale_factor (int): The factor to upscale the image (must be 2 or 4).
#         """
#         super(Generator, self).__init__()
        
#         if upscale_factor not in [2, 4]:
#             raise ValueError("upscale_factor must be 2 or 4.")
            
#         # === Initial Convolutional Block ===
#         # k9n64s1
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
#             nn.PReLU()
#         )

#         # === Residual Blocks ===
#         # B residual blocks
#         res_blocks = []
#         for _ in range(n_residual_blocks):
#             res_blocks.append(ResidualBlock(in_features=64))
#         self.res_blocks = nn.Sequential(*res_blocks)

#         # === Post-Residual Block ===
#         # k3n64s1
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         # The Elementwise Sum for the long skip connection is in the forward pass.

#         # === Upsampling Blocks ===
#         # The diagram shows two blocks for 4x upscaling.
#         # Each block is k3n256s1 -> PixelShuffler x2 -> PReLU
        
#         upsampling_blocks = []
#         for _ in range(int(upscale_factor / 2)):
#             upsampling_blocks += [
#                 # k3n256s1 (Note: 256 = 64 * (upscale_factor/2)^2 = 64 * 2^2 = 64 * 4)
#                 nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
#                 # PixelShuffler x2
#                 nn.PixelShuffle(upscale_factor=2),
#                 nn.PReLU()
#             ]
#         self.upsampling = nn.Sequential(*upsampling_blocks)

#         # === Final Output Convolution ===
#         # k9n3s1
#         self.conv3 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
#         # Note: The SRGAN paper uses nn.Tanh() here, but the diagram
#         # omits it. Tanh scales output to [-1, 1].
#         # self.final_act = nn.Tanh()

#     def forward(self, x):
#         # Initial block output
#         out1 = self.conv1(x)
        
#         # Residual blocks
#         out = self.res_blocks(out1)
        
#         # Post-residual block + long skip connection (Elementwise Sum)
#         out2 = self.conv2(out)
#         out = out1 + out2
        
#         # Upsampling
#         out = self.upsampling(out)
        
#         # Final convolution
#         out = self.conv3(out)
        
#         # Apply final activation if needed (e.g., Tanh)
#         # out = self.final_act(out)
        
#         return out