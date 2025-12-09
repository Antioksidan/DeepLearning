import torch
import torch.nn as nn
import math

class ResidualDenseBlock(nn.Module):
    """
    The internal Dense Block used inside the RRDB. Based on ESRGAN github code.
    """
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, nf: number of filters
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB).
    """
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # Residual scaling (paper suggests scaling by 0.2)
        return out * 0.2 + x

class Generator_RRDB(nn.Module):
    def __init__(self, in_channels=3, num_res_blocks=23, scale_factor=4): 
        # ESRGAN typically uses 23 blocks (vs 16 in SRGAN)
        super(Generator_RRDB, self).__init__()

        self.initial = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # RRDB blocks
        self.res_blocks = nn.Sequential(
            *[RRDB(nf=64, gc=32) for _ in range(num_res_blocks)]
        )

        self.conv_body = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        upsample_blocks = []
        n_upsample = int(math.log2(scale_factor))
        for _ in range(n_upsample):
            upsample_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
            upsample_blocks.append(nn.Conv2d(64, 64, 3, 1, 1))
            upsample_blocks.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.upsample = nn.Sequential(*upsample_blocks)

        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1),
            # No Sigmoid/Tanh here in many ESRGAN impls to allow full range, 
            # but since our data is [0,1], we keep Sigmoid.
            nn.Sigmoid()
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.res_blocks(initial)
        x = self.conv_body(x) + initial # Global skip connection
        x = self.upsample(x)
        x = self.final(x)
        return x


