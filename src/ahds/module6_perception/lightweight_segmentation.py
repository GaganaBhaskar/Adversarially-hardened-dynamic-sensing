
"""
Compact mask-aware semantic segmentation network for AHDS.

Inputs:
    B x 4 x H x W
    RGB readout and binary spatial sampling mask.

Outputs:
    B x 4 x H x W semantic logits.

No optical ground truth, attack metadata, controller state,
or unsampled sensor values are accepted as model inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.layers(x)


class FeatureBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.layers = nn.Sequential(

            SeparableConv(
                in_channels,
                out_channels,
            ),

            SeparableConv(
                out_channels,
                out_channels,
            ),
        )

    def forward(self, x):

        return self.layers(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):

        super().__init__()

        self.project = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

        self.fuse = FeatureBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):

        x = self.project(x)

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, skip], dim=1)

        return self.fuse(x)


class LightweightSegmentationNet(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        if num_classes < 2:
            raise ValueError("At least two classes required.")

        self.num_classes = num_classes

        # A conventional stem mixes RGB and sampling-mask
        # information before depthwise separable processing.

        self.stem = nn.Sequential(

            nn.Conv2d(
                4,
                8,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(8),

            nn.ReLU(inplace=True),
        )

        self.encoder1 = FeatureBlock(8, 8)
        self.encoder2 = FeatureBlock(8, 16)
        self.encoder3 = FeatureBlock(16, 24)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = FeatureBlock(24, 32)

        self.decoder3 = DecoderBlock(32, 24, 24)
        self.decoder2 = DecoderBlock(24, 16, 16)
        self.decoder1 = DecoderBlock(16, 8, 8)

        self.head = nn.Conv2d(
            8,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):

        if x.ndim != 4 or x.shape[1] != 4:
            raise ValueError("Expected B x 4 x H x W.")

        if min(x.shape[-2:]) < 8:
            raise ValueError("Input spatial dimensions too small.")

        e1 = self.encoder1(self.stem(x))

        e2 = self.encoder2(self.pool(e1))

        e3 = self.encoder3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.decoder3(b, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        return self.head(d1)
