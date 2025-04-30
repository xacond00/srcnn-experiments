# Implementation of https://openaccess.thecvf.com/content_CVPRW_2019/papers/WiCV/Hu_RUNet_A_Robust_UNet_Architecture_for_Image_Super-Resolution_CVPRW_2019_paper.pdf
# Trained model available at: https://drive.google.com/file/d/1tpOeYo_IzkXWSI9bAcSNOqRaBURMTLuF/view?usp=drive_link


import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super(DownBlock, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)

        if residual and in_channels != out_channels:
        # if residual and in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        identity = self.res_conv(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu2(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upscale_factor=2):
        super(UpBlock, self).__init__()
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor) if upscale_factor > 1 else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels // (upscale_factor ** 2) + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels// (upscale_factor ** 2)+skip_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, skip):
        x = self.pixel_shuffle(x)

        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)  # skip connection
        x = self.bn(x)  
        x = self.relu(self.conv1(x))  
        x = self.relu(self.relu(self.conv2(x)))
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))


class RUNet(nn.Module):
    def __init__(self, input_size=256, upscale_factor=2):
        super(RUNet, self).__init__()
        self.upscale_factor = upscale_factor

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        self.pool = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            DownBlock(64, 64),
            DownBlock(64, 64),
            DownBlock(64, 64),
            DownBlock(64, 128, residual=True)
        )
        self.down3 = nn.Sequential(
            DownBlock(128, 128),
            DownBlock(128, 128),
            DownBlock(128, 128),
            DownBlock(128, 256, residual=True)
        )
        self.down4 = nn.Sequential(
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 512, residual=True)
        )
        self.down5 = nn.Sequential(
            DownBlock(512, 512),
            DownBlock(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            Bottleneck(512, 1024),
            Bottleneck(1024, 512)
        )

        self.up1 = UpBlock(512, 512, 512, upscale_factor=2)
        self.up2 = UpBlock(512, 512, 384, upscale_factor=2)
        self.up3 = UpBlock(384, 256, 256, upscale_factor=2)
        self.up4 = UpBlock(256, 128, 96, upscale_factor=2)

        self.final_up = nn.PixelShuffle(self.upscale_factor)
        self.final_conv = nn.Sequential(
            nn.Conv2d(96 // (self.upscale_factor ** 2) + 64, 99, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(99, 99, kernel_size=3, padding=1),
            nn.ReLU(inplace=False)
        )

        self.output_layer = nn.Conv2d(99, 3, kernel_size=1)

    def forward(self, x):
        down_1 = self.input_conv(x)

        x = self.pool(down_1)
        down_2 = self.down2(x)

        x = self.pool(down_2)
        down_3 = self.down3(x)

        x = self.pool(down_3)
        down_4 = self.down4(x)

        x = self.pool(down_4)
        down_5 = self.down5(x)

        x = self.bottleneck(down_5)

        x = self.up1(x, down_5)
        x = self.up2(x, down_4)
        x = self.up3(x, down_3)
        x = self.up4(x, down_2)
        # print("Upscaling Done")

        x = self.final_up(x)
        # print("final_up Done")
        # interpolate to match dimensions (not in the original paper)
        if x.shape[2:] != down_1.shape[2:]:
            down_1 = F.interpolate(down_1, size=x.shape[2:], mode="nearest")

        x = torch.cat([x, down_1], dim=1)
        x = self.final_conv(x)
        # print("final_conv Done")

        out = self.output_layer(x)
        # print("out done")
        return out
