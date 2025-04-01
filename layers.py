import torch
from torch import nn
import torchvision
import math

class ClipLayer(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self):
        super(ClipLayer, self).__init__()
    def forward(self, input):
        return torch.clamp_(input, 0.0, 1.0)

class LinhLayer(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self):
        super(ClipLayer, self).__init__()
    def forward(self, input):
        return torch.clamp_(input, -1.0, 1.0)

class ActivLayer(nn.Module):
    def __init__(self, activation="lrelu"):
        super(ActivLayer, self).__init__()

        if activation is not None:
            activation = activation.lower()

        self.operation = None
        if activation == 'prelu':
            self.operation = (nn.PReLU())
        elif activation == 'relu':
            self.operation = (nn.ReLU(True))
        elif activation == 'lrelu':
            self.operation = (nn.LeakyReLU(inplace=True))
        #elif activation == 'prelu_ch':
        #    self.operation = (nn.PReLU())
        elif activation == 'tanh':
            self.operation = (nn.Tanh())
        elif activation == 'linh':
            self.operation = LinhLayer()
        elif activation == 'clip':
            self.operation = ClipLayer()
        else:
            self.operation = nn.Identity()

    def forward(self, input):
        return self.operation(input)

class ConvLayer(nn.Module):
    """
    A convolutional block, comprising convolutional, BN, activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, groups = 1, activation="lrelu", batch_norm=False):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channe;s
        :param kernel_size: kernel size
        :param stride: stride
        :param batch_norm: include a BN layer?
        :param activation: Type of activation; None if none
        """
        super(ConvLayer, self).__init__()

        # A container that will hold the layers in this convolutional block
        layers = list()

        # A convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=groups, bias=(not batch_norm)))

        # A batch normalization (BN) layer, if wanted
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if(activation):
            layers.append(ActivLayer(activation))

        # Put together the convolutional block as a sequence of the layers in this container
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv_block(input)  # (N, out_channels, w, h)

class ResLayer(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, batch_norm = False, activation='lrelu'):
        super(ResLayer, self).__init__()
        # The first convolutional block
        self.conv_block1 = ConvLayer(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=batch_norm, activation=activation) #prelu uses too much memory

        # The second convolutional block
        self.conv_block2 = ConvLayer(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=batch_norm, activation=None)

    def forward(self, input):
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output += residual  # (N, n_channels, w, h)

        return output


class ShufConvLayer2(nn.Module):

    def __init__(self,  in_channels, out_channels, kernel_size=3, scaling_factor=2, groups = 1, activation=None):
        super().__init__()
        self.conv = ConvLayer(
          in_channels, 
          out_channels * (scaling_factor ** 2), 
          kernel_size, 
          1, 
          groups, 
          activation,
        )
        self.pixel_shuffle = nn.PixelShuffle(
          upscale_factor=scaling_factor
        )
        self.prelu = nn.PReLU()

    def forward(self, input):
        output = self.conv(input) 
        output = self.pixel_shuffle(output)
        return self.prelu(output)


class ShufConvLayer(nn.Module):

    def __init__(self,  in_channels, out_channels, kernel_size=3, scaling_factor=2, groups = 1, activation="clip"):

        super(ShufConvLayer, self).__init__()
        self.conv = ConvLayer(in_channels, out_channels * (scaling_factor ** 2), kernel_size, 1, groups, activation)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)

    def forward(self, input):
        output = self.conv(input) 
        output = self.pixel_shuffle(output) 
        return output


class SqrtLoss(torch.nn.Module):
    def __init__(self):
        super(SqrtLoss, self).__init__()

    def forward(self, x, y):
       return torch.sqrt((x - y).abs() + 1e-6).mean()

