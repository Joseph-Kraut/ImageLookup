"""
This file defines the models used in the PyTorch implementation of the project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_padding_size(dimension_size, stride, kernel_size):
    """
    Returns the padding needed to keep the output of a convolution layer the same size

    :param dimension_size: The size of the dimension we're convolving over
    :param stride: The stride we convolve at
    :param kernel_size: The size of the convolution kernel
    :return: The padding needed to maintain a "same" convolution
    """
    return (dimension_size * (stride - 1) + kernel_size - stride) // 2

class ResidualBlock(nn.Module):
    """
    The residual block to be chained together in the ResNet
    """
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride=1,
                 depth=3, downsample_after=True):
        super(ResidualBlock, self).__init__()

        self.input_size = input_size
        self.out_channels = out_channels
        self.downsample_after = downsample_after

        # Define the structure of the residual block
        padding_amount = int(get_padding_size(input_size, stride, kernel_size))

        # Define the convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding_amount)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv_layers = [nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding_amount)
                            for _ in range(depth - 1)]
        self.batch_norms = [nn.BatchNorm2d(out_channels) for _ in range(depth - 1)]

        # Define the downsampling layer if necessary
        if downsample_after:
            self.strided_conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride=2, padding=1)
            self.final_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Computes the forward pass on an input
        """
        x = self.bn1(F.relu(self.conv1(x)))

        for layer_index in range(len(self.conv_layers)):
            x = self.batch_norms[layer_index](F.relu(self.conv_layers[layer_index](x)))

        if self.downsample_after:
            x = self.final_bn(F.relu(self.strided_conv(x)))

        return x

    def get_num_flat_features(self):
        """
        Returns the number of flattened output features for downstream linear layers
        :return: The number of flat features
        """
        if self.downsample_after:
            return (self.input_size // 2) ** 2 * self.out_channels
        else:
            return self.input_size ** 2 * self.out_channels

class ImageEncoder(nn.Module):
    def __init__(self, input_size, latent_dim, num_blocks=4):
        """
        Initializes the network that serves as an encoder from Image Input --> Latent Space
        """
        super(ImageEncoder, self).__init__()

        # Build 4 residual blocks to downsample
        input_sizes = [input_size * 2 ** (-1 * i) for i in range(1, num_blocks)]
        self.init_block = ResidualBlock(input_size, 3, 128, 3)

        self.res_blocks = [ResidualBlock(in_size, 128, 128, 3) for in_size in input_sizes]

        final_input_size = int(input_size * (2 ** (-num_blocks)))
        self.fully_connected = nn.Linear(128 * (final_input_size) ** 2, latent_dim)

    def forward(self, x):
        """
        Performs the model forward pass
        :param x: The model input
        :return: The model output
        """
        x = self.init_block(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = x.view(x.size()[0], -1)

        return F.relu(self.fully_connected(x))


