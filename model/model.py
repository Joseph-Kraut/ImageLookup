"""
This file defines the models used in the PyTorch implementation of the project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
HELPERS
"""
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

        self.conv_layers = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding_amount)
                            for _ in range(depth - 1)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(depth - 1)])

        # Define the downsampling layer if necessary
        if downsample_after:
            self.strided_conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride=2, padding=1)
            self.final_bn = nn.BatchNorm2d(out_channels)

        # Build the architecture needed to complete the residual connection
        self.depth_expansion = nn.Conv2d(in_channels, out_channels, 1)
        self.expansion_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Computes the forward pass on an input
        """
        initial_input = x
        x = self.bn1(F.relu(self.conv1(x)))

        for layer_index in range(len(self.conv_layers)):
            x = self.batch_norms[layer_index](F.relu(self.conv_layers[layer_index](x)))

        # Add the residual connection
        x += self.expansion_norm(self.depth_expansion(initial_input))

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

class UpsampleBlock(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, depth=3):
        """
        Upsample by factor of 2 then apply residual block
        :param depth: The depth (number of conv layers before next upsample)
        """
        super(UpsampleBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.res_block = ResidualBlock(input_size, out_channels, out_channels, 3, depth=2, downsample_after=False)

    def forward(self, x):
        """
        Runs the forward pass on the upsample block
        """
        x = self.upsample(x)
        return self.res_block(x)

"""
IMAGE ENCODER AND DECODER
"""

class ImageEncoder(nn.Module):
    def __init__(self, input_size, latent_dim, num_blocks=4):
        """
        Initializes the network that serves as an encoder from Image Input --> Latent Space
        """
        super(ImageEncoder, self).__init__()

        # Build 4 residual blocks to downsample
        input_sizes = [input_size * 2 ** (-1 * i) for i in range(1, num_blocks)]
        self.init_block = ResidualBlock(input_size, 3, 128, 3)

        self.res_blocks = nn.ModuleList([ResidualBlock(in_size, 128, 128, 3) for in_size in input_sizes])

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

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim, input_size, num_blocks=4):

        """
        The image decoder
        """
        super(ImageDecoder, self).__init__()
        # Compute what the dimension of the first deconv input will be
        self.dim_after_downsample = int(input_size // (2 ** num_blocks))
        self.fc1 = nn.Linear(latent_dim, self.dim_after_downsample ** 2 * 128)
        self.bn1 = nn.BatchNorm1d(self.dim_after_downsample ** 2 * 128)

        # The deconvolution layers
        sizes = [int(input_size // (2 ** i)) for i in range(1, num_blocks)]
        sizes.reverse()
        self.upsample_layers = nn.ModuleList([UpsampleBlock(size, 128, 128) for size in sizes])

        self.final_upsample = UpsampleBlock(input_size, 128, 128)
        self.channel_shrink = nn.Conv2d(128, 3, 1)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))

        # Reshape correctly
        x = x.view(-1, 128, self.dim_after_downsample, self.dim_after_downsample)

        for upsample in self.upsample_layers:
            x = upsample(x)

        x = self.final_upsample(x)
        return self.channel_shrink(x)

input = torch.rand((10, 3, 256, 256))
encoder = ImageEncoder(256, 12)
decoder = ImageDecoder(12, 256)

criterion = nn.MSELoss()
optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)


"""
TEXT ENCODER AND DECODER
"""