"""
This file defines a series of utils for training, testing, and debugging
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

def write_graph(network, input_dims):
    """
    Writes the graph given by <network> to tensorboard for inspection
    :param network: The network we want to put on tensorboard
    :param input_dims: The dimensions this network expects as input
    :return: None
    """
    # Build the torch writer
    writer = SummaryWriter("../logs/graphs")

    # Get a dummy input, must be in this form for jit trace
    dummy_input = (torch.zeros(input_dims))

    # Write to tensorboard
    writer.add_graph(network, input_to_model=dummy_input, verbose=False)
    writer.flush()
    writer.close()

def show_img_and_cap(img, cap):
    """
    Displays an image and captions
    :param img: The image to display as a tensor
    :param cap: The list of captions to print out
    :return: None
    """
    plt.imshow(np.transpose(img.numpy(), (1,2,0)))
    plt.show()

    print("Captions:")
    for index, caption in cap:
        print(f"{index}. {caption}")




