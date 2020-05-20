"""
This file defines a series of utils for training, testing, and debugging
"""
import torch
from torch.utils.tensorboard import SummaryWriter

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

