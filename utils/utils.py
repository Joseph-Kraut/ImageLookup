"""
This file defines a series of utils for training, testing, and debugging
"""
import torch
from torch.utils.tensorboard import SummaryWriter

def write_graph(network, dummy_input):
    """
    Writes the graph given by <network> to tensorboard for inspection
    :param network: The network we want to put on tensorboard
    :return: None
    """
    # Build the torch writer
    writer = SummaryWriter("../logs/graphs")
    # Write to tensorboard
    writer.add_graph(network, input_to_model=dummy_input, verbose=False)
    writer.flush()
    writer.close()

