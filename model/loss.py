"""
This file defines losses for the models at play in this project
Notation:
    f(x) - The encoder that maps an input down to the shared latent space
    g(x) - The decoder that maps latent space representations back to input space
    h(x) - The contrastive loss distance metric defined as:
        h(x1, x2) = exp((<x1, x2>) / (||x1|| * ||x2||) * 1/energy)
"""

import torch
from _collections import deque

def identity_loss(encoder, decoder, input, norm_order=2):
    """
    The identity loss defined as ||g(f(input)) - input||_p
    :param encoder: The encoder function
    :param decoder: The decoder function
    :param input: The input tensor
    :param norm_order: The order of the norm (usually 1 or 2)
    :return: The resultant loss
    """
    return torch.norm(decoder(encoder(input)) - input, p=norm_order)

def translation_loss(encoder, decoder, input, output, norm_order=2):
    """
    The translation loss from the input space --> latent space --> output space
    Defined as: ||g_2(f_1(x)) - y||_p

    :param encoder: The encoder from input space --> latent space
    :param decoder: The decoder from latent space --> output space
    :param input: The vector in the input space
    :param output: The vector result in the output space we expect after encoder and decoder are applied
    :param norm_order: The order of the norm to use in computation
    :return: The scalar translation loss
    """
    return torch.norm(decoder(encoder(input)) - output, p=norm_order)

def contrastive_distance_metric(latent_1, latent_2, energy=1):
    """
    Computes the contrastive distance metric h(x) defined above
    :param latent_1: The first latent vector to compute over
    :param latent_2: The second latent vector to compute over
    :param energy: The energy parameter in the contrastive norm (optional)
    :return: The value of h(x)
    """
    return torch.exp((torch.dot(latent_1, latent_2)) / (torch.norm(latent_1, p=2) * torch.norm(latent_2, p=2)) * 1 / energy)

def get_contrastive_loss(memory_bank_size=20):
    """
    Gets a function for the contrastive loss that will cache a memory bank with it
    :param memory_bank_size: The size of the memory bank of stale latents to keep
    :return: The function to call to compute contrastive loss
    """
    memory_bank_class_1 = deque(maxlen=memory_bank_size)
    memory_bank_class_2 = deque(maxlen=memory_bank_size)

    def contrastive_loss(latent_view_1, latent_view_2):
        """
        Computes the contrastive loss with both views as anchors
        :param latent_view_1: The latent vector of the first view
        :param latent_view_2: The latent vector of the second view
        :return: The scalar noise contrastive loss of the two vectors
        """
        final_loss = 0

        # The contrastive distance between our correct latent views
        contrastive_distance = contrastive_distance_metric(latent_view_1, latent_view_2)

        # Sum the losses \sum_{i=1}^k h({latent_view_1, v_2})
        anchor_1_denom = sum([contrastive_distance_metric(latent_view_1, stale_latent) for stale_latent in memory_bank_class_2])
        final_loss += contrastive_distance / anchor_1_denom

        # Sum the losses \sum_{i=1}^k h({v_1, latent_view_2})
        anchor_2_denom = sum([contrastive_distance_metric(stale_latent, latent_view_2) for stale_latent in memory_bank_class_1])
        final_loss += contrastive_distance / anchor_2_denom

        # Enqueue the given latent vectors for future use
        memory_bank_class_1.append(latent_view_1)
        memory_bank_class_2.append(latent_view_2)

        return final_loss