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

def decoded_loss(decoder, encoded_input, expected_output, norm_order=2):
    """
    This function is overloaded to perform the same loss computation for both identity and translation losses
    These losses are defined as
        Identity: || g_1(f_1(x)) - x ||_p
        Translation: || g_2(f_1(x)) - y ||_p
    :param decoder: The decoder function
    :param encoded_input: The input after being mapped to latent space
    :param expected_output: The expected output, either the original input (for identity loss)
    or paired output for translation loss
    :param norm_order: The order of the norm (usually 1 or 2)
    :return: The resultant loss
    """
    return torch.norm(decoder(encoded_input) - expected_output, p=norm_order, dim=1)

def contrastive_distance_metric(latent_1: torch.Tensor, latent_2: torch.Tensor):
    """
    Computes the contrastive distance metric h(x) defined above
    :param latent_1: The first latent vector to compute over
    :param latent_2: The second latent vector to compute over
    :param energy: The energy parameter in the contrastive norm (optional)
    :return: The value of h(x)
    """
    # Reshape into flattened vectors in the rows
    latent_1 = latent_1.view((latent_1.size()[0], 1, -1))
    latent_2 = latent_2.view((latent_2.size()[0], -1, 1))

    multiplication = torch.bmm(latent_1, latent_2).view(-1, 1)
    norm_1 = torch.norm(latent_1, p=2, dim=2)
    norm_2 = torch.norm(latent_2, p=2, dim=1)
    return torch.exp(multiplication / (norm_1 * norm_2))

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
        if len(memory_bank_class_2) > 0:
            anchor_1_denom = sum([contrastive_distance_metric(latent_view_1, stale_latent) for stale_latent in memory_bank_class_2])
        else:
            anchor_1_denom = 1

        final_loss += contrastive_distance / anchor_1_denom

        # Sum the losses \sum_{i=1}^k h({v_1, latent_view_2})
        if len(memory_bank_class_1) > 0:
            anchor_2_denom = sum([contrastive_distance_metric(stale_latent, latent_view_2) for stale_latent in memory_bank_class_1])
        else:
            anchor_2_denom = 1

        final_loss += contrastive_distance / anchor_2_denom

        # Enqueue the given latent vectors for future use
        memory_bank_class_1.append(latent_view_1)
        memory_bank_class_2.append(latent_view_2)

        return -1 * final_loss

    return contrastive_loss

def get_full_loss(encoder_1, decoder_1, encoder_2, decoder_2, alpha, beta, gamma):
    """
    Returns a function to compute the full loss given the encoders present.
    This is defined as:
        L = alpha * L_identity + beta * L_translation + gamma * L_contrastive

    :param encoder_1: The encoder of from Input Space --> Latent Space
    :param decoder_1: The decoder from Latent Space --> Input Space
    :param encoder_2: The encoder from Output Space --> Latent Space
    :param decoder_2: The decoder from Latent Space --> Output Space
    :param alpha: The identity loss weight
    :param beta: The translation loss weight
    :param gamma: The contrastive loss weight
    :return: A function to compute the scalar weight
    """
    contrastive_loss = get_contrastive_loss()

    def full_loss(input_sample, output_sample):
        """
        Computes the full loss on the example
        :param input_sample: The sample from the Input Space
        :param output_sample: The sample from the Output Space
        :return: The scalar loss on this sample
        """
        # The loss we will return
        total_loss = 0

        # Encode both vectors
        input_encoded = encoder_1(input_sample)
        output_encoded = encoder_2(output_sample)

        # Compute identity losses
        total_loss += alpha * decoded_loss(decoder_1, input_encoded, input_sample)
        total_loss += alpha * decoded_loss(decoder_2, output_encoded, output_sample)

        # Compute the translation losses
        total_loss += beta * decoded_loss(decoder_1, output_encoded, input_sample)
        total_loss += beta * decoded_loss(decoder_2, input_encoded, output_sample)

        # Compute the contrastive loss
        total_loss += gamma * contrastive_loss(input_encoded, output_encoded).view(-1)

        return total_loss

    return full_loss