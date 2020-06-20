"""
This file defines the methods associated with loading in the data
"""

import torchvision.datasets as torchdata
from torch.utils.data import DataLoader
import torchvision.transforms as trans

def get_data_loader(image_folder, annotation_file, batch_size=32, transforms=None):
    """
    Get the torch data loader for the COCO dataset
    :param image_folder: The folder containing the COCO images
    :param annotation_file: The .json file for the image annotations
    :param batch_size: The batch size
    :param transforms: The transformations the data will take, expected in composition or single transform
    :return: A COCO captioning data loader
    """
    # Build the transformations if none
    if transforms is None:
        transforms = trans.Compose([
            trans.Resize((512, 512)),
            trans.ToTensor()
        ])

    # Build the torch dataset
    coco_dataset = torchdata.CocoCaptions(image_folder, annotation_file, transform=transforms)
    # Return a dataloader object over this dataset
    return DataLoader(coco_dataset, batch_size=batch_size)