import os
from torch.utils.data import Dataset
from data.ink import read_inkml_file, render_ink
from typing import Tuple, List
import PIL
import torch 


class ImgLatexDataset(Dataset):
    """
    A PyTorch dataset for loading images and corresponding LaTeX labels from a directory of inkml files

    Parameters
    ----------
    data_dir : str
        Path to the directory containing your .inkml files

    Attributes
    ----------
    dir : str
        Root directory of the .inkml dataset
    dir_list : list of str
        Sorted list of filenames in the dataset directory
    """

    def __init__(self, data_dir: str):
        self.dir_list = sorted(os.listdir(data_dir))
        self.dir = data_dir

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx: int) -> Tuple[PIL.Image.Image, str]:
        """
        Loads the raw image and LaTeX label given an index

        Parameters
        ----------
        idx : int
            Index of the data sample to retrieve

        Returns
        -------
        Tuple containing
            - PIL.Image.Image: The rendered ink image in RGB PIL image format
            - str: The normalized LaTeX label
        """
        img_path = os.path.join(self.dir, self.dir_list[idx])
        ink = read_inkml_file(img_path)
        img_raw = render_ink(ink).convert("RGB")

        label_raw = ink.annotations.get('normalizedLabel')

        return img_raw, label_raw


def preprocess_data(batch: List[Tuple[PIL.Image.Image, str]], processor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocesses a batch of raw (image, label) pairs ready for input into the model

    Parameters
    ----------
    batch : list of tuple of (PIL.Image.Image, str)
        A batch of raw image-label pairs
    processor : transformers.TrOCRProcessor
        A reference to the processor asociate with your model for preparing the inputs

    Returns
    -------
    tuple of torch.Tensor
        - images_processed : torch.Tensor
            Tensor of shape (batch_size, 3, 384, 384) representing processed image inputs
        - labels_processed : torch.Tensor
            Tensor of shape (batch_size, seq_len) containing tokenized LaTeX labels
    """
    images_raw, labels_raw = zip(*batch)

    # Images processed and stacked into a tensor of size (# training examples, 3, 384, 384)
    images_processed = processor.image_processor(images=images_raw, return_tensors="pt").pixel_values

    # Process LaTeX labels using pretrained tokenizer
    labels_processed = processor.tokenizer(labels_raw, padding=True, return_tensors="pt").input_ids

    return images_processed, labels_processed
