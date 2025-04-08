from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from typing import Tuple

def load_hub_model(model_name_or_path: str) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel, str]:
    """
    Loads a TrOCR model and processor from the Hugging Face, either via model name or a local path

    Parameters
    ----------
    model_name_or_path : str
        Either a model repo for Huggingface (e.g. 'user/model') or a path to where the model was saved
    
    Returns
    -------
    Tuple containing:
        - TrOCRProcessor: The associated processor.
        - VisionEncoderDecoderModel: The loaded model.
        - str: The device the model is moved to ("cuda" or "cpu").
    """
    processor = TrOCRProcessor.from_pretrained(model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return processor, model, device


