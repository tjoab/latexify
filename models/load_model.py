import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def load_hub_model(model_name_or_path):
    processor = TrOCRProcessor.from_pretrained(model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return processor, model, device


