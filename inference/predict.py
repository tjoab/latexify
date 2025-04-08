from tools import open_PIL_image
from typing import List
from models.load_model import load_hub_model

def inference(paths: List[str], model_name: str, max_gen_length=128) -> List[str]:
    """
    Runs inference on a batch of images and returns predicted LaTeX strings

    Parameters
    ----------
    paths : list of str
        List of paths to each image
    model_name : str
        Name of Huggingface model (e.g. 'user/model') or path to local model
    max_gen_length : int, optional (default is 128)
        Maximum length of generated LaTeX output

    Returns
    -------
    list of str
        List of predicted LaTeX strings for each image
    """
    # Load all image paths as a batch
    images = [open_PIL_image(path) for path in paths]
    
    # Load model and processor
    processor, model, device = load_hub_model(model_name)

    # Preprocess the images 
    preproc_image = processor.image_processor(images=images, return_tensors="pt").pixel_values
    preproc_image = preproc_image.to(device)

    # Generate and inference
    pred_ids = model.generate(preproc_image, max_length=max_gen_length)
    latex_pred = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    return latex_pred