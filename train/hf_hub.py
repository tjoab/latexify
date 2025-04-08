from huggingface_hub import login
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def save_to_hfhub(repo_name: str, 
                model: VisionEncoderDecoderModel, 
                processor: TrOCRProcessor, 
                hf_token: str = None) -> None:
    """
    Logs in to Huggingface and pushes the given model and processor to a specified repo on the hub.

    Parameters
    ----------
    repo_name : str
        Name of the Huggingface repo to push to (e.g. 'username/model-name')
    model : VisionEncoderDecoderModel
        A refrence to the model being uploaded
    processor : TrOCRProcessor
        A refrence to the associated processor being uploaded
    hf_token : str, optional
        Hugging Face access token for seamless login. If not provided, will prompt for login
    """
    if hf_token:
        login(token=hf_token)
    else:
        login() 
        
    processor.push_to_hub(repo_name)
    model.push_to_hub(repo_name)