from functools import partial
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from evaluate import load

from models.load_model import load_model
from train.dataset import ImgLatexDataset, preprocess_data
from tools import load_config



def test(config_path: str) -> float:
    """
    Runs an evaluation of the model computing character error rate (CER) over the test set

    Parameters
    ----------
    config_path : str
        Path to the YAML config file containing all model and data configs

    Returns
    -------
    float
        CER over the test dataset
    """
    config = load_config(config_path)
    
    # Load model and assign it to correct device (CPU v GPU)
    processor, model, device = load_model(config['model_name'])

    # Create dataloader for torch using the testing data directory
    dataset = ImgLatexDataset(config['data_dir'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=partial(preprocess_data, processor=processor))

    cer_metric = load('cer')
    progress_bar = tqdm(range(len(dataloader)))

    predicted_labels = []
    reference_labels = []

    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Generate inference on image batch
        pred_ids = model.generate(images, max_length=config['max_gen_length'])
        predicted_labels += processor.batch_decode(pred_ids, skip_special_tokens=True)
        reference_labels += processor.batch_decode(labels, skip_special_tokens=True)
        progress_bar.update(1)

    # Compute character error rate based on testing data
    cer = cer_metric.compute(predictions=predicted_labels, references=reference_labels)
    return cer