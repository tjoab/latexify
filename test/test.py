
import yaml
from functools import partial
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from evaluate import load

from models.load_model import load_model
from train.dataset import ImgLatexDataset, preprocess_data


# TODO: Make sure yaml paths resolve correctly
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    


def test(config_path):
    config = load_config(config_path)
    
    processor, model, device = load_model(config['model_name'])

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

        pred_ids = model.generate(images, max_length=256)
        predicted_labels += processor.batch_decode(pred_ids, skip_special_tokens=True)
        reference_labels += processor.batch_decode(labels, skip_special_tokens=True)
        progress_bar.update(1)

    cer = cer_metric.compute(predictions=predicted_labels, references=reference_labels)

    return cer