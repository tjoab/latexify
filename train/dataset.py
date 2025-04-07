import os
from torch.utils.data import Dataset
from data.inkml_reader import read_inkml_file
from data.render import render_ink

class ImgLatexDataset(Dataset):
    def __init__(self, data_dir):
        self.dir_list = sorted(os.listdir(data_dir))
        self.dir = data_dir

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.dir_list[idx])
        ink = read_inkml_file(img_path)
        img_raw = render_ink(ink).convert("RGB")

        label_raw = ink.annotations.get('normalizedLabel')

        return img_raw, label_raw

def preprocess_data(batch, processor):
    images_raw, labels_raw = zip(*batch)

    # Images processed and stacked into a tensor of size (# training examples, 3, 384, 384)
    images_processed = processor.image_processor(images=images_raw, return_tensors="pt").pixel_values

    # Process LaTeX labels using pretrained tokenizer
    labels_processed = processor.tokenizer(labels_raw, padding=True, return_tensors="pt").input_ids

    return images_processed, labels_processed
