


'''
When you extend a tokenizer's vocabulary, you're adding new IDs at the end of the vocab. But pretrained model's weights (especially the embedding matrix) are tied directly to the original vocab size.

So the pre-trained model doesn't know what to do with those new 500 tokens — there's no pretrained embedding for them. At best, they're randomly initialized, and the model's performance on anything non-LaTeX can degrade badly.

So we check to see if the tokenizer can actually 'tokenize' our LaTeX corpus.

'''

from tqdm.auto import tqdm
from evaluate import load
import os 
from data.inkml_reader import read_inkml_file


def compute_cer(path, processor):
    """
    Computes CER for a group of InkML images at file path 'path'
    Param path: path to directory of inkml files
    """
    cer_metric = load('cer')
    # Test to see how well the pretrained tokenizer can cover the LaTeX vocab
    predictions = []
    references = []

    progress_bar = tqdm(range(len(os.listdir(path))))

    for ink_path in os.listdir(path):
        ink = read_inkml_file(os.path.join(path, ink_path))
        latex = ink.annotations.get('normalizedLabel')

        token_list = processor.tokenizer.tokenize(latex)
        id_list = processor.tokenizer.convert_tokens_to_ids(token_list)
        decoded_latex = processor.tokenizer.decode(id_list)

        predictions.append(decoded_latex)
        references.append(latex)
        progress_bar.update(1)

    return cer_metric.compute(predictions=predictions, references=references)
