from transformers import TrOCRProcessor
from tqdm.auto import tqdm
from evaluate import load
import os 
from data.ink import read_inkml_file


'''
Extending a tokenizer adds new token IDs, but the pretrained model's embeddings only cover the 
original vocabulary. The new tokens are randomly initialized, so the model has no learned 
understanding of them. This can lead to unstable training, slower convergence, and poor 
generalization—especially on non-LaTeX tasks where the original embeddings were well-tuned. 
To avoid this, we first check if the tokenizer can already handle our LaTeX corpus.
'''
def check_tokenizer_coverage(path: str, processor: TrOCRProcessor) -> float:
    """
    Checks how well a tokenizer can 'cover' the LaTeX vocabulary for data at a specficied path,
    by tokenizing the labels then subsequenty decoding and checking for errors (CER)
    
    Parameters
    ----------
    path : str
        Path to the directory of .inkml data
    processor : TrOCRProcessor
        A refrence to the processor being tested
    
    Returns
    -------
    float
        The character error rate (CER) over the specified data directory
    """
    
    cer_metric = load('cer')
    predictions, references = [], []

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
