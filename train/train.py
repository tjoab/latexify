import os
import csv
from datetime import datetime
from functools import partial

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from typing import Tuple
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from models.load_model import load_model
from train.dataset import ImgLatexDataset, preprocess_data
from tools import load_config


'''
Training all in one go isnt always feasible, so the training loop and setup here allows for the
user to train in chucks, saving partial results as a checkpoint. The script automatically restarts
training a the same point without resuing data previous trained on (within the same epoch)
'''
def load_model_from_checkpoint(checkpoint_path: str, model, optimizer, scheduler, scaler) -> int:
    """
    Loads training state from a checkpoint, including model weights,
    optimizer, scheduler, and gradient scaler state.

    Parameters
    ----------
    checkpoint_path : str
        Path to previously saved checkpoint (.pth)
    model : VisionEncoderDecoderModel
        Refrence to the model used
    optimizer : torch.optim.Optimizer
        Refrence to the optimizer used
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Refrence to the scheduler used
    scaler : torch.cuda.amp.GradScaler
        Refrence to the AMP gradient scaler for mixed-precision training

    Returns
    -------
    int
        The step in which to resume training from
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_step = checkpoint['step']-1
    return start_step


def train(config_path: str) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """
    Runs finetuning/training for a TrOCR-based image-to-LaTeX model using gradient accumulation
    and mixed precision (AMP). Supports checkpointing, logging, and resume training.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file containing all training parameters and configs

    Returns
    -------
    model : VisionEncoderDecoderModel
        The finetuned model
    processor : TrOCRProcessor
        The associated processor
    """
    config = load_config(config_path)
    
    # Create log directory and log file if toggled on
    if config['perform_logs']:
        os.makedirs(config['log_dir'], exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_path = os.path.join(config['log_dir'], f'log_{timestamp}.csv')
        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])
    

    # Load model and assign it to correct device (CPU v GPU)
    processor, model, device = load_model(config['model_name'])


    # Create dataloader for torch using the training data directory
    dataset = ImgLatexDataset(config['data_dir'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        collate_fn=partial(preprocess_data, processor=processor))
    num_training_steps = len(dataloader) // config['grad_accumulation']


    # Define optimizer, scheduler and scaler
    # This model is BIG, so we are going to use mixed precision to speed up training
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()
    scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )

    # Define start step and load from checkpoint if toggled
    start_step = 0
    if config['load_from_checkpoint']:
        start_step = load_model_from_checkpoint(os.path.join(config['checkpoint_dir'], config['checkpoint_name']), 
                                                model, optimizer, scheduler, scaler
                                                )
    

    # Set nodel to train mode and define a random seed this way our dataloader is shuffled the
    # same way and we can 'restart' traning from a checkpoint without worry of resuing data
    model.train()
    torch.manual_seed(config['random_seed'])
    
    # Main train loop using GRADIENT ACCUMULATION and MIXED PRECISION to save on memory
    progress_bar = tqdm(range(num_training_steps), initial=start_step)
    for step, batch in enumerate(dataloader):
        if step < start_step:
            continue
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Run forward pass with mixed precision (supported on T4 GPU)
        # TODO: Will this crash if device='cpu'?
        with autocast():
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            # Scale loss due to accumulation
            loss = loss / config['grad_accumulation']

        # Backwards pass
        scaler.scale(loss).backward()

        # Update optimizers and take a step every 'grad accumulation' steps
        if (step+1) % config['grad_accumulation'] == 0 or (step+1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # Note: log reporting can only be as granular as how large the grad accumulation step is
            if config['perform_logs']:
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step + 1, loss.item() * config['grad_accumulation']])


        # Save checkpoint at specified interval if toggled
        if config['perform_checkpoints'] and (step+1) % config['checkpoint_freq'] == 0:
            os.makedirs(config['checkpoint_dir'], exist_ok=True)
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"checkpoint_step_{step+1}.pth")
            torch.save({
                        'step': step + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    }, checkpoint_path)

    # Save model and processor locally if toggled
    if config['perform_save']:
        os.makedirs(config['save_dir'], exist_ok=True)
        model.save_pretrained(config['save_dir'])
        processor.save_pretrained(config['save_dir'])

    return model, processor
