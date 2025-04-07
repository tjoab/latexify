import os
import yaml
import csv
from datetime import datetime
from functools import partial

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from models.load_model import load_model
from train.dataset import ImgLatexDataset, preprocess_data

# TODO: Make sure yaml paths resolve correctly
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_model_from_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_step = checkpoint['step']-1
    return start_step


def train(config_path):
    config = load_config(config_path)
    
    if config['perform_logs']:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_path = os.path.join(config['log_dir'], f'log_{timestamp}.csv')
        with open(log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])
    

    processor, model, device = load_model(config['model_name'])

    dataset = ImgLatexDataset(config['data_dir'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        collate_fn=partial(preprocess_data, processor=processor))
    num_training_steps = len(dataloader) // config['grad_accumulation']



    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()
    scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    start_step = 0
    if config['load_from_checkpoint']:
        start_step = load_model_from_checkpoint(
            os.path.join(config['checkpoint_dir'], config['checkpoint_name']), 
            model, optimizer, scheduler, scaler)
    

    model.train()
    torch.manual_seed(config['random_seed'])
    progress_bar = tqdm(range(num_training_steps), initial=start_step)

    # Main train loop using GRADIENT ACCUMULATION and MIXED PRECISION to save on memory
    for step, batch in enumerate(dataloader):
        if step < start_step:
            continue
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Run forward pass with mixed precision (supported on collab T4)
        with autocast():
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            # Scale loss due to accumulation
            loss = loss / config['grad_accumulation']

        # Backwards pass
        scaler.scale(loss).backward()

        # Update optimizers and take a step every GRAD_ACCUMULATION steps
        if (step+1) % config['grad_accumulation'] == 0 or (step+1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if config['perform_logs']:
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step + 1, loss.item() * config['grad_accumulation']])


        if config['perform_logs'] and (step+1) % config['log_freq'] == 0:
            print(f"Step: {step+1}/{len(dataloader)} Loss {loss.item() * config['grad_accumulation']:.4f}")


        if config['perform_checkpoints'] and (step+1) % config['checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"checkpoint_step_{step+1}.pth")
            torch.save({
                        'step': step + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    }, checkpoint_path)

    if config['perform_save']:
        model.save_pretrained(config['save_dir'])
        processor.save_pretrained(config['save_dir'])

    return model, processor


'''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train/config.yaml")
    args = parser.parse_args()
    train(args.config)
'''