# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'math'
wandb_run_name = 'mini-gpt'

dataset = 'math'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 2048 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 1e-6 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# init_from = 'resume'
#######################################################################################
import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


num_workers = batch_size
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

class MathDataset(Dataset):
    def __init__(self, input_file_path, split):
        with open(input_file_path, 'r') as f:
            lines = f.readlines()
        if split == "train":
            lines = lines[:int(len(lines)*0.9)]
        else:
            lines = lines[int(len(lines)*0.9):]
        full_lines = []
        for l in lines:
            if len(l) <= block_size:
                l = l + '\n' * (block_size - len(l) + 1)
            full_lines.append(l)
        self.data = full_lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx][:block_size+1]
        sp = line.find('=')
        dix = [stoi[s] for s in line]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        y[:sp] = -1
        return x, y

train_dataset = MathDataset(os.path.join(data_dir, 'input.txt'), 'train')
train_loader = DataLoader(
            train_dataset,
            sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
train_data_iter = iter(train_loader)

val_dataset = MathDataset(os.path.join(data_dir, 'input.txt'), 'val')
val_loader = DataLoader(
            val_dataset,
            sampler=torch.utils.data.RandomSampler(val_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
val_data_iter = iter(val_loader)

def get_batch(split):
    global train_data_iter, val_data_iter
    if split == "train":
        try:
            x, y = next(train_data_iter)
        except StopIteration:
            train_data_iter = iter(train_loader)
            x, y = next(train_data_iter)
    else:
        try:
            x, y = next(val_data_iter)
        except StopIteration:
            val_data_iter = iter(val_loader)
            x, y = next(val_data_iter)

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
