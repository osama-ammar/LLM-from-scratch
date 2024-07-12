
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from model_architecture import *
from helper_functions import * 
import yaml


"""
- this code is to train the model in a small dataset , of characters rather than words because we don't here to actually train rather than digesting the main concepts

"""

# Load the configuration file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Dynamically set the device parameter
config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Now you can use the config object in your training script
batch_size = config['training']['batch_size']
block_size = config['training']['block_size']
n_embd = config['training']['n_embd']
n_head = config['training']['n_head']
n_layer = config['training']['n_layer']
dropout = config['training']['dropout']
device = config['training']['device']

#parser = argparse.ArgumentParser(description='This is a demonstration program')
# Here we add an argument to the parser, specifying the expected type, a help message, etc.
# parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')
# args = parser.parse_args()
# Now we can use the argument value in our program.
# print(f'batch size: {args.batch_size}')
# batch_size = args.batch_size # to use the batch_size cmd arg -> python file_name.py -batch_size 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split="train"):
    filename = "data/output_train.txt" if split == 'train' else "data/output_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(tokenizer(decoded_block,mode="encoder"), dtype=torch.long)
            
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #print("ix : ",ix.shape) # 4
    x = torch.stack([data[i:i+block_size] for i in ix])
    #print("x : ",x.shape) #[4, 8]
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    #print("y : ",y.shape) #[4, 8]
    x, y = x.to(device), y.to(device)
    return x, y

chars = ""
with open("data/vocab.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
        
        
vocab_size = len(chars)
max_iters = 500
learning_rate = 3e-4
eval_iters = 100
model = GPTLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model = model.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# create a PyTorch optimizer

for iter in range(max_iters):
    #print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')


prompt = 'Hello! Can you see me?'
context = torch.tensor(tokenizer(prompt,mode="encoder"), dtype=torch.long, device=device)
model_output =model.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist()
generated_chars = tokenizer(model_output,mode="decoder")
print(generated_chars)












