import torch
import pickle
from model_architecture import *
from helper_functions import char_tokenizer, get_random_chunk
import yaml


"""
- this code is to train the model in a small dataset , of characters rather than words because we don't want  here to actually
train rather than digesting the main concepts

Block size: Controls how many tokens per sequence (e.g., 8 tokens in this case).
Batch size: Controls how many sequences are processed together (e.g., 2 sequences at a time).
n_embd: Controls the dimensionality of the embedding space for each token (e.g., 768).

"""

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Dynamically set the device parameter
config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

# Now you can use the config object in your training script
batch_size = config["training"]["batch_size"]
block_size = config["training"]["block_size"]
n_embd = config["training"]["n_embd"]
n_head = config["training"]["n_head"]
n_layer = config["training"]["n_layer"]
dropout = config["training"]["dropout"]
device = config["training"]["device"]


chars = ""
with open("data/vocab.txt", "r", encoding="utf-8") as f:
    text = f.read()
    chars = sorted(list(set(text)))


device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = len(chars)
max_iters = 500
learning_rate = 3e-4
eval_iters = 100
model = GPTLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model = model.to(device)


def get_batch(split):
    data = get_random_chunk(chars, batch_size, block_size, split=split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print("ix : ",ix.shape) # 4
    x = torch.stack([data[i : i + block_size] for i in ix])
    # print("x : ",x.shape) #[4, 8]
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    # print("y : ",y.shape) #[4, 8]
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
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
    # print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(
            f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open("model-01.pkl", "wb") as f:
    pickle.dump(model, f)
print("model saved")


prompt = "Hello! Can you see me?"
context = torch.tensor(
    char_tokenizer(prompt, chars, mode="encoder"), dtype=torch.long, device=device
)
model_output = model.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist()
generated_chars = char_tokenizer(model_output, chars, mode="decoder")
print(generated_chars)
