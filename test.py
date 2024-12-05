
import torch
from model_architecture import *
from helper_functions import *


chars = ""
with open("data/vocab_chars.txt", "r", encoding="utf-8") as f:
    text = f.read()
    chars = sorted(list(set(text)))



prompt = "Hello! Can you see me?"
context = torch.tensor(
    char_tokenizer(prompt, chars, mode="encoder"), dtype=torch.long, device=device
)
model=  torch.load("model.pth")
model_output = model.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist()
generated_chars = char_tokenizer(model_output, chars, mode="decoder")
print(generated_chars)
