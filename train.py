import torch
from model_architecture import *
from helper_functions import *
import yaml
import mlflow
import os
import warnings

warnings.filterwarnings("ignore")
"""

- this code is to train the model in a small dataset , of characters rather than words because we don't want  here to actually
train rather than digesting the main concepts
=
Block size: Controls how many tokens per sequence (e.g., 8 tokens in this case).
Batch size: Controls how many sequences are processed together (e.g., 2 sequences at a time).
n_embd: Controls the dimensionality of the embedding space for each token (e.g., 768).

=====================================================

With utf-8: The chars list includes all unique characters, regardless of whether they are ASCII or Unicode.
With ASCII: Only ASCII characters are included, and errors will occur if non-ASCII characters are present in the file.
With Other Encodings: The interpretation of non-ASCII characters depends on the encoding. Some characters may be misinterpreted, replaced by placeholders, or cause errors.

"""

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# Dynamically set the device parameter
config["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = config["training"]["batch_size"]
block_size = config["training"]["block_size"]
device = config["training"]["device"]
use_mlflow = config["training"]["use_mlflow"]
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = config["training_params"]["max_iters"]
learning_rate = config["training_params"]["learning_rate"]
eval_iters = config["training_params"]["eval_iters"]

with open("data/vocab_chars.txt", "r", encoding="utf-8") as f:
    text = f.read()
    chars = sorted(list(set(text)))
vocab_size = len(chars)



model = GPTLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model = model.to(device)

# getting data batch for training 
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


def run_training():
    # to suppress a warning in mlflow
    for iter in range(max_iters):
        # print(iter)
        if iter % eval_iters == 0:
            losses = estimate_loss()
            print(
                f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}"
            )

        # sample a batch of data
        input_char, output = get_batch("train")

        # evaluate the loss
        logits, loss = model.forward(input_char, output)
        optimizer.zero_grad(set_to_none=True) #zering gradiants to avoid accumelation
        loss.backward() #calculate new gradiants
        optimizer.step() #update weights 
        if use_mlflow:
            mlflow.log_metric("train_loss", losses["train"], step=iter)
            mlflow.log_metric("validation_loss", losses["val"], step=iter)
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name="llm_model",
                input_example=None,
            )

    print(loss.item())
    save_weights(model, ".", "pth")


if use_mlflow:
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.set_experiment("Baseline Model")
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model, artifact_path="model", pip_requirements="requirements.txt"
        )
        run_training()
else:
    run_training()


