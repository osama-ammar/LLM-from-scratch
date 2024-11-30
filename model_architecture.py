import torch
import torch.nn as nn
from torch.nn import functional as F

import yaml

"""
- this code is to train the model in a small dataset , of characters rather than words because we don't here to actually train rather 
than digesting the main concepts

the processs
==============
The input tokens are passed into the GPTLanguageModel.
The model embeds the tokens (via token_embedding_table) and adds positional information (position_embedding_table).
The embeddings are passed through several Blocks:

    Inside each Block:
        MultiHeadAttention lets tokens communicate, identifying important relationships.
        FeedForward processes the refined token information.

After all the Blocks, the final token representations are sent to lm_head, which predicts the next token for each position.



when text generation (inference ):
    The predicted token is added to the input sequence.(Autoregressive process)
    The extended sequence is fed back into the model to predict the next token, repeating this process until the desired length of text is generated.

"""

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Now you can use the config object in your  script
block_size = config["model"]["block_size"]
n_embd = config["model"]["n_embd"]
n_head = config["model"]["n_head"]
n_layer = config["model"]["n_layer"]
dropout = config["model"]["dropout"]
device = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    """
    Represents a single head of self-attention.
    This module calculates attention weights and performs weighted aggregation.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # Key projection
        self.query = nn.Linear(n_embd, head_size, bias=False)  # query projection
        self.value = nn.Linear(n_embd, head_size, bias=False)  # value projection
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the attention head.
        x: Tensor of shape (batch, time_steps, embedding_dim)
        Returns: Tensor of shape (batch, time_steps, head_size)
        """
        B, T, C = x.shape
        k = self.key(x)  # Keys: (B, T, head_size)
        q = self.query(x)  # Queries: (B, T, head_size)

        # Compute attention scores
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # Scaled dot-product attention (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # Apply mask for causality
        wei = F.softmax(wei, dim=-1)  # Normalize scores
        wei = self.dropout(wei)  # Apply dropout

        # Aggregate values
        v = self.value(x)  # Values: (B, T, head_size)
        out = wei @ v  # Weighted sum: (B, T, head_size)
        return out


# [1, 0, 0]
# [1, 0.6, 0]
# [1, 0.6, 0.4]
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )  # Create multiple heads
        self.proj = nn.Linear(
            head_size * num_heads, n_embd
        )  # Combine outputs of all heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for multi-head attention.
        x: Tensor of shape (batch, time_steps, embedding_dim)
        Returns: Tensor of shape (batch, time_steps, embedding_dim)
        """
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # Concatenate outputs of all heads
        out = self.dropout(self.proj(out))  # Project back to embedding size
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand dimension
            nn.ReLU(),  # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Reduce back to original dimension
            nn.Dropout(dropout),  # Dropout for regularization
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Represents  a Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # Multi-head attention
        self.ffwd = FeedFoward(n_embd)  # Feedforward layer
        self.ln1 = nn.LayerNorm(n_embd)  # Layer normalization
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass for a transformer block.
        x: Tensor of shape (batch, time_steps, embedding_dim)
        Returns: Tensor of the same shape
        """
        x = self.ln1(x + self.sa(x))  # Residual connection + attention
        x = self.ln2(x + self.ffwd(x))  # Residual connection + feedforward
        return x


class GPTLanguageModel(nn.Module):
    """
    GPT-like language model with embeddings, transformer blocks, and a language modeling head.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # Token embeddings
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )  # Positional embeddings
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )  # Transformer layers
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Language modeling head

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for linear and embedding layers.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        """
        Forward pass of the language model.
        index: Tensor of token indices (batch, time_steps)
        targets: Tensor of target indices (batch, time_steps), optional
        Returns: Logits (batch, time_steps, vocab_size) and loss (if targets provided)
        """
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # Token embeddings
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # Positional embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.blocks(x)  # Apply transformer blocks
        x = self.ln_f(x)  # Final layer normalization
        logits = self.lm_head(x)  # Predictions

        loss = None
        if targets is not None:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # Compute cross-entropy loss

        return logits, loss

    def generate(self, index, max_new_tokens):
        """
        Generate new tokens given a context.
        index: Tensor of initial token indices (batch, time_steps)
        max_new_tokens: Number of tokens to generate
        Returns: Tensor of generated token indices
        """
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]  # Use only the last block_size tokens
            logits, _ = self.forward(index_cond)  # Get logits
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)  # Convert to probabilities
            index_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            index = torch.cat((index, index_next), dim=1)  # Append to the sequence
        return index


if __name__ == "__main__":
    pass
