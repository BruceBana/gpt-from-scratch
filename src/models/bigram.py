import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config

class BigramLanguageModel(nn.Module):
    """
    Simplest possible language model - just predicts next token
    based on current token (no context beyond 1 token).
    
    This serves as baseline to show GPT improvement.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # Each token directly reads logits for next token from lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.vocab_size)
    
    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target indices (optional)
        
        Returns:
            logits: (B, T, vocab_size) predictions
            loss: scalar (if targets provided)
        """
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy: needs (B*T, C) and (B*T)
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate max_new_tokens new tokens, conditioning on idx.
        
        Args:
            idx: (B, T) tensor of conditioning tokens
            max_new_tokens: number of tokens to generate
        
        Returns:
            idx: (B, T + max_new_tokens) tensor
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(idx)
            # Focus only on last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        
        return idx