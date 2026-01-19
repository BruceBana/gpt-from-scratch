import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from src.models.transformer_block import Block

class GPT(nn.Module):
    """
    Full GPT (Generatively Pretrained Transformer) model.
    
    Architecture:
    1. Token embeddings (what token is this?)
    2. Position embeddings (where is this token?)
    3. Stack of Transformer blocks (process the sequence)
    4. Layer norm (stabilize final output)
    5. Linear head (predict next token)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        
        # Embeddings
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small random values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target indices (optional)
        
        Returns:
            logits: (B, T, vocab_size) predictions
            loss: scalar (if targets provided)
        """
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        
        # Pass through transformer blocks
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)    # (B, T, C)
        
        # Get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: (B, T) tensor of conditioning tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
        
        Returns:
            idx: (B, T + max_new_tokens) tensor
        """
        for _ in range(max_new_tokens):
            # Crop to block_size (GPT can only see last block_size tokens)
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        
        return idx