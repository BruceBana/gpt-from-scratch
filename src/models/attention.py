import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """
    Single head of self-attention.
    
    Key insight: Each token creates query/key/value vectors.
    - Query: "What am I looking for?"
    - Key: "What do I contain?"  
    - Value: "What do I communicate?"
    
    Attention scores = how much each query matches each key.
    Output = weighted sum of values based on attention scores.
    """
    
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.head_size = head_size
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Causal mask: tokens can only attend to past (not future)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) input embeddings
        Returns:
            out: (B, T, head_size) attended features
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        # Compute attention scores
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # Apply causal mask (prevent looking at future)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Normalize to probabilities
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Weighted aggregation of values
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of attention in parallel.
    
    Why multiple heads? Different heads can attend to different patterns:
    - Head 1: might learn syntax relationships
    - Head 2: might learn semantic relationships  
    - Head 3: might learn positional patterns
    """
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) 
            for _ in range(n_head)
        ])
        
        # Projection back to embedding dimension
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) input
        Returns:
            out: (B, T, C) attended output
        """
        # Run all heads in parallel and concatenate
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # Project back and apply dropout
        out = self.dropout(self.proj(out))
        
        return out


class FeedForward(nn.Module):
    """
    Simple MLP that processes each position independently.
    
    Applied after attention - allows network to "think" about 
    what attention gathered before passing to next layer.
    """
    
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Project back
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)