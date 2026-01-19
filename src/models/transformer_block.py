import torch.nn as nn
from src.models.attention import MultiHeadAttention, FeedForward

class Block(nn.Module):
    """
    Transformer block: communication (attention) followed by computation (feedforward).
    
    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> + (residual)
          -> LayerNorm -> FeedForward -> + (residual)
    
    Key design choices:
    1. Pre-norm (LayerNorm before attention/FFN) - helps training stability
    2. Residual connections (+) - helps gradient flow
    3. Attention first, then FFN - standard transformer pattern
    """
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) input embeddings
        Returns:
            x: (B, T, C) transformed embeddings
        """
        # Self-attention with residual connection
        x = x + self.sa(self.ln1(x))
        
        # Feed-forward with residual connection  
        x = x + self.ffwd(self.ln2(x))
        
        return x