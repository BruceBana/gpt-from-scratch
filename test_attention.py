import torch
from src.config import Config
from src.models.attention import Head, MultiHeadAttention

def test_single_head():
    config = Config()
    B, T, C = 4, 8, 32  # batch, time, channels
    
    head = Head(
        n_embd=C, 
        head_size=16, 
        block_size=config.block_size,
        dropout=0.1
    )
    
    x = torch.randn(B, T, C)
    out = head(x)
    
    print(f"Single head test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (B, T, 16), f"Expected (4, 8, 16), got {out.shape}"
    print(f"Single head working!\n")

def test_multi_head():
    config = Config()
    B, T, C = 4, 8, 384
    
    mha = MultiHeadAttention(
        n_embd=C,
        n_head=6,
        block_size=config.block_size,
        dropout=0.1
    )
    
    x = torch.randn(B, T, C)
    out = mha(x)
    
    print(f"Multi-head attention test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (B, T, C), f"Expected {x.shape}, got {out.shape}"
    print(f"Multi-head attention working!\n")

if __name__ == '__main__':
    test_single_head()
    test_multi_head()
    print("="*50)
    print("All attention tests passed!")
    print("="*50)