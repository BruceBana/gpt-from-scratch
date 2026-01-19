# src/config.py
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Data
    data_path: str = 'data/input.txt'
    
    # Model - REDUCED for faster training on M1
    vocab_size: int = 65
    n_embd: int = 256        # Was 384, now 256
    n_head: int = 4          # Was 6, now 4
    n_layer: int = 4         # Was 6, now 4
    block_size: int = 128    # Was 256, now 128
    dropout: float = 0.2
    
    # Training - FASTER iterations
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 3000    # Was 5000, now 3000
    eval_interval: int = 300  # Was 500, now 300
    eval_iters: int = 200
    
    # Device - USE MPS (M1 GPU)
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    seed: int = 1337