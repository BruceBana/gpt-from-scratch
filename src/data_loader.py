import torch
from src.tokenizer import CharTokenizer
from src.config import Config

class ShakespeareDataset:
    """Dataset for character-level language modeling."""
    
    def __init__(self, config: Config):
        """
        Initialize dataset.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Load text
        with open(config.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Loaded {len(text):,} characters")
        
        # Create tokenizer
        self.tokenizer = CharTokenizer(text)
        
        # Update config with vocab size
        config.vocab_size = self.tokenizer.vocab_size
        
        # Encode entire dataset
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        # Train/val split (90/10)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens: {len(self.val_data):,}")
    
    def get_batch(self, split: str):
        """
        Generate a batch of data.
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            x: Input tensor (batch_size, block_size)
            y: Target tensor (batch_size, block_size)
        """
        data = self.train_data if split == 'train' else self.val_data
        
        # Random starting positions
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        
        # Extract sequences
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.config.block_size+1] for i in ix])
        
        # Move to device
        x, y = x.to(self.config.device), y.to(self.config.device)
        
        return x, y
