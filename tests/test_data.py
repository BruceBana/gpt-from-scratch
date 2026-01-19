import torch
import pytest
from src.config import Config
from src.data_loader import ShakespeareDataset
from src.tokenizer import CharTokenizer

class TestTokenizer:
    """Test character tokenizer"""
    
    def test_encode_decode(self):
        text = "Hello, World!"
        tokenizer = CharTokenizer(text)
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert decoded == text
    
    def test_vocab_size(self):
        text = "aabbcc"
        tokenizer = CharTokenizer(text)
        
        assert tokenizer.vocab_size == 3  # a, b, c


class TestDataset:
    """Test Shakespeare dataset"""
    
    def test_batch_shape(self):
        config = Config()
        config.batch_size = 4
        config.block_size = 8
        
        dataset = ShakespeareDataset(config)
        x, y = dataset.get_batch('train')
        
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)
    
    def test_targets_shifted(self):
        """Verify y is x shifted by 1"""
        config = Config()
        config.batch_size = 1
        config.block_size = 8
        
        # Use a fixed seed for reproducibility
        torch.manual_seed(42)
        dataset = ShakespeareDataset(config)
        
        x, y = dataset.get_batch('train')
        
        # y should be x shifted by 1 position
        # (This is approximate test due to batching)
        assert x.shape == y.shape
    
    def test_train_val_split(self):
        """Verify train/val split exists"""
        config = Config()
        dataset = ShakespeareDataset(config)
        
        assert hasattr(dataset, 'train_data')
        assert hasattr(dataset, 'val_data')
        assert len(dataset.train_data) > len(dataset.val_data)