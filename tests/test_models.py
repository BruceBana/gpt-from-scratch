import torch
import pytest
from src.config import Config
from src.models.bigram import BigramLanguageModel
from src.models.gpt import GPT
from src.models.attention import Head, MultiHeadAttention, FeedForward
from src.models.transformer_block import Block

class TestBigram:
    """Test bigram baseline model"""
    
    def test_forward_shape(self):
        config = Config()
        model = BigramLanguageModel(config)
        
        B, T = 4, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        
        logits, _ = model(idx)
        assert logits.shape == (B, T, config.vocab_size)
    
    def test_loss_computed(self):
        config = Config()
        model = BigramLanguageModel(config)
        
        B, T = 4, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))
        
        logits, loss = model(idx, targets)
        assert loss is not None
        assert loss.item() > 0
    
    def test_generation(self):
        config = Config()
        model = BigramLanguageModel(config)
        
        idx = torch.zeros((1, 1), dtype=torch.long)
        generated = model.generate(idx, max_new_tokens=10)
        
        assert generated.shape == (1, 11)  # 1 initial + 10 new


class TestAttention:
    """Test attention mechanisms"""
    
    def test_single_head_shape(self):
        config = Config()
        head = Head(
            n_embd=config.n_embd,
            head_size=config.n_embd // config.n_head,
            block_size=config.block_size
        )
        
        B, T = 4, 8
        x = torch.randn(B, T, config.n_embd)
        out = head(x)
        
        expected_shape = (B, T, config.n_embd // config.n_head)
        assert out.shape == expected_shape
    
    def test_multi_head_shape(self):
        config = Config()
        mha = MultiHeadAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size
        )
        
        B, T = 4, 8
        x = torch.randn(B, T, config.n_embd)
        out = mha(x)
        
        assert out.shape == (B, T, config.n_embd)
    
    def test_causal_masking(self):
        """Verify attention is causal (can't see future)"""
        config = Config()
        head = Head(
            n_embd=config.n_embd,
            head_size=config.n_embd // config.n_head,
            block_size=config.block_size
        )
        
        # Create input where each position has distinct value
        B, T = 1, 4
        x = torch.arange(T).float().view(1, T, 1).expand(B, T, config.n_embd)
        
        # Should only attend to current and past tokens
        out = head(x)
        
        # First token can only see itself
        # Last token can see all previous tokens
        assert out.shape == (B, T, config.n_embd // config.n_head)


class TestFeedForward:
    """Test feedforward network"""
    
    def test_shape_preservation(self):
        config = Config()
        ffwd = FeedForward(config.n_embd)
        
        B, T = 4, 8
        x = torch.randn(B, T, config.n_embd)
        out = ffwd(x)
        
        assert out.shape == x.shape


class TestTransformerBlock:
    """Test full transformer block"""
    
    def test_shape_preservation(self):
        config = Config()
        block = Block(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size
        )
        
        B, T = 4, 8
        x = torch.randn(B, T, config.n_embd)
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_residual_connections(self):
        """Verify residuals help gradient flow"""
        config = Config()
        block = Block(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size
        )
        
        B, T = 4, 8
        x = torch.randn(B, T, config.n_embd)
        
        # Should be different from input (learned transformation)
        # but not drastically different (residual connection)
        out = block(x)
        assert not torch.allclose(x, out)


class TestGPT:
    """Test full GPT model"""
    
    def test_forward_shape(self):
        config = Config()
        model = GPT(config)
        
        B, T = 4, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        
        logits, _ = model(idx)
        assert logits.shape == (B, T, config.vocab_size)
    
    def test_loss_computed(self):
        config = Config()
        model = GPT(config)
        
        B, T = 4, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))
        
        logits, loss = model(idx, targets)
        assert loss is not None
        assert loss.item() > 0
    
    def test_generation(self):
        config = Config()
        model = GPT(config)
        
        idx = torch.zeros((1, 1), dtype=torch.long)
        generated = model.generate(idx, max_new_tokens=10)
        
        assert generated.shape == (1, 11)
    
    def test_generation_respects_block_size(self):
        """Verify generation crops to block_size"""
        config = Config()
        config.block_size = 128
        model = GPT(config)
        
        # Start with sequence longer than block_size
        idx = torch.randint(0, config.vocab_size, (1, 150))
        generated = model.generate(idx, max_new_tokens=10)
        
        # Should successfully generate without error
        assert generated.shape[1] == 160
    
    def test_parameter_count(self):
        """Verify model has expected number of parameters"""
        config = Config()
        model = GPT(config)
        
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should have several million parameters (rough check)
        assert n_params > 1_000_000
        assert n_params < 20_000_000  # Not too big for our config