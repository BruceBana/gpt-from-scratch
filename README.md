# GPT from Scratch

I built a complete GPT implementation in PyTorch following Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial. Started as a learning exercise but ended up going further with proper tests, ablation studies, and clean code structure.

## What I Built

Implemented everything from scratch to understand how transformers actually work - attention mechanisms, layer norms, residual connections. No `nn.Transformer` shortcuts.

Also ran ablation studies to figure out which architectural choices actually matter vs which are just convention.

## Results

GPT model gets validation loss down to 1.55 vs 3.62 for the bigram baseline (57% improvement). Trained on 1MB of Shakespeare text.

### Multi-Head Attention

| Attention Heads | Val Loss | Improvement |
|----------------|----------|-------------|
| 1              | 1.7172   | baseline    |
| 2              | 1.6691   | +2.8%       |
| 4              | 1.6393   | +4.5%       |
| 8              | 1.6288   | +5.1%       |

Most gains come from 1→4 heads. After that it's diminishing returns - 4→8 heads only gets you 0.6%. For this model size, 4 heads is enough.

### Layer Depth

| Layers | Val Loss | Parameters | Improvement |
|--------|----------|-----------|-------------|
| 2      | 1.6960   | 1.6M      | baseline    |
| 4      | 1.6422   | 3.2M      | +3.2%       |
| 6      | 1.6331   | 4.8M      | +3.7%       |
| 8      | 1.6056   | 6.4M      | +5.3%       |

Depth keeps helping without overfitting, which suggests character-level modeling benefits more from deeper networks than I expected.

## Architecture

**Self-Attention**
- Q, K, V projections with scaled dot-product
- Causal masking so the model can't look ahead
- Multi-head version to learn different patterns

**Transformer Block**
- Pre-norm layer normalization
- Residual connections
- Feed-forward network (4x expansion)

**Full Model**
- Token + positional embeddings (transformers don't have built-in position awareness)
- Stack of transformer blocks
- Language modeling head

### Config
```python
n_embd = 256        # Embedding dimension
n_head = 4          # Number of attention heads  
n_layer = 4         # Number of transformer layers
block_size = 128    # Context length
dropout = 0.2       # Dropout rate
vocab_size = 65     # Character vocabulary
```

Total parameters: ~10.8M

## Project Structure
```
gpt-from-scratch/
├── src/
│   ├── models/
│   │   ├── attention.py          # Self-attention mechanisms
│   │   ├── transformer_block.py  # Transformer block
│   │   ├── gpt.py                # Full GPT model
│   │   └── bigram.py             # Baseline
│   ├── config.py
│   ├── tokenizer.py
│   └── data_loader.py
├── experiments/
│   ├── ablation_attention_heads.py
│   ├── ablation_layer_depth.py
│   └── results_*.json
├── tests/
│   ├── test_models.py
│   └── test_data.py
├── train_bigram.py
├── train_gpt.py
└── README.md
```

## Running It

### Setup
```bash
conda create -n gpt python=3.11
conda activate gpt
pip install torch numpy
```

### Training
```bash
# Baseline
python train_bigram.py

# Full GPT
python train_gpt.py

# Ablation studies
python -m experiments.ablation_attention_heads
python -m experiments.ablation_layer_depth
```

### Tests
```bash
pip install pytest pytest-cov
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Key Takeaways

Self-attention provides a massive improvement over bigrams (57% loss reduction). The model actually learns long-range dependencies.

Multi-head attention helps - different heads specialize in different patterns (syntax, semantics, position). But 4 heads is enough for small models, going to 8 doesn't add much.

Residual connections are essential for training deep networks. Without them gradients vanish.

Pre-norm (normalizing before attention) is more stable than post-norm.

Positional embeddings are necessary - transformers have no inherent sense of sequence order.

For character-level modeling on this dataset, depth matters more than width. Adding layers consistently improves performance without overfitting.

## What I Learned

Writing tests first caught shape mismatches and edge cases early. Modular code made running ablation studies easy.

Fixed random seeds everywhere and logged all configs/results to JSON for reproducibility.

## References

- Tutorial: ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
- Paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- Dataset: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Next Steps

- Implement Rotary Position Embeddings (RoPE)
- Add KV-cache for faster inference
- Try BPE tokenization instead of character-level
- Scale to OpenWebText
- Implement Flash Attention
- Add gradient checkpointing for larger models

## License

MIT

---

January 2026