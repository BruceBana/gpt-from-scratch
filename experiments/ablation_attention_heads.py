"""
Ablation Study: Impact of Number of Attention Heads

Question: Does using multiple attention heads improve performance?

Hypothesis: Multiple heads allow the model to attend to different patterns
(syntax, semantics, position) simultaneously, improving performance.

Experiment: Train GPT with 1, 2, 4, 6 heads and compare validation loss.
"""

import torch
import json
from src.config import Config
from src.data_loader import ShakespeareDataset
from src.models.gpt import GPT

@torch.no_grad()
def estimate_loss(model, dataset, config, n_iters=100):
    """Quick loss estimation"""
    model.eval()
    losses = torch.zeros(n_iters)
    for k in range(n_iters):
        X, Y = dataset.get_batch('val')
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def train_with_n_heads(n_heads, max_iters=1500):
    """Train model with specified number of heads"""
    print(f"\n{'='*60}")
    print(f"Training with {n_heads} attention head(s)")
    print(f"{'='*60}")
    
    # Setup config
    config = Config()
    config.n_head = n_heads
    config.n_embd = 256  # Must be divisible by n_heads
    config.max_iters = max_iters
    config.eval_interval = 500
    
    # Ensure n_embd is divisible by n_head
    assert config.n_embd % config.n_head == 0, \
        f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
    
    torch.manual_seed(config.seed)
    
    # Load data
    dataset = ShakespeareDataset(config)
    
    # Create model
    model = GPT(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    losses_history = []
    
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0:
            val_loss = estimate_loss(model, dataset, config, n_iters=50)
            print(f"  Step {iter:4d}: val loss {val_loss:.4f}")
            losses_history.append({'iter': iter, 'val_loss': val_loss})
        
        xb, yb = dataset.get_batch('train')
        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Final evaluation
    final_loss = estimate_loss(model, dataset, config, n_iters=200)
    print(f"  Final val loss: {final_loss:.4f}")
    
    return {
        'n_heads': n_heads,
        'final_val_loss': final_loss,
        'n_params': n_params,
        'history': losses_history
    }

def run_experiment():
    """Run ablation study on number of attention heads"""
    
    results = []
    
    # Test different numbers of heads
    # n_embd=256, so valid divisors: 1, 2, 4, 8, 16, 32, 64, 128, 256
    for n_heads in [1, 2, 4, 8]:
        result = train_with_n_heads(n_heads, max_iters=1500)
        results.append(result)
    
    # Save results
    with open('experiments/results_attention_heads.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS: Attention Heads")
    print("="*60)
    print(f"{'Heads':<10} {'Final Val Loss':<20} {'Parameters':<15}")
    print("-"*60)
    
    for r in results:
        print(f"{r['n_heads']:<10} {r['final_val_loss']:<20.4f} {r['n_params']:<15,}")
    
    # Find best
    best = min(results, key=lambda x: x['final_val_loss'])
    print("\n" + "="*60)
    print(f"✅ Best configuration: {best['n_heads']} heads")
    print(f"   Val loss: {best['final_val_loss']:.4f}")
    print("="*60)

if __name__ == '__main__':
    run_experiment()