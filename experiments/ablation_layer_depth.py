# experiments/ablation_layer_depth.py
"""
Ablation Study: Impact of Model Depth

Question: How does the number of transformer layers affect performance?

Hypothesis: Deeper models can learn more complex patterns but may be
harder to train and risk overfitting on small datasets.

Experiment: Train GPT with 2, 4, 6, 8 layers and compare.
"""

import torch
import json
from src.config import Config
from src.data_loader import ShakespeareDataset
from src.models.gpt import GPT

@torch.no_grad()
def estimate_loss(model, dataset, config, n_iters=100):
    model.eval()
    losses = torch.zeros(n_iters)
    for k in range(n_iters):
        X, Y = dataset.get_batch('val')
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def train_with_n_layers(n_layers, max_iters=1500):
    """Train model with specified number of layers"""
    print(f"\n{'='*60}")
    print(f"Training with {n_layers} layer(s)")
    print(f"{'='*60}")
    
    config = Config()
    config.n_layer = n_layers
    config.max_iters = max_iters
    config.eval_interval = 500
    
    torch.manual_seed(config.seed)
    
    dataset = ShakespeareDataset(config)
    model = GPT(config).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
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
    
    final_loss = estimate_loss(model, dataset, config, n_iters=200)
    print(f"  Final val loss: {final_loss:.4f}")
    
    return {
        'n_layers': n_layers,
        'final_val_loss': final_loss,
        'n_params': n_params,
        'history': losses_history
    }

def run_experiment():
    """Run ablation study on model depth"""
    
    results = []
    
    for n_layers in [2, 4, 6, 8]:
        result = train_with_n_layers(n_layers, max_iters=1500)
        results.append(result)
    
    with open('experiments/results_layer_depth.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS: Layer Depth")
    print("="*60)
    print(f"{'Layers':<10} {'Final Val Loss':<20} {'Parameters':<15}")
    print("-"*60)
    
    for r in results:
        print(f"{r['n_layers']:<10} {r['final_val_loss']:<20.4f} {r['n_params']:<15,}")
    
    best = min(results, key=lambda x: x['final_val_loss'])
    print("\n" + "="*60)
    print(f"✅ Best configuration: {best['n_layers']} layers")
    print(f"   Val loss: {best['final_val_loss']:.4f}")
    print("="*60)

if __name__ == '__main__':
    run_experiment()