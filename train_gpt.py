import torch
from src.config import Config
from src.data_loader import ShakespeareDataset
from src.models.gpt import GPT

@torch.no_grad()
def estimate_loss(model, dataset, config):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = dataset.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    # Setup
    config = Config()
    torch.manual_seed(config.seed)
    
    # Load data
    print("Loading data...")
    dataset = ShakespeareDataset(config)
    
    # Create model
    print(f"\nInitializing GPT model...")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Number of heads: {config.n_head}")
    print(f"Number of layers: {config.n_layer}")
    print(f"Block size: {config.block_size}")
    
    model = GPT(config)
    model = model.to(config.device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print(f"\nTraining on {config.device}...")
    print(f"Max iterations: {config.max_iters}")
    print(f"Batch size: {config.batch_size}\n")
    
    best_val_loss = float('inf')
    
    for iter in range(config.max_iters):
        # Evaluate loss periodically
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config)
            print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                    'iter': iter,
                }, 'models/gpt_best.pt')
        
        # Sample batch
        xb, yb = dataset.get_batch('train')
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Generate sample
    print("\n" + "="*70)
    print("Generating sample from trained GPT model...")
    print("="*70 + "\n")
    
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=500)
    print(dataset.tokenizer.decode(generated[0].tolist()))
    
    # Final save
    print("\n" + "="*70)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_loss': losses['train'],
        'val_loss': losses['val'],
    }, 'models/gpt_final.pt')
    
    print(f"Best model saved: models/gpt_best.pt (val loss: {best_val_loss:.4f})")
    print(f"Final model saved: models/gpt_final.pt (val loss: {losses['val']:.4f})")
    print(f"\nImprovement over bigram baseline:")
    print(f"   Bigram val loss: 3.6156")
    print(f"   GPT val loss: {losses['val']:.4f}")
    print(f"   Reduction: {((3.6156 - losses['val']) / 3.6156 * 100):.1f}%")

if __name__ == '__main__':
    train()