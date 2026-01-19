import torch
from src.config import Config
from src.data_loader import ShakespeareDataset
from src.models.bigram import BigramLanguageModel

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
    config.max_iters = 3000  # Fewer iterations for bigram baseline
    
    torch.manual_seed(config.seed)
    
    # Load data
    print("Loading data...")
    dataset = ShakespeareDataset(config)
    
    # Create model
    print(f"\nInitializing Bigram model...")
    print(f"Vocabulary size: {config.vocab_size}")
    model = BigramLanguageModel(config)
    model = model.to(config.device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print(f"\nTraining on {config.device}...")
    print(f"Max iterations: {config.max_iters}")
    print(f"Batch size: {config.batch_size}")
    print(f"Block size: {config.block_size}\n")
    
    for iter in range(config.max_iters):
        # Evaluate loss periodically
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config)
            print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Sample batch
        xb, yb = dataset.get_batch('train')
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Generate sample
    print("\n" + "="*50)
    print("Generating sample from trained model...")
    print("="*50 + "\n")
    
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=500)
    print(dataset.tokenizer.decode(generated[0].tolist()))
    
    # Save model
    print("\n" + "="*50)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_loss': losses['train'],
        'val_loss': losses['val'],
    }, 'models/bigram_baseline.pt')
    print("✅ Model saved to models/bigram_baseline.pt")
    print(f"Final train loss: {losses['train']:.4f}")
    print(f"Final val loss: {losses['val']:.4f}")

if __name__ == '__main__':
    train()