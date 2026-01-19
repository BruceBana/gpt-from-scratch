class CharTokenizer:
    """Simple character-level tokenizer for text."""
    
    def __init__(self, text: str):
        """
        Initialize tokenizer with vocabulary from text.
        
        Args:
            text: Training corpus to extract vocabulary from
        """
        # Get unique characters and sort
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(chars)}")
    
    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs."""
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens: list[int]) -> str:
        """Convert list of token IDs to text."""
        return ''.join([self.itos[i] for i in tokens])
