import torch
import torch.nn as nn
import torch.optim as optim
from cobrassm import CobraSSM
import random
import time

def generate_associative_recall_batch(batch_size, seq_len, vocab_size):
    """
    Generates a synthetic associative recall task.
    Format: K1 V1 K2 V2 ... Kn Vn | Query_K | Target_V
    We mask out the loss for everything except the final Target_V prediction.
    """
    # Leave 0 as padding, 1 as query separator
    num_pairs = (seq_len - 2) // 2
    
    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    for b in range(batch_size):
        keys = random.sample(range(2, vocab_size), num_pairs)
        vals = [random.randint(2, vocab_size - 1) for _ in range(num_pairs)]
        
        # Fill the sequence
        for i in range(num_pairs):
            x[b, 2*i] = keys[i]
            x[b, 2*i + 1] = vals[i]
            
        # Select a random key to query
        query_idx = random.randint(0, num_pairs - 1)
        query_key = keys[query_idx]
        target_val = vals[query_idx]
        
        # Place query token and separator
        x[b, seq_len - 2] = 1 # Separator token
        x[b, seq_len - 1] = query_key
        
        # Target wants to predict target_val at the final position
        # We set y to -100 to ignore loss on all other positions
        y[b, :] = -100
        y[b, seq_len - 1] = target_val
        
    return x, y

def train():
    vocab_size = 100
    batch_size = 64
    seq_len = 16 # Shortened for debugging
    iterations = 500
    
    # AMD GPU Support on Windows (DirectML) or standard ROCm/CUDA
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("Using standard CUDA/ROCm.")
        else:
            print("No fast GPU found. To use AMD on Windows, run: pip install torch-directml")
    
    print(f"Training on: {device}")
    
    # Initialize our CobraSSM model
    # Increased depth to 4 layers to allow causal combinatorial shift
    # Reduced slots for small seq_len to avoid empty bucket dilation
    model = CobraSSM(vocab_size=vocab_size, d_model=128, n_layers=4, d_state=16, num_scales=4, num_slots=16)
    model.to(device)
    
    print(f"Model parameters: {model.parameter_count()/1e6:.2f} M")
    
    # Increased learning rate for fast synthetic convergence
    optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    
    for i in range(iterations):
        x, y = generate_associative_recall_batch(batch_size, seq_len, vocab_size)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, _, _ = model(x)
        
        # Align target and prediction
        # Output at t predicts y[t]. y is filled with -100 except at seq_len - 1.
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        
        loss.backward()
        
        # Gradient clipping for stability
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if i % 50 == 0:
            # Calculate exact match accuracy for the batch
            preds = logits[:, -1, :].argmax(dim=-1)
            targets = y[:, -1]
            accuracy = (preds == targets).float().mean().item()
            print(f"Iteration {i} | Loss: {loss.item():.4f} | Accuracy: {accuracy * 100:.1f}% | GNorm: {gnorm:.4f}")

if __name__ == "__main__":
    train()
