import torch
import time
from cobrassm import CobraConfig, CobraForCausalLM
from transformers import GPT2Config, GPT2LMHeadModel

def measure_throughput():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        import torch_directml
        device = torch_directml.device()
        print("Using DirectML.")
    except ImportError:
        pass
        
    print(f"Benchmarking on: {device}")
    
    # Cobra Config (Small for throughput analysis)
    config = CobraConfig(
        vocab_size=100,
        d_model=128, # Reduced from 256 to fit in GPU memory
        num_hidden_layers=4,
        d_state=16,
        num_scales=4
    )
    model = CobraForCausalLM(config).to(device).eval()
    
    # GPT2 Config (Similar scale)
    config_gpt = GPT2Config(
        vocab_size=100,
        n_positions=4096,
        n_embd=128,
        n_layer=4,
        n_head=4
    )
    model_gpt = GPT2LMHeadModel(config_gpt).to(device).eval()
    
    batch_size = 1 # Reduced for long sequence testing
    # Test across different sequence lengths
    lengths = [128, 512, 1024, 2048, 4096]

    
    print(f"{'SeqLen':>8} | {'Cobra (ms)':>12} | {'GPT2 (ms)':>12} | {'Cobra TPS':>12}")
    print("-" * 55)
    
    for length in lengths:
        input_ids = torch.randint(0, 100, (batch_size, length)).to(device)
        
        # Bench Cobra
        with torch.no_grad():
            for _ in range(3): model(input_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(5): model(input_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_cobra = (time.time() - start) * 1000 / 5
        
        # Bench GPT2
        with torch.no_grad():
            for _ in range(3): model_gpt(input_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(5): model_gpt(input_ids)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_gpt = (time.time() - start) * 1000 / 5
        
        tokens_per_sec = (batch_size * length) / (time_cobra / 1000)
        
        print(f"{length:8d} | {time_cobra:12.2f} | {time_gpt:12.2f} | {tokens_per_sec:12.1f}")

if __name__ == "__main__":
    measure_throughput()
