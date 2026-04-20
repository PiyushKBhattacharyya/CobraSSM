import torch
import time
from cobrassm import CobraConfig, CobraForCausalLM

def linear_stability_test():
    print("=" * 60)
    print("  COBRASSM LINEAR STABILITY TEST (22M MODEL)")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = CobraConfig(
        vocab_size=50257, 
        d_model=256, 
        num_hidden_layers=6, 
        d_state=16, 
        num_scales=4
    )
    
    print(f"Initializing 22M model on {device}...")
    model = CobraForCausalLM(config).to(device).eval()
    
    lengths = [1024, 2048, 4096, 8192]
    batch_size = 1
    
    # Warmup
    torch.randint(0, 50257, (1, 64), device=device)
    with torch.no_grad():
        _ = model(torch.randint(0, 50257, (1, 64), device=device))

    print(f"\n{'Length':<10} | {'Latency':<10} | {'Throughput':<15}")
    print("-" * 45)
    
    for L in lengths:
        input_ids = torch.randint(0, 50257, (batch_size, L), device=device)
        
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_ids)
            
        if device.type == "mps":
            torch.mps.synchronize()
        duration = time.time() - start_time
        
        throughput = (batch_size * L) / duration
        print(f"{L:<10} | {duration:>8.4f}s | {throughput:>10.1f} tokens/sec")

    print("\n" + "=" * 60)
    print("  ANALYSIS:")
    print("  If the 'Throughput' stays relatively constant, the model has")
    print("  successfully defeated the O(L^2) Transformer bottleneck.")
    print("=" * 60)

if __name__ == "__main__":
    linear_stability_test()
