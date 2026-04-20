import torch
import time
from cobrassm import CobraConfig, CobraForCausalLM

def benchmark_scaling():
    print("=" * 60)
    print("  COBRASSM PARALLEL SCAN THROUGHPUT BENCHMARK")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config = CobraConfig(
        d_model=128, 
        n_layers=2, 
        d_state=16, 
        num_scales=4, 
        vocab_size=100
    )
    model = CobraForCausalLM(config).to(device).eval()
    
    # Test different sequence lengths
    lengths = [512, 1024, 2048, 4096]
    batch_size = 4
    
    # Warmup
    dummy = torch.randint(0, 100, (1, 64), device=device)
    with torch.no_grad():
        _ = model(dummy)
    
    for L in lengths:
        input_ids = torch.randint(0, 100, (batch_size, L), device=device)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_ids)
        if device.type == "mps":
            torch.mps.synchronize()
        duration = time.time() - start_time
        
        tokens_per_sec = (batch_size * L) / duration
        print(f"Length: {L:<5} | Latency: {duration:.4f}s | Throughput: {tokens_per_sec:8.1f} tokens/sec")

    print("\nBenchmark Complete.")
    print("Note: On Mac/MPS, the Parallel Scan significantly reduces the time ")
    print("spent in Python overhead for long sequences.")

if __name__ == "__main__":
    benchmark_scaling()
