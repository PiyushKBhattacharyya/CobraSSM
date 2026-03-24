"""
Needle-in-a-Haystack: Long-Context Retrieval Stress Test for CobraSSM.

Inserts a "needle" fact at various depths in a long context of filler tokens,
then queries the model to retrieve it. Tests retrieval accuracy across
multiple context lengths and needle depths.
"""
import torch
import time
from transformers import GPT2Tokenizer
from cobrassm import CobraConfig, CobraForCausalLM

def create_haystack(tokenizer, context_length, needle_depth, needle_text, query_text):
    """
    Build a sequence: [filler...] [needle] [filler...] [query]
    needle_depth: float 0.0-1.0, where to insert the needle
    """
    # Tokenize needle and query
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)
    
    # Calculate filler needed
    filler_needed = context_length - len(needle_ids) - len(query_ids)
    if filler_needed < 0:
        raise ValueError(f"Context length {context_length} too short for needle+query")
    
    # Create filler (repeating pattern of common tokens)
    filler_text = "The weather is nice today and the birds are singing in the trees. "
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    
    # Repeat filler to fill the context
    full_filler = (filler_ids * (filler_needed // len(filler_ids) + 1))[:filler_needed]
    
    # Insert needle at the specified depth
    insert_pos = int(len(full_filler) * needle_depth)
    
    # Build: [filler_before] [needle] [filler_after] [query]
    sequence = full_filler[:insert_pos] + needle_ids + full_filler[insert_pos:] + query_ids
    
    # Trim to exact context_length
    sequence = sequence[:context_length]
    
    return torch.tensor([sequence])


def run_needle_test():
    print("=" * 60)
    print("  NEEDLE-IN-A-HAYSTACK: CobraSSM Long-Context Stress Test")
    print("=" * 60)
    
    # Setup
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Use a model trained on WikiText or just test the architecture
    config = CobraConfig(
        vocab_size=len(tokenizer),
        d_model=128,
        num_hidden_layers=4,
        d_state=16,
        num_scales=4
    )
    
    # Try DirectML first, fall back to CPU
    device = torch.device('cpu')
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using DirectML: {device}")
    except ImportError:
        print("Using CPU")
    
    model = CobraForCausalLM(config).to(device)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    # Define test parameters
    needle_text = " The secret color of the magic bird is definitely magenta. "
    query_text = " What is the secret color of the magic bird? The answer is:"
    target_word = "magenta"
    target_id = tokenizer.encode(" magenta", add_special_tokens=False)[0]
    
    context_lengths = [512, 1024, 2048, 4096, 8192]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    print(f"\nNeedle: '{needle_text.strip()}'")
    print(f"Query:  '{query_text.strip()}'")
    print(f"Target: '{target_word}' (token {target_id})")
    
    # Results table
    print(f"\n{'':>8}", end="")
    for d in depths:
        print(f" | {d*100:>5.0f}%", end="")
    print(" |")
    print("-" * (10 + 9 * len(depths)))
    
    results = {}
    
    for ctx_len in context_lengths:
        print(f"{ctx_len:>7}T", end="")
        results[ctx_len] = {}
        
        for depth in depths:
            try:
                input_ids = create_haystack(
                    tokenizer, ctx_len, depth, needle_text, query_text
                ).to(device)
                
                start = time.time()
                with torch.no_grad():
                    generated = model.generate(
                        input_ids, 
                        max_new_tokens=5,
                        do_sample=False
                    )
                elapsed = time.time() - start
                
                # Check if target appears in generated tokens
                new_tokens = generated[0, input_ids.shape[1]:]
                found = target_id in new_tokens.tolist()
                
                # Also check top-1 prediction (even if not generated)
                with torch.no_grad():
                    outputs = model(input_ids)
                top_pred = outputs.logits[0, -1].argmax().item()
                top1_match = (top_pred == target_id)
                
                status = "✓" if found else ("~" if top1_match else "✗")
                results[ctx_len][depth] = status
                print(f" | {status:>5}", end="")
                
            except RuntimeError as e:
                if "not enough GPU video memory" in str(e):
                    results[ctx_len][depth] = "OOM"
                    print(f" |   OOM", end="")
                else:
                    results[ctx_len][depth] = "ERR"
                    print(f" |   ERR", end="")
        
        print(f" | {elapsed:.1f}s" if 'elapsed' in dir() else " |")
    
    # Summary
    print(f"\n{'':>8}", end="")
    for d in depths:
        print(f" | {d*100:>5.0f}%", end="")
    print(" |")
    
    total = sum(1 for r in results.values() for v in r.values() if v in ("✓", "~"))
    total_tests = sum(1 for r in results.values() for v in r.values() if v != "OOM")
    
    print(f"\nRetrieval Score: {total}/{total_tests}")
    print(f"Legend: ✓=found in generation, ~=top-1 match, ✗=miss, OOM=out of memory")
    
    print("\n" + "=" * 60)
    print("  NOTE: This is an UNTRAINED model architecture test.")
    print("  With training, retrieval accuracy should approach 100%.")
    print("=" * 60)


if __name__ == "__main__":
    run_needle_test()
