"""
Train CobraSSM on synthetic associative recall, save checkpoint,
then run Needle-in-Haystack to demonstrate trained retrieval.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from cobrassm import CobraConfig, CobraForCausalLM

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cobra_trained')

# ─── Data Generation ───────────────────────────────────────────────
def generate_recall_batch(batch_size, num_pairs, vocab_size=100):
    """Generates key-value pairs followed by SEP + query."""
    key_range = list(range(10, 50))
    val_range = list(range(51, 91))
    seq_len = 2 * num_pairs + 2  # pairs + SEP + query

    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    sep_pos = 2 * num_pairs
    query_pos = sep_pos + 1

    for b in range(batch_size):
        keys = random.sample(key_range, num_pairs)
        vals = [random.choice(val_range) for _ in range(num_pairs)]

        for i in range(num_pairs):
            x[b, 2*i] = keys[i]
            x[b, 2*i+1] = vals[i]

        query_idx = random.randint(0, num_pairs - 1)
        x[b, sep_pos] = 1   # SEP
        x[b, query_pos] = keys[query_idx]

        # Only supervise the answer position
        y[b, query_pos] = vals[query_idx]

    return x, y, query_pos


# ─── Needle-in-Haystack Test ──────────────────────────────────────
def run_needle_test(model, device, vocab_size=100):
    """
    Embed a key-value pair, surround with filler, query after SEP.
    Tests if the model can retrieve the value at various depths.
    """
    model.eval()
    key_range = list(range(10, 50))
    val_range = list(range(51, 91))
    
    context_lengths = [32, 64, 128, 256]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    print(f"\n{'':>8}", end="")
    for d in depths:
        print(f" | {d*100:>5.0f}%", end="")
    print(" |")
    print("-" * (10 + 9 * len(depths)))
    
    total_correct = 0
    total_tests = 0
    
    for ctx_len in context_lengths:
        print(f"{ctx_len:>7}T", end="")
        
        for depth in depths:
            # Create the test: filler + needle_pair + filler + SEP + query
            num_filler_pairs = (ctx_len - 4) // 2  # -4 for needle_k, needle_v, SEP, query
            if num_filler_pairs < 1:
                print(f" |   N/A", end="")
                continue
            
            # Build sequence
            needle_key = random.choice(key_range)
            needle_val = random.choice(val_range)
            
            # Filler: random key-value pairs (different keys)
            available_keys = [k for k in key_range if k != needle_key]
            filler_keys = random.choices(available_keys, k=num_filler_pairs)
            filler_vals = random.choices(val_range, k=num_filler_pairs)
            
            # Build filler token list
            filler_tokens = []
            for fk, fv in zip(filler_keys, filler_vals):
                filler_tokens.extend([fk, fv])
            
            # Insert needle at depth
            insert_pos = int(len(filler_tokens) * depth)
            insert_pos = insert_pos - (insert_pos % 2)  # Align to even position
            
            seq = filler_tokens[:insert_pos] + [needle_key, needle_val] + filler_tokens[insert_pos:]
            seq = seq[:ctx_len - 2]  # Leave room for SEP + query
            seq += [1, needle_key]   # SEP + query_key
            
            input_ids = torch.tensor([seq], dtype=torch.long, device=device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                pred = outputs.logits[0, -1].argmax().item()
            
            correct = (pred == needle_val)
            total_correct += int(correct)
            total_tests += 1
            
            status = "✓" if correct else "✗"
            print(f" | {status:>5}", end="")
        
        print(" |")
    
    accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0
    print(f"\nRetrieval Accuracy: {total_correct}/{total_tests} = {accuracy:.1f}%")
    return accuracy


# ─── Main Pipeline ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CobraSSM: Train → Save → Needle-in-Haystack")
    print("=" * 60)
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    vocab_size = 100
    config = CobraConfig(
        vocab_size=vocab_size,
        d_model=128,
        num_hidden_layers=1,
        d_state=16,
        num_scales=4,
        num_slots=32
    )
    
    model = CobraForCausalLM(config).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # ─── Phase 1: Train on Synthetic Recall ───
    print("\n--- Phase 1: Synthetic Recall Training ---")
    
    num_pairs = 7  # 7 key-value pairs per sequence
    batch_size = 64
    iterations = 500
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01, foreach=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    start_time = time.time()
    
    for i in range(iterations):
        x, y, query_pos = generate_recall_batch(batch_size, num_pairs, vocab_size)
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x, labels=None)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if i % 50 == 0:
            with torch.no_grad():
                preds = logits[:, query_pos, :].argmax(-1)
                targets = y[:, query_pos]
                acc = (preds == targets).float().mean().item()
            elapsed = time.time() - start_time
            print(f"  Step {i:4d} | Loss: {loss.item():.4f} | RecallAcc: {acc*100:5.1f}% | {elapsed:.0f}s")
    
    # Final accuracy
    with torch.no_grad():
        x, y, query_pos = generate_recall_batch(256, num_pairs, vocab_size)
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = outputs.logits[:, query_pos, :].argmax(-1)
        final_acc = (preds == y[:, query_pos]).float().mean().item()
    
    train_time = time.time() - start_time
    print(f"\n  Final Recall Accuracy: {final_acc*100:.1f}%")
    print(f"  Training Time: {train_time:.0f}s")
    
    # ─── Phase 2: Save Checkpoint ───
    print(f"\n--- Phase 2: Saving Checkpoint ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    # Save manually to avoid HF tied-weight serialization issues
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'pytorch_model.bin'))
    config.save_pretrained(SAVE_DIR)
    print(f"  Saved to: {SAVE_DIR}")
    
    # ─── Phase 3: Reload & Needle-in-Haystack ───
    print(f"\n--- Phase 3: Needle-in-Haystack (Trained Model) ---")
    
    # Reload from checkpoint to verify save/load works
    loaded_config = CobraConfig.from_pretrained(SAVE_DIR)
    loaded_model = CobraForCausalLM(loaded_config).to(device)
    state = torch.load(os.path.join(SAVE_DIR, 'pytorch_model.bin'), map_location=device, weights_only=True)
    loaded_model.load_state_dict(state, strict=False)
    loaded_model.eval()
    print(f"  Loaded from checkpoint ✓")

    
    accuracy = run_needle_test(loaded_model, device, vocab_size)
    
    # ─── Summary ───
    print("\n" + "=" * 60)
    print(f"  SUMMARY")
    print(f"  Train Recall: {final_acc*100:.1f}%")
    print(f"  Needle Retrieval: {accuracy:.1f}%")
    print(f"  Training Time: {train_time:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
