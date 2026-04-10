"""
Stage 2: Curriculum Recall Training (16T → 512T)
Warm-starts each stage from the previous checkpoint.
Runs Needle-in-Haystack after each curriculum stage.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
from cobrassm import CobraConfig, CobraForCausalLM

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cobra_trained')

# ─── Data ─────────────────────────────────────────────────────────
def generate_recall_batch(batch_size, num_pairs, vocab_size=100):
    key_range = list(range(10, 50))
    val_range = list(range(51, 91))
    seq_len = 2 * num_pairs + 2

    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    sep_pos = 2 * num_pairs
    query_pos = sep_pos + 1

    for b in range(batch_size):
        keys = random.sample(key_range, min(num_pairs, len(key_range)))
        if num_pairs > len(key_range):
            keys += random.choices(key_range, k=num_pairs - len(key_range))
        vals = [random.choice(val_range) for _ in range(num_pairs)]

        for i in range(num_pairs):
            x[b, 2*i] = keys[i]
            x[b, 2*i+1] = vals[i]

        query_idx = random.randint(0, num_pairs - 1)
        x[b, sep_pos] = 1
        x[b, query_pos] = keys[query_idx]
        y[b, query_pos] = vals[query_idx]

    return x, y, query_pos


# ─── Needle Test ──────────────────────────────────────────────────
def run_needle_test(model, device, max_len=256):
    model.eval()
    key_range = list(range(10, 50))
    val_range = list(range(51, 91))
    
    context_lengths = [l for l in [32, 64, 128, 256, 512] if l <= max_len]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    total_correct = 0
    total_tests = 0
    
    print(f"  {'':>6}", end="")
    for d in depths:
        print(f" {d*100:>4.0f}%", end="")
    print()
    
    for ctx_len in context_lengths:
        print(f"  {ctx_len:>5}T", end="")
        
        for depth in depths:
            num_filler = (ctx_len - 4) // 2
            if num_filler < 1:
                print(f"   --", end="")
                continue
            
            correct_count = 0
            trials = 5  # Average over 5 trials
            
            for _ in range(trials):
                needle_key = random.choice(key_range)
                needle_val = random.choice(val_range)
                
                available = [k for k in key_range if k != needle_key]
                filler_tokens = []
                for _ in range(num_filler):
                    filler_tokens.extend([random.choice(available), random.choice(val_range)])
                
                insert_pos = int(len(filler_tokens) * depth)
                insert_pos -= insert_pos % 2
                
                seq = filler_tokens[:insert_pos] + [needle_key, needle_val] + filler_tokens[insert_pos:]
                seq = seq[:ctx_len - 2] + [1, needle_key]
                
                input_ids = torch.tensor([seq], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    outputs = model(input_ids)
                    pred = outputs.logits[0, -1].argmax().item()
                
                if pred == needle_val:
                    correct_count += 1
            
            acc = correct_count / trials
            total_correct += correct_count
            total_tests += trials
            
            if acc >= 0.8:
                print(f"   ✓", end="")
            elif acc >= 0.4:
                print(f"   ~", end="")
            else:
                print(f"   ✗", end="")
        
        print()
    
    overall = total_correct / total_tests * 100 if total_tests > 0 else 0
    print(f"  Overall: {total_correct}/{total_tests} = {overall:.0f}%")
    model.train()
    return overall


# ─── Curriculum ───────────────────────────────────────────────────
CURRICULUM = [
    # (num_pairs, steps, lr, batch_size)
    (7,    500, 3e-3, 64),
    (15,   300, 2e-3, 32),
    (31,   400, 1e-3, 16),
    (63,   500, 5e-4,  8),
    (127,  600, 3e-4,  4),
]


def main():
    print("=" * 60)
    print("  CobraSSM: Curriculum Recall Training")
    print("=" * 60)
    
    device = torch.device('cpu')
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
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params\n")
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_start = time.time()
    
    for stage_idx, (num_pairs, steps, lr, bs) in enumerate(CURRICULUM):
        seq_len = 2 * num_pairs + 2
        print(f"--- Stage {stage_idx+1}: {seq_len}T ({num_pairs} pairs) ---")
        print(f"    LR: {lr}, Batch: {bs}, Steps: {steps}")
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=False)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr/10)
        
        model.train()
        stage_start = time.time()
        
        for i in range(steps):
            x, y, qpos = generate_recall_batch(bs, num_pairs, vocab_size)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x).logits
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if i % 100 == 0 or i == steps - 1:
                with torch.no_grad():
                    acc = (logits[:, qpos, :].argmax(-1) == y[:, qpos]).float().mean().item()
                print(f"    Step {i:4d} | Loss: {loss.item():.4f} | Acc: {acc*100:.0f}%")
        
        stage_time = time.time() - stage_start
        print(f"    Stage time: {stage_time:.0f}s")
        
        # Save checkpoint
        ckpt_dir = os.path.join(BASE_DIR, f'stage_{stage_idx+1}')
        model.save_pretrained(ckpt_dir)
        print(f"    Saved: {ckpt_dir}")
        
        # Needle test
        print(f"\n    Needle-in-Haystack (max {seq_len}T):")
        run_needle_test(model, device, max_len=seq_len)
        print()
    
    total_time = time.time() - total_start
    
    # Final comprehensive needle test
    print("=" * 60)
    print("  FINAL Needle-in-Haystack (all lengths)")
    print("=" * 60)
    run_needle_test(model, device, max_len=512)
    
    # Save final
    model.save_pretrained(os.path.join(BASE_DIR, 'final'))
    print(f"\nTotal training time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
