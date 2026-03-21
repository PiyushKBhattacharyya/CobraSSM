import torch
import torch.nn as nn
import torch.optim as optim
from cobrassm import CobraSSM
import random


def generate_associative_recall_batch(batch_size, seq_len, vocab_size):
    key_range = list(range(10, 50))
    val_range = list(range(51, 91))
    assert (seq_len - 2) % 2 == 0
    num_pairs = (seq_len - 2) // 2
    sep_pos   = 2 * num_pairs
    query_pos = sep_pos + 1

    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        keys = random.sample(key_range, num_pairs)
        vals = [random.choice(val_range) for _ in range(num_pairs)]

        for i in range(num_pairs):
            x[b, 2*i]     = keys[i]
            x[b, 2*i + 1] = vals[i]

        query_idx  = random.randint(0, num_pairs - 1)
        query_key  = keys[query_idx]
        target_val = vals[query_idx]

        x[b, sep_pos]   = 1
        x[b, query_pos] = query_key

        for i in range(num_pairs):
            y[b, 2*i]   = vals[i]
            y[b, 2*i+1] = keys[i+1] if i+1 < num_pairs else 1
        y[b, sep_pos]   = query_key
        y[b, query_pos] = target_val

    return x, y


def train():
    vocab_size = 100
    batch_size = 64
    seq_len    = 16
    iterations = 1000

    try:
        import torch_directml
        device = torch_directml.device()
        print("Using DirectML.")
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using: {device}")

    model = CobraSSM(
        vocab_size = vocab_size,
        d_model    = 128,
        n_layers   = 1,
        d_state    = 16,
        num_scales = 4,
        num_slots  = 32,
    )
    model.to(device)
    print(f"Parameters: {model.parameter_count()/1e6:.3f} M  (n_layers=1)")

    grad_log = {}
    for tag, pname in [
        ('kq', 'blocks.0.memory.kq_proj.weight'),
        ('v',  'blocks.0.memory.v_proj.weight'),
        ('fg', 'blocks.0.fusion_gate.weight'),
    ]:
        param = dict(model.named_parameters()).get(pname)
        if param is not None:
            def make_hook(t):
                def h(g): grad_log[t] = g.norm().item()
                return h
            param.register_hook(make_hook(tag))

    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01, foreach=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=3e-4)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

    num_pairs = (seq_len - 2) // 2
    sep_pos   = 2 * num_pairs
    query_pos = sep_pos + 1

    model.train()
    for i in range(iterations):
        x, y = generate_associative_recall_batch(batch_size, seq_len, vocab_size)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        grad_log.clear()
        logits, _, _ = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if i % 25 == 0:
            logits_q    = logits[:, query_pos, :]
            targets_q   = y[:, query_pos]
            preds       = logits_q.argmax(-1)
            accuracy    = (preds == targets_q).float().mean().item()
            probs       = torch.softmax(logits_q, dim=-1)
            target_prob = probs[0, targets_q[0]].item()
            top5        = torch.topk(probs[0], 5)

            with torch.no_grad():
                decay_v = model.blocks[0].memory.decay.item()
                temp_v  = model.blocks[0].memory.temp.item()

            print(
                f"Iter {i:4d} | Loss: {loss.item():.4f} | "
                f"RecallAcc: {accuracy*100:5.1f}% | GNorm: {gnorm:.3f}"
            )
            print(
                f"         Target: {targets_q[0].item()} (P={target_prob:.4f}) | "
                f"Top5: {top5.indices.tolist()} {[f'{p:.3f}' for p in top5.values.tolist()]}"
            )
            print(
                f"         decay={decay_v:.3f} temp={temp_v:.2f} | "
                f"Grads kq:{grad_log.get('kq',0):.2e} "
                f"v:{grad_log.get('v',0):.2e} "
                f"fg:{grad_log.get('fg',0):.2e}"
            )
            if torch.isnan(loss):
                print("WARNING: NaN. Stopping.")
                break


if __name__ == "__main__":
    train()