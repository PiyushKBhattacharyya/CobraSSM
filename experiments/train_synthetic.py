import torch
import torch.nn as nn
import torch.optim as optim
from cobrassm import CobraSSM
import random


def generate_associative_recall_batch(batch_size, seq_len, vocab_size):
    """
    Format: K1 V1 K2 V2 ... Kn Vn SEP Q_K
                                        ^--- model must emit Answer_V here

    seq_len = 2*num_pairs + 2   (even number, e.g. 16 -> 7 pairs)

    The sequence ends at Q_K. There is NO ans_pos token in x.
    This eliminates the token-0 shortcut: the model cannot peek at
    what comes after query_pos because nothing does.

    Vocab:
        0        : unused entirely (never in x, never in y)
        1        : SEP
        10–49    : keys
        51–90    : values

    Labels y (next-token prediction, causal):
        pos 0 (K1)        -> y = V1
        pos 1 (V1)        -> y = K2
        ...
        pos 2n-2 (Kn)     -> y = Vn
        pos 2n-1 (Vn)     -> y = SEP (token 1)   <-- completes the chain
        pos 2n   (SEP)    -> y = Q_K             <-- predicts the query key
        pos 2n+1 (Q_K)    -> y = Answer_V        <-- the main recall target
    """
    key_range = list(range(10, 50))
    val_range = list(range(51, 91))

    # seq_len = 2*num_pairs + 2  ->  num_pairs = (seq_len - 2) // 2
    assert (seq_len - 2) % 2 == 0, "seq_len must be even (= 2*num_pairs + 2)"
    num_pairs = (seq_len - 2) // 2
    assert num_pairs <= len(key_range), "too many pairs for key vocab"

    sep_pos   = 2 * num_pairs        # e.g. 14 for seq_len=16
    query_pos = sep_pos + 1          # 15  — last position in x

    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    for b in range(batch_size):
        keys = random.sample(key_range, num_pairs)
        vals = [random.choice(val_range) for _ in range(num_pairs)]

        # Fill K-V pairs
        for i in range(num_pairs):
            x[b, 2 * i]     = keys[i]
            x[b, 2 * i + 1] = vals[i]

        # Pick recall target
        query_idx  = random.randint(0, num_pairs - 1)
        query_key  = keys[query_idx]
        target_val = vals[query_idx]

        x[b, sep_pos]   = 1           # SEP
        x[b, query_pos] = query_key   # Q_K  — last token

        # Aux labels: complete next-token chain through the K-V pairs and SEP
        for i in range(num_pairs):
            y[b, 2 * i]     = vals[i]                          # Ki -> Vi
            y[b, 2 * i + 1] = keys[i + 1] if i + 1 < num_pairs else 1  # Vi -> K(i+1) or SEP

        y[b, sep_pos]   = query_key   # SEP -> Q_K
        y[b, query_pos] = target_val  # Q_K -> Answer_V  (main recall target)

    return x, y


def train():
    vocab_size = 100
    batch_size = 64
    seq_len    = 16   # even: 7 pairs + SEP + Q_K, no ans_pos token
    iterations = 1000

    try:
        import torch_directml
        device = torch_directml.device()
        print("Using DirectML (AMD GPU).")
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using: {device}")

    model = CobraSSM(
        vocab_size = vocab_size,
        d_model    = 128,
        n_layers   = 4,
        d_state    = 16,
        num_scales = 4,
        num_slots  = 16,
    )
    model.to(device)
    print(f"Parameters: {model.parameter_count()/1e6:.2f} M")

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
        logits, _, _ = model(x)

        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if i % 25 == 0:
            logits_q  = logits[:, query_pos, :]
            targets_q = y[:, query_pos]

            preds    = logits_q.argmax(dim=-1)
            accuracy = (preds == targets_q).float().mean().item()

            probs       = torch.softmax(logits_q, dim=-1)
            target_prob = probs[0, targets_q[0]].item()
            top5        = torch.topk(probs[0], 5)
            zero_pct    = (preds == 0).float().mean().item()

            # Also report aux-task accuracy (predicting next key/val in pairs)
            logits_aux  = logits[:, :sep_pos, :]             # (b, 2n, vocab)
            targets_aux = y[:, :sep_pos]
            valid_aux   = targets_aux != -100
            if valid_aux.any():
                preds_aux  = logits_aux.argmax(-1)
                acc_aux    = (preds_aux[valid_aux] == targets_aux[valid_aux]).float().mean().item()
            else:
                acc_aux = 0.0

            print(
                f"Iter {i:4d} | Loss: {loss.item():.4f} | "
                f"RecallAcc: {accuracy*100:5.1f}% | "
                f"AuxAcc: {acc_aux*100:5.1f}% | "
                f"GNorm: {gnorm:.3f} | Tok0%: {zero_pct*100:.0f}%"
            )
            print(
                f"         Target: {targets_q[0].item()} (P={target_prob:.4f}) | "
                f"Top5: {top5.indices.tolist()} "
                f"{[f'{p:.3f}' for p in top5.values.tolist()]}"
            )

            if torch.isnan(loss):
                print("WARNING: NaN loss. Stopping.")
                break


if __name__ == "__main__":
    train()