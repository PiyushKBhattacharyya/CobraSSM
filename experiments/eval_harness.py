"""
Lightweight LM Evaluation Harness wrapper for CobraSSM.
Computes per-token log-likelihoods for multiple-choice benchmarks.

Usage:
    python experiments/eval_harness.py
"""
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from cobrassm import CobraConfig, CobraForCausalLM


def compute_loglikelihood(model, tokenizer, context, continuation, device):
    """
    Compute log-likelihood of continuation given context.
    Returns: (total_log_prob, num_tokens)
    """
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    
    full_ids = torch.tensor([ctx_ids + cont_ids], device=device)
    
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # (1, L, V)
    
    # Get log-probs for the continuation tokens only
    # logits[:, t] predicts token t+1
    cont_start = len(ctx_ids) - 1  # -1 because logits[t] predicts t+1
    cont_logits = logits[0, cont_start:cont_start + len(cont_ids)]  # (cont_len, V)
    
    log_probs = F.log_softmax(cont_logits, dim=-1)
    cont_tensor = torch.tensor(cont_ids, device=device)
    
    token_log_probs = log_probs[range(len(cont_ids)), cont_tensor]
    total_ll = token_log_probs.sum().item()
    
    return total_ll, len(cont_ids)


def eval_hellaswag_sample(model, tokenizer, device):
    """
    Evaluate on a few HellaSwag-style examples.
    Each example has a context and 4 possible continuations.
    """
    examples = [
        {
            "context": "A woman is standing in a kitchen. She picks up a knife and",
            "choices": [
                " begins to chop vegetables on the cutting board.",
                " throws it at the wall and laughs loudly.",
                " eats the knife whole without any hesitation.",
                " the knife transforms into a beautiful flower."
            ],
            "label": 0
        },
        {
            "context": "The football player catches the ball and",
            "choices": [
                " runs toward the end zone for a touchdown.",
                " starts reading a book on the field.",
                " dissolves into thin air immediately.",
                " plants the ball in the ground like a seed."
            ],
            "label": 0
        },
        {
            "context": "A student opens a textbook and",
            "choices": [
                " begins reading the chapter on mathematics.",
                " eats every page one by one.",
                " the textbook starts flying around the room.",
                " discovers a portal to another dimension."
            ],
            "label": 0
        },
    ]
    
    correct = 0
    total = len(examples)
    
    for i, ex in enumerate(examples):
        scores = []
        for choice in ex["choices"]:
            ll, n_tokens = compute_loglikelihood(model, tokenizer, ex["context"], choice, device)
            scores.append(ll / n_tokens)  # Length-normalized
        
        pred = max(range(len(scores)), key=lambda j: scores[j])
        is_correct = pred == ex["label"]
        correct += int(is_correct)
        
        print(f"  Example {i+1}: pred={pred} label={ex['label']} {'✓' if is_correct else '✗'}")
        for j, (choice, score) in enumerate(zip(ex["choices"], scores)):
            marker = " ←" if j == pred else ""
            print(f"    [{j}] {score:.4f} {choice[:50]}...{marker}")
    
    return correct, total


def main():
    print("=" * 60)
    print("  CobraSSM: LM Evaluation Harness")
    print("=" * 60)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    config = CobraConfig(
        vocab_size=len(tokenizer),
        d_model=128,
        num_hidden_layers=4,
        d_state=16,
        num_scales=4
    )
    
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
    print(f"NOTE: This is an UNTRAINED model. Scores reflect random baseline.\n")
    
    # HellaSwag-style evaluation
    print("--- HellaSwag-style Multiple Choice ---")
    correct, total = eval_hellaswag_sample(model, tokenizer, device)
    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"Random baseline: 25.0%")
    
    print("\n" + "=" * 60)
    print("  To evaluate on real benchmarks, train the model first,")
    print("  then run: lm_eval --model hf --model_args pretrained=./cobra_trained")
    print("=" * 60)


if __name__ == "__main__":
    main()
