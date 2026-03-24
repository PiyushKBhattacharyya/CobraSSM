"""
Test O(1) recurrent generation via model.generate().
Verifies that:
1. Prefill (full sequence) works
2. generate() produces tokens without errors
3. Generation uses O(1) per-token recurrence (CobraCache)
"""
import torch
from cobrassm import CobraConfig, CobraForCausalLM, CobraCache

def test_generate():
    print("=== O(1) Generation Test ===")
    config = CobraConfig(vocab_size=100, d_model=64, num_hidden_layers=2, d_state=8, num_scales=2)
    device = torch.device('cpu')
    model = CobraForCausalLM(config).to(device)
    model.eval()

    input_ids = torch.randint(2, 100, (1, 16)).to(device)
    print(f"Input shape: {input_ids.shape}")

    # 1. Test prefill (full forward)
    print("\n1. Testing prefill...")
    with torch.no_grad():
        outputs = model(input_ids)
    print(f"   Logits shape: {outputs.logits.shape}")
    assert isinstance(outputs.past_key_values, CobraCache), "Cache should be CobraCache"
    print(f"   Cache type: CobraCache ✓")

    # 2. Test single-step O(1) forward
    print("\n2. Testing O(1) step...")
    cache = outputs.past_key_values
    next_token = torch.tensor([[50]]).to(device)
    with torch.no_grad():
        step_out = model(next_token, past_key_values=cache)
    print(f"   Step logits shape: {step_out.logits.shape}")
    assert step_out.logits.shape == (1, 1, 100), f"Expected (1,1,100), got {step_out.logits.shape}"
    print(f"   O(1) step ✓")

    # 3. Test HF generate()
    print("\n3. Testing model.generate()...")
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print(f"   Generated shape: {generated.shape}")
    assert generated.shape[1] == input_ids.shape[1] + 20, \
        f"Expected {input_ids.shape[1] + 20} tokens, got {generated.shape[1]}"
    print(f"   Generated tokens: {generated[0].tolist()}")
    print(f"   HF generate() ✓")

    # 4. Test training forward+backward on CPU
    print("\n4. Testing training forward+backward...")
    model.train()
    labels = input_ids.clone()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    print(f"   Loss: {loss.item():.4f}")
    loss.backward()
    print(f"   Backward ✓")

    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    test_generate()
