"""Test save/load roundtrip."""
import torch
from cobrassm import CobraConfig, CobraForCausalLM

c = CobraConfig(vocab_size=100, d_model=64, num_hidden_layers=1)
m = CobraForCausalLM(c)
m.save_pretrained('./test_save')
print("Save ✓")

m2 = CobraForCausalLM.from_pretrained('./test_save')
print("Load ✓")

ids = torch.randint(0, 100, (1, 8))
o1 = m(ids)
o2 = m2(ids)
diff = (o1.logits - o2.logits).abs().max().item()
print(f"Max diff: {diff:.8f}")
assert diff < 1e-5, f"Mismatch: {diff}"
print("Stage 1 PASSED ✓")
