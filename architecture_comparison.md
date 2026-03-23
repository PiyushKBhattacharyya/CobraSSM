# Architecture Comparison: CobraSSM vs. Mamba vs. Transformers

| Feature | Transformer (GPT) | Mamba (S6 SSM) | **CobraSSM (Hybrid)** |
| :--- | :--- | :--- | :--- |
| **Inference Scaling** | $O(L^2)$ - Quadratic | $O(L)$ - Linear | **$O(L)$ - Linear** |
| **Memory State** | Grows with $L$ (KV Cache) | Fixed $\approx 2048$ dims | **Fixed SSM + Bounded KV Memory** |
| **Recall Accuracy** | Exact (via attention) | Lossy (via compression) | **Exact (via Strike Mechanism)** |
| **Training Speed** | FlashAttention (Fused) | Parallel Scan (Triton) | **Vectorized Scan (Pure PyTorch)** |
| **Structural Bias** | Relies on Position Embeds | Causal Sequence-Aware | **Multi-Scale Time Pointers** |
| **Retrieval Power** | Global | Limited to Recent Context | **Gated Long-Term Slot Memory** |

## Key Differentiator
While Mamba struggles with "exact token recall" (associative recall) due to its compression into a hidden state, **CobraSSM** uses a **Differentiable Memory Buffer** to store symbolic associations. This gives it Transformer-like recall accuracy without the $O(L^2)$ memory cost.
