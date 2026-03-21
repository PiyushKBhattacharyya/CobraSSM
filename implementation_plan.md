# Goal Description

Design and implement **CobraSSM**, a novel sequence model combining a Selective State Space Model (SSM) backbone with an event-driven sparse attention mechanism ("strike mechanism") and a structured read-write memory buffer. This model is designed to maintain linear time complexity $O(n)$ while achieving strong long-context understanding, exact token recall, and improved stability over architectures like Mamba.

## User Review Required

> [!IMPORTANT]
> Please review the core mathematical formulation and the memory logic (especially the read/write mechanism for the structured memory) below. If the proposed event detector design aligns with your vision, we can proceed to PyTorch implementation.

## Proposed Architecture Design

### 1. Multi-Scale Selective SSM Backbone
- **Low-Rank Selective B**: To prevent projection dimension blowup, $B_t$ is implemented as a low-rank mixing of an input-dependent vector and a learned per-feature matrix ($B_{mix}$).
- **Principled A-scaling**: $A$ matrices are initialized with evenly spaced log-decays to capture multi-scale temporal dependencies.
- **Mamba-style Skip**: Includes a learned $D$ skip connection for direct signal propagation.

### 2. Event-Driven Strike Mechanism (Refined)
- **Hard Structural Read Gate**: Instead of a purely learned gate, the model uses a hard structural gate triggered by the `SEP` token (ID=1). Memory reads are only enabled at the position immediately following the `SEP` token (the query position).
- **Strike Trigger**: The `EventDetector` still provides a soft write-strength $S_t$ to control which information enters the memory.

### 3. Linear Associative Memory Module
- **Formulation**: Replaced slot attention with a Linear Associative Memory ($M_t = \lambda M_{t-1} + S_t (v_t \otimes k_t)$).
- **Tied Key-Query**: Uses a unified `kq_proj` with normalization to ensure stable semantic matching.
- **Learned Dynamics**: Incorporates learned `decay` (recency bias) and `temperature` (sharpness) parameters.

### 4. Dual-Path Fusion and Residual Stability
- **Fusion Logic**: $y_{block} = y^{ssm} + \sigma(\text{FusionGate}) \odot \text{Linear}(y^{mem})$.
- **Residual Path**: Retains the strictly residual $x + \text{Block}(x)$ design for depth stability.

## PyTorch Implementation Strategy

We will structure the PyTorch implementation into modular components under `d:\Projects\CobraSSM`.

### Components
- **[selective_scan.py](file:///d:/Projects/CobraSSM/selective_scan.py)**: Contains the efficient parallel scan or chunk-wise recurrent mechanism for the SSM.
- **[event_detector.py](file:///d:/Projects/CobraSSM/event_detector.py)**: Implements the lightweight scoring function and the trigger logic for read/write.
- **[memory_buffer.py](file:///d:/Projects/CobraSSM/memory_buffer.py)**: Manages the bounded KV buffer states across steps/chunks.
- **`strike_attention.py`**: Handles the cross-attention between token queries and the memory buffer.
- **[cobra_block.py](file:///d:/Projects/CobraSSM/cobra_block.py)**: Combines [SSM](file:///d:/Projects/CobraSSM/cobrassm/model.py#5-65), `Memory`, `Attention`, and `EventDetector` into a unified `CobraSSMBlock`. Incorporates SwiGLU feed-forward networks (if utilized) and RMSNorm.
- **[model.py](file:///d:/Projects/CobraSSM/model.py)**: The overarching language model class containing embeddings, $L$ blocks, and the language modeling head.

### Constraints & Considerations
- **$O(n)$ Complexity**: By restricting attention to a fixed-size buffer ($M$), the attention cost per token is $O(M)$, making the total process $O(nM)$ (effectively linear W.R.T sequence length).
- **Batch Processing**: The event detection will be vectorized. Memory buffer updates will use mask-based scattering for batch friendly operations.

## Training Strategy
- **Initialization**: 
  - $A$ matrix initialized mathematically (e.g., HiPPO diagonal or purely real negative diagonal).
  - Projections initialized to near-identity to ease training early on.
- **Loss**: Standard Cross-Entropy for language modeling. We can also add an auxiliary sparsity loss on $S_t$ to encourage the model not to exceed the buffer's capacity.
- **Stabilization**: Liberal use of RMSNorm pre-attention and pre-SSM. Memory fusion will be gated to zero at initialization so the model relies on the stable SSM path first.
- **Curriculum Learning**: Train on short sequences (e.g., 512, 1024 tokens) to develop local syntax via the SSM, then increase sequence length to 8k+ to force the model to rely on the memory buffer for long-range recall tasks.

## Evaluation Plan

### Baselines
- Transformer Baseline (Standard Causal self-attention, e.g., GPT-2/Llama architecture)
- Mamba Backbone Baseline (Pure S6 architecture)

### Tasks
1. **Long Sequence Modeling**: Perplexity drops on long-context datasets (e.g., PG-19 or extended language modeling datasets).
2. **Copy / Induction Tasks**: Synthetic tasks requiring exact token recall (e.g., synthetic associative recall masks, repeating sequences). This specifically stresses the memory limit of pure SSMs.
3. **Standard Language Modeling**: Zero-shot evaluations to ensure foundational reasoning has not degraded.

---

## Phase 2: Optimization and Validation
Building upon the completed Phase 1 PyTorch architecture, we now focus on training optimization, hardware throughput, and ecosystem integration.

### 1. Synthetic Associative Recall Benchmark
Before large-scale training, we must empirically validate the Differentiable Memory Buffer's ability to solve the exact-token recall problem.
- **Task Design**: Implement a synthetic "Copy Task" dataset (e.g. sequence of random key-value mappings, followed by a query key).
- **Execution Strategy**: Develop a lightweight `train_synthetic.py` script. Compare CobraSSM against a standard Transformer and a baseline Mamba model.
- **Success Criteria**: CobraSSM achieves near 100% accuracy on this task across long context windows (e.g., up to 4k tokens).

### 2. Triton Associative Scan Kernel
The current [selective_scan.py](file:///d:/Projects/CobraSSM/selective_scan.py) uses a standard PyTorch [for](file:///d:/Projects/CobraSSM/cobrassm/cobra_block.py#46-106) loop. We must write a custom parallel kernel for true $O(n)$ hardware speed.
- **Design**: Implement a hardware-aware parallel associative scan in Triton for the Multi-Scale SSM path. Standard Mamba utilizes a similar approach to avoid strict sequential computational bottlenecks.
- **Implementation Location**: Create `cobrassm/ops/triton_scan.py`.
- **Integration**: Update `MultiScaleSSM.forward()` to route to the compiled Triton kernel automatically.

### 3. Hugging Face Ecosystem Integration
To ensure immediate usability and simplified evaluation.
- **Configuration**: Create `CobraConfig` inheriting from `PretrainedConfig`.
- **Model Wrapper**: Create `CobraForCausalLM` inheriting from `PreTrainedModel`.
- **Testing**: Ensure model weights can be saved and reloaded using the standard `.save_pretrained()` and `AutoModelForCausalLM.from_pretrained()` API paradigm.
