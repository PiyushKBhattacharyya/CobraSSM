# Goal Description

Design and implement **CobraSSM**, a novel sequence model combining a Selective State Space Model (SSM) backbone with an event-driven sparse attention mechanism ("strike mechanism") and a structured read-write memory buffer. This model is designed to maintain linear time complexity $O(n)$ while achieving strong long-context understanding, exact token recall, and improved stability over architectures like Mamba.

## User Review Required

> [!IMPORTANT]
> Please review the core mathematical formulation and the memory logic (especially the read/write mechanism for the structured memory) below. If the proposed event detector design aligns with your vision, we can proceed to PyTorch implementation.

## Proposed Architecture Design

### 1. Multi-Scale Selective SSM Backbone
Maintains continuous hidden states with input-dependent parameterization. Uses multiple state spaces with different decay rates to significantly improve over baseline single Mamba models.
- **Formulation (Multi-Scale)**:
  - Multiple components for $A$, each initialized with different timescale priors.
  - $\Delta_t = \text{softplus}(\text{Linear}(x_t))$
  - $\bar{A}_k = \exp(\Delta_t A_k)$ (for each scale $k$)
  - $\bar{B}_k = \Delta_t B_k(x_t)$
  - $h_{t,k} = \bar{A}_k h_{t-1,k} + \bar{B}_k x_t$
  - $y_t^{ssm} = \sum_k C_k(x_t) h_{t,k}$

### 2. Event-Driven Sparse Attention ("Strike Mechanism")
Dynamically detects important tokens to trigger an attention read, avoiding $O(n^2)$ global attention.
- **Event Detector**: An improved gating module that incorporates the latest SSM hidden state $h_t$ alongside input $x_t$ (and potentially an entropy/surprise signal) to compute an importance score $S_t \in [0, 1]$.
- **Trigger**: The primary mechanism is **soft gating** during training, where memory output is modulated smoothly by $S_t$. During inference, an optional top-k threshold can be applied for efficiency.
- **Sparse Attention**: When triggered, the current token generates queries to read from a structured differentiable memory buffer. To ensure stability, **RMSNorm** is applied before the Query, Key, and Value projections.

### 3. Differentiable Structured Memory Module
A bounded, fully differentiable Key-Value (KV) memory buffer of size $M \ll N$ with time encodings.
- **Write Operation (Soft Probability + Slot Attention/Overwrite)**: Replaces hard LRU with an attention-weighted overwrite mechanism (or slot attention). Entries are updated smoothly with soft write probability modulated by $S_t$. Old memories naturally decay.
- **Temporal Bias**: Incorporates positional decay or relative time encoding into the memory keys/values so that the attention mechanism does not treat all entries equally but has awareness of recency.
- **Read Operation**: Query $Q_t$ attends over the buffer $K$. $y_t^{mem} = \text{Softmax}(Q_t K^T / \sqrt{d} + M_{\text{pos}}) V$, where $M_{\text{pos}}$ is a temporal decay bias.

### 4. Dual-Path Processing and Residual Stability
Unifies continuous representation and exact-recall symbolic logic.
- **Path 1 (Continuous Flow)**: Returns $y_t^{ssm}$, modeling local interactions spanning multiple scales.
- **Path 2 (Symbolic Flow)**: Returns $y_t^{mem}$, retrieving specific exact tokens from the memory buffer.
- **Fusion and Residual Path**: To ensure deep network stability, the block output is strictly residual: $y_t^{out} = x_t + \text{CobraBlock}(x_t)$. Inside the block, the paths merge as $y_{block} = y_t^{ssm} + g_t \odot \text{Linear}(y_t^{mem})$.

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
