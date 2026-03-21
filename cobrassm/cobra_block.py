import torch
import torch.nn as nn
import torch.nn.functional as F
from .selective_scan import MultiScaleSSM
from .event_detector import EventDetector
from .memory_buffer import DifferentiableMemoryBuffer


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm.type_as(x) * self.weight


class CobraBlock(nn.Module):
    """
    Integrates the Continuous SSM path and the Symbolic Memory path.

    Memory read/write change (matches new memory_buffer.py):
      - Write: key = f(h_t),  value = f(x_t)   [was both from fused x_t+h]
      - Read:  query = memory.state_proj(h_t summary)  [was norm(x_t)]

    This is required for associative recall: the hidden state at a value
    position V_i encodes the preceding key K_i, so writing key=h(V_i pos)
    creates a slot that can be retrieved by querying with h(Q_K pos) when
    Q_K matches K_i.

    norm_mem_q is removed — the read query is now derived from h_t directly
    via memory.state_proj (shared with the write path for consistent geometry).
    """

    def __init__(self, d_model, d_state=16, num_scales=4, num_slots=64):
        super().__init__()
        self.d_model = d_model

        self.norm_ssm = RMSNorm(d_model)
        self.ssm      = MultiScaleSSM(d_model, d_state, num_scales)

        self.event_detector = EventDetector(d_model, d_state, num_scales)
        self.memory         = DifferentiableMemoryBuffer(d_model, d_state, num_slots)

        # norm for KV write path only (query path now uses h_t directly)
        self.norm_mem_kv = RMSNorm(d_model)

        # Fusion gate
        self.fusion_gate = nn.Linear(d_model, d_model)
        nn.init.constant_(self.fusion_gate.bias, 1.0)
        self.mem_out_proj = nn.Linear(d_model, d_model, bias=False)

        # MLP (SwiGLU)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp_up   = nn.Linear(d_model, d_model * 4 * 2)
        self.mlp_down = nn.Linear(d_model * 4, d_model)

    def forward(self, x, ssm_state=None, mem_state=None):
        """
        x   : (batch, seq_len, d_model)
        Returns: out, ssm_state_out, mem_state_out
        """
        b, seq_len, d = x.shape
        residual = x

        # ── PATH 1: Continuous SSM ────────────────────────────────────────────
        x_norm = self.norm_ssm(x)
        y_ssm, h_seq, ssm_state_out = self.ssm(x_norm, state=ssm_state)

        # ── PATH 2: Symbolic Memory ───────────────────────────────────────────
        y_mem_list        = []
        current_mem_state = mem_state
        h_prev            = None

        for t in range(seq_len):
            x_t = x_norm[:, t, :]     # (b, d_model)
            h_t = h_seq[:, t]         # (b, num_scales, d_model, d_state)

            # 1. Event detection: soft importance score S_t ∈ [0,1]
            S_t = self.event_detector(x_t, h_t, h_prev)
            h_prev = h_t.clone()

            # 2. Write: key = f(h_t), value = f(x_t)
            #    norm is applied to x_t (the value source) before writing
            x_t_kv = self.norm_mem_kv(x_t)
            current_mem_state = self.memory(x_t_kv, h_t, S_t, current_mem_state)

            # 3. Read: query derived from hidden state h_t (same space as write keys)
            #    memory.state_proj maps h_t summary → d_model, matching write key geometry
            q_h = self.memory._summarise_state(h_t)   # (b, d_model)
            y_mem_t = self.memory.read_buffer(q_h, current_mem_state)

            y_mem_list.append(y_mem_t)

        y_mem = torch.stack(y_mem_list, dim=1)   # (b, seq_len, d_model)

        # ── FUSION ────────────────────────────────────────────────────────────
        g_t    = torch.sigmoid(self.fusion_gate(x_norm))
        y_block = y_ssm + g_t * self.mem_out_proj(y_mem)

        # Residual 1
        x = residual + y_block

        # ── MLP (SwiGLU) ──────────────────────────────────────────────────────
        residual    = x
        x_mlp_norm  = self.mlp_norm(x)
        up          = self.mlp_up(x_mlp_norm)
        gate, val   = up.chunk(2, dim=-1)
        mlp_out     = self.mlp_down(F.silu(gate) * val)

        # Residual 2
        out = residual + mlp_out
        return out, ssm_state_out, current_mem_state