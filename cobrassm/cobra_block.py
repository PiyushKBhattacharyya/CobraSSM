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
    CobraBlock with hard structural read gate.

    Why learned gates keep failing
    --------------------------------
    Every learned gate we tried (FusionGate, ReadGate(x_t), ReadGate(x_t, x_prev))
    gets trained by the same loss that supervises all positions. At V_i positions the
    memory already contains useful K→V associations, so the aux-task gradient rewards
    opening the gate there too. The gate learns to open everywhere useful — which is
    correct from a loss perspective but wrong for our architectural intent.

    The rule "only read from memory after SEP" is a hard structural fact about the
    sequence format, not something that needs to be learned. SEP (token id=1) appears
    exactly once per sequence, at sep_pos. The position after it is query_pos.

    Fix: track seen_sep from raw token IDs (available in the block via input_ids arg).
    At each step t:
        seen_sep[:, t] = (input_ids[:, t-1] == SEP_ID)   # True only at query_pos
        read_mask = seen_sep  → (b, 1), float 0 or 1

    This is a hard binary mask, not a sigmoid. No gradient flows through it.
    The memory path still has gradients via kq_proj, v_proj, mem_out_proj.
    Only the ON/OFF routing is structural.

    The block receives input_ids as an optional argument. When not provided
    (e.g. at inference with streaming), it falls back to a uniform-open gate.
    """

    SEP_ID = 1

    def __init__(self, d_model, d_state=16, num_scales=4, num_slots=64):
        super().__init__()
        self.d_model = d_model

        self.norm_ssm = RMSNorm(d_model)
        self.ssm      = MultiScaleSSM(d_model, d_state, num_scales)

        self.event_detector = EventDetector(d_model, d_state, num_scales)
        self.memory         = DifferentiableMemoryBuffer(d_model, d_state, num_slots)

        self.norm_mem = RMSNorm(d_model)

        # Fusion gate (still learned — merges SSM and memory outputs)
        self.fusion_gate  = nn.Linear(d_model, d_model)
        nn.init.constant_(self.fusion_gate.bias, 1.0)
        self.mem_out_proj = nn.Linear(d_model, d_model, bias=False)

        # MLP (SwiGLU)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp_up   = nn.Linear(d_model, d_model * 4 * 2)
        self.mlp_down = nn.Linear(d_model * 4, d_model)

    def forward(self, x, ssm_state=None, mem_state=None, input_ids=None):
        """
        x         : (batch, seq_len, d_model)
        input_ids : (batch, seq_len) int64
        """
        b, seq_len, d = x.shape
        x_norm = self.norm_ssm(x)

        # 1. Multi-Scale SSM path (Vectorized)
        y_ssm, ssm_seq, next_ssm_state = self.ssm(x_norm, state=ssm_state)

        # 2. Event Detection (Full sequence)
        S = self.event_detector.forward_sequence(x_norm, ssm_seq, input_ids)

        # 3. Differentiable Memory Path (Vectorized)
        # Shifted input for key generation
        x_prev = torch.cat([
            torch.zeros(b, 1, d, device=x.device, dtype=x.dtype),
            self.norm_mem(x[:, :-1, :])
        ], dim=1)
        
        x_mem_in = self.norm_mem(x)
        M_seq, next_mem_state = self.memory.forward_batch(x_mem_in, x_prev, S, memory_init=mem_state)

        # 4. Gated Read
        y_mem_raw = self.memory.read_buffer_batch(x_mem_in, M_seq)
        
        # Hard structural read gating (SEP)
        if input_ids is not None:
            # Shift IDs right to trigger after SEP
            prev_ids = torch.zeros_like(input_ids)
            prev_ids[:, 1:] = input_ids[:, :-1]
            read_mask = (prev_ids == self.SEP_ID).float().unsqueeze(-1)
            y_mem = y_mem_raw * read_mask
        else:
            y_mem = y_mem_raw

        # 5. Fusion
        g_t     = torch.sigmoid(self.fusion_gate(x_norm))
        y_block = y_ssm + g_t * self.mem_out_proj(y_mem)
        x       = x + y_block

        # 6. MLP (SwiGLU)
        residual  = x
        up        = self.mlp_up(self.mlp_norm(x))
        gate, val = up.chunk(2, dim=-1)
        out       = residual + self.mlp_down(F.silu(gate) * val)

        return out, next_ssm_state, next_mem_state