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
        input_ids : (batch, seq_len) int64 — raw token IDs, used for SEP detection
                    If None, memory is read at every position (inference fallback).
        """
        b, seq_len, d = x.shape
        residual = x

        # ── PATH 1: SSM ───────────────────────────────────────────────────────
        x_norm = self.norm_ssm(x)
        y_ssm, h_seq, ssm_state_out = self.ssm(x_norm, state=ssm_state)

        # ── Build hard read mask from token IDs ───────────────────────────────
        # read_mask[b, t] = 1.0 iff input_ids[b, t-1] == SEP_ID
        # i.e. the current position is immediately after SEP → query_pos
        if input_ids is not None:
            # Shift IDs right by 1: prev_ids[t] = input_ids[t-1]
            prev_ids = torch.zeros_like(input_ids)
            prev_ids[:, 1:] = input_ids[:, :-1]
            # (b, seq_len) float mask, 1.0 only at query_pos
            read_mask = (prev_ids == self.SEP_ID).float()
        else:
            read_mask = torch.ones(b, seq_len, device=x.device)

        # ── PATH 2: Memory ────────────────────────────────────────────────────
        y_mem_list = []
        current_M  = mem_state
        h_prev     = None

        for t in range(seq_len):
            x_t    = self.norm_mem(x[:, t, :])
            x_prev = self.norm_mem(x[:, t-1, :]) if t > 0 \
                     else torch.zeros_like(x_t)
            h_t    = h_seq[:, t]

            # Write gate (EventDetector)
            S_t    = self.event_detector(x_t, h_t, h_prev)
            h_prev = h_t.clone()
            current_M = self.memory(x_t, x_prev, h_t, S_t, current_M)

            # Hard structural read: only open at query_pos (after SEP)
            r_t     = read_mask[:, t].unsqueeze(1)          # (b, 1)
            y_raw   = self.memory.read_buffer(x_t, current_M)
            y_mem_t = r_t * y_raw                           # zero everywhere except query_pos
            y_mem_list.append(y_mem_t)

        y_mem = torch.stack(y_mem_list, dim=1)   # (b, seq_len, d)

        # ── Fusion ────────────────────────────────────────────────────────────
        g_t     = torch.sigmoid(self.fusion_gate(x_norm))
        y_block = y_ssm + g_t * self.mem_out_proj(y_mem)
        x       = residual + y_block

        # ── MLP (SwiGLU) ──────────────────────────────────────────────────────
        residual  = x
        up        = self.mlp_up(self.mlp_norm(x))
        gate, val = up.chunk(2, dim=-1)
        out       = residual + self.mlp_down(F.silu(gate) * val)

        return out, ssm_state_out, current_M