import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableMemoryBuffer(nn.Module):
    """
    Bounded differentiable KV memory buffer.

    Key architectural fix vs previous version
    -----------------------------------------
    OLD (broken for associative recall):
        fused = x_t + h_context
        k_t   = k_proj(fused)     # key   mixes current token INTO key
        v_t   = v_proj(fused)     # value mixes hidden state INTO value
        query = norm(x_t)         # read query uses raw token embedding

    Result: at K_i position the slot key ≈ K_i but slot value ≈ K_i (not V_i).
            at V_i position the slot value ≈ V_i but slot key ≈ V_i (not K_i).
            Query at Q_K matches the K_i slot but retrieves K_i info, not V_i.

    NEW (correct K→V binding):
        k_t   = k_proj(h_context)   # key   from SSM state only
        v_t   = v_proj(x_t)         # value from current token only
        query = state_proj(h_t_summary)   passed in from CobraBlock

    Why this works:
        At V_i position (t = 2i+1):
            h_context = state_proj(h_t.mean(scales, d_state)) encodes K_i via SSM recurrence
            x_t = V_i embedding
        → slot key   encodes K_i ✓
        → slot value encodes V_i ✓

        At query_pos with token Q_K (= K_j for some j):
            h_context encodes Q_K (SSM just processed Q_K)
            read query = state_proj(h_t) ≈ h_context at K_j position
        → dot(query, key_from_Kj_position) is high ✓  → retrieves V_j ✓

    h aggregation: mean over (scales, d_state) dims → (b, d_model)
    state_proj: Linear(d_model, d_model)  [was Linear(d_state, d_model)]
    """

    def __init__(self, d_model, d_state=16, num_slots=64):
        super().__init__()
        self.d_model   = d_model
        self.d_state   = d_state
        self.num_slots = num_slots

        # Fixed slot centroid keys for routing writes
        self.slot_keys = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)

        # h aggregation: (b, d_model, d_state) → mean over d_state → (b, d_model)
        # then project to d_model
        self.state_proj = nn.Linear(d_model, d_model, bias=False)

        # key comes from hidden state; value comes from current token
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def _summarise_state(self, h_t):
        """
        h_t: (b, num_scales, d_model, d_state)
        Returns h_context: (b, d_model)
        """
        # Mean over scales (dim 1) and d_state (dim 3) → (b, d_model)
        h_mean = h_t.mean(dim=(1, 3))          # (b, d_model)
        return self.state_proj(h_mean)          # (b, d_model)

    def forward(self, x_t, h_t, S_t, memory_state=None):
        """
        Write operation.
        x_t  : (b, d_model)  — current token (becomes the VALUE)
        h_t  : (b, num_scales, d_model, d_state)  — SSM state (becomes the KEY)
        S_t  : (b, 1)  — soft write gate from EventDetector
        """
        b = x_t.size(0)

        h_context = self._summarise_state(h_t)   # (b, d_model)

        # Key from SSM state (encodes the PREVIOUS token context, i.e. K_i at V_i pos)
        k_t = self.k_proj(h_context).unsqueeze(1)   # (b, 1, d_model)
        # Value from current token (the actual V_i token embedding)
        v_t = self.v_proj(x_t).unsqueeze(1)         # (b, 1, d_model)

        if memory_state is None:
            mem_k = torch.zeros(b, self.num_slots, self.d_model, device=x_t.device)
            mem_v = torch.zeros(b, self.num_slots, self.d_model, device=x_t.device)
        else:
            mem_k, mem_v = memory_state

        # Route to slots via fixed centroid keys
        write_logits  = torch.matmul(k_t, self.slot_keys.transpose(-1, -2)) / (self.d_model ** 0.5)
        write_weights = torch.softmax(write_logits, dim=-1)        # (b, 1, num_slots)

        write_strength = write_weights.transpose(-1, -2) * S_t.unsqueeze(-1)  # (b, num_slots, 1)
        decay_factor   = 1.0 - write_strength

        updated_mem_k = mem_k * decay_factor + k_t * write_strength
        updated_mem_v = mem_v * decay_factor + v_t * write_strength

        return (updated_mem_k, updated_mem_v)

    def read_buffer(self, q_h, memory_state):
        """
        Read operation.
        q_h : (b, d_model)  — query derived from hidden state (NOT raw token embedding)
        """
        mem_k, mem_v = memory_state
        q = q_h.unsqueeze(1)   # (b, 1, d_model)

        attn_logits  = torch.matmul(q, mem_k.transpose(-1, -2)) / (self.d_model ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        read_out     = torch.matmul(attn_weights, mem_v)   # (b, 1, d_model)

        return read_out.squeeze(1)   # (b, d_model)