import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableMemoryBuffer(nn.Module):
    """
    Linear Associative Memory with tied key/query projection.
    Stable recurrent version.
    """

    def __init__(self, d_model, d_state=16, num_slots=64):
        super().__init__()
        self.d_model = d_model

        self.kq_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)

        self.log_temp    = nn.Parameter(torch.tensor(1.386))  # exp(1.386) ≈ 4.0
        self.decay_logit = nn.Parameter(torch.tensor(3.0))    # sigmoid(3.0) ≈ 0.952

        self.state_proj = nn.Linear(d_model, d_model, bias=False)

    @property
    def temp(self):
        # Explicitly cast to half-precision if necessary to avoid MPS mismatches
        return torch.exp(self.log_temp).to(dtype=self.kq_proj.weight.dtype)

    @property
    def decay(self):
        # Explicitly cast to half-precision if necessary to avoid MPS mismatches
        return torch.sigmoid(self.decay_logit).to(dtype=self.kq_proj.weight.dtype)

    def forward_batch(self, x, x_prev, S, memory_init=None):
        b, seq_len, d = x.shape
        device = x.device
        dtype = x.dtype
        
        k_all = F.normalize(self.kq_proj(x_prev), dim=-1)
        v_all = self.v_proj(x)
        decay = self.decay
        
        current_M = memory_init if memory_init is not None else \
                    torch.zeros(b, d, d, device=device, dtype=dtype)
        
        M_seq_list = []
        for t in range(seq_len):
            k = k_all[:, t]
            v = v_all[:, t]
            s = S[:, t]
            
            # Write: v ⊗ k
            write = s.unsqueeze(-1) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            current_M = decay * current_M + write
            M_seq_list.append(current_M)
            
        M_seq = torch.stack(M_seq_list, dim=1)
        return M_seq, current_M

    def forward_step(self, x_t, x_prev, S_t, memory_state):
        """
        Single step forward for O(1) generation.
        x_t, x_prev: (b, d)
        S_t: (b, 1)
        memory_state: (b, d, d)
        """
        k = F.normalize(self.kq_proj(x_prev), dim=-1)
        v = self.v_proj(x_t)
        write = S_t.unsqueeze(-1) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        new_M = self.decay * memory_state + write
        return new_M

    def forward(self, x_t, x_prev, h_t, S_t, memory_state=None):
        """Legacy single step forward."""
        return self.forward_step(x_t, x_prev, S_t, memory_state)

    def read_buffer_batch(self, x, M_seq):
        q = F.normalize(self.kq_proj(x), dim=-1)
        temp_q = (self.temp * q).unsqueeze(-1)
        return torch.matmul(M_seq, temp_q).squeeze(-1)

    def read_buffer(self, x_t, memory_state):
        q = F.normalize(self.kq_proj(x_t), dim=-1)
        return torch.bmm(memory_state, (self.temp * q).unsqueeze(2)).squeeze(2)

    def _summarise_state(self, h_t):
        return self.state_proj(h_t.mean(dim=(1, 3)))