import torch
import torch.nn as nn
import torch.nn.functional as F
from .hybrid_utils import hybrid_apply


class DifferentiableMemoryBuffer(nn.Module):
    """
    Linear Associative Memory with tied key/query projection.
    Supports Hybrid GPU-Forward/CPU-Backward for DirectML stability.
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
        return torch.exp(self.log_temp)

    @property
    def decay(self):
        return torch.sigmoid(self.decay_logit)


    def _core_forward_batch(self, *args):
        """Pure functional forward for Hybrid Autograd."""
        # args: x, x_prev, S, kq_weight, v_weight, log_temp, decay_logit, memory_init
        x, x_prev, S = args[0], args[1], args[2]
        kq_weight, v_weight = args[3], args[4]
        log_temp, decay_logit = args[5], args[6]
        memory_init = args[7]
        
        b, seq_len, d = x.shape
        device = x.device
        dtype = x.dtype
        
        k_all = F.normalize(F.linear(x_prev, kq_weight), dim=-1)
        v_all = F.linear(x, v_weight)
        decay = torch.sigmoid(decay_logit)
        
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

    def forward_batch(self, x, x_prev, S, memory_init=None):
        params = [self.kq_proj.weight, self.v_proj.weight, self.log_temp, self.decay_logit]
        
        if x.device.type == 'privateuseone' and x.requires_grad:
             return hybrid_apply(self._core_forward_batch, params, x, x_prev, S, *params, memory_init)
        else:
             return self._core_forward_batch(x, x_prev, S, *params, memory_init)

    def forward(self, x_t, x_prev, h_t, S_t, memory_state=None):
        """Single step forward (Inference)."""
        k = F.normalize(self.kq_proj(x_prev), dim=-1)
        v = self.v_proj(x_t)
        write = S_t.unsqueeze(-1) * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        return self.decay * memory_state + write

    def read_buffer_batch(self, x, M_seq):
        q = F.normalize(self.kq_proj(x), dim=-1)
        temp_q = (self.temp * q).unsqueeze(-1)
        return torch.matmul(M_seq, temp_q).squeeze(-1)

    def read_buffer(self, x_t, memory_state):
        q = F.normalize(self.kq_proj(x_t), dim=-1)
        return torch.bmm(memory_state, (self.temp * q).unsqueeze(2)).squeeze(2)

    def _summarise_state(self, h_t):
        return self.state_proj(h_t.mean(dim=(1, 3)))