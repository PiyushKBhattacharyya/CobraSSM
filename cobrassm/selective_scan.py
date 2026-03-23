import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hybrid_utils import hybrid_apply


class MultiScaleSSM(nn.Module):
    """
    Multi-Scale Selective State Space Model backbone for CobraSSM.
    Supports Hybrid GPU-Forward/CPU-Backward for DirectML stability.
    """

    def __init__(self, d_model, d_state=16, num_scales=4, dt_rank="auto"):
        super().__init__()
        self.d_model    = d_model
        self.d_state    = d_state
        self.num_scales = num_scales
        self.dt_rank    = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        A_logs = []
        for k in range(num_scales):
            base    = 0.05 + (3.0 - 0.05) * k / max(num_scales - 1, 1)
            A_log_k = torch.full((d_model, d_state), math.log(base))
            A_log_k = A_log_k + torch.randn(d_model, d_state) * 0.01
            A_logs.append(A_log_k)
        self.A_log = nn.Parameter(torch.stack(A_logs))

        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state + d_state, bias=False)
        self.B_mix = nn.Parameter(torch.ones(d_model, d_state))

        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, d_model, bias=True)
            for _ in range(num_scales)
        ])
        for proj in self.dt_projs:
            nn.init.constant_(proj.bias, math.log(math.expm1(1.0)))

        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model * num_scales, d_model, bias=False)

    def _core_forward(self, *args):
        """
        Pure functional-ish forward pass.
        args: x, A_log, x_proj_weight, B_mix, *dt_weights, *dt_biases, D, out_proj_weight, state
        """
        # Manual Unpacking to match params list
        x = args[0]
        A_log = args[1]
        x_proj_weight = args[2]
        B_mix = args[3]
        
        # dt_projs: w0..w3, b0..b3
        idx = 4
        dt_weights = args[idx : idx + self.num_scales]
        idx += self.num_scales
        dt_biases = args[idx : idx + self.num_scales]
        idx += self.num_scales
        
        D = args[idx]
        out_proj_weight = args[idx+1]
        state = args[idx+2]
        
        b, seq_len, d = x.shape
        device = x.device
        dtype = x.dtype

        x_proj_out = F.linear(x, x_proj_weight)
        dt_inp, b_vec, C_all = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        B_all = b_vec.unsqueeze(2) * B_mix.unsqueeze(0).unsqueeze(0)
        
        dts_list = [F.softplus(F.linear(dt_inp, w, b)) for w, b in zip(dt_weights, dt_biases)]
        dt_all_seq = torch.stack(dts_list, dim=2)
        dt_all_seq = torch.clamp(dt_all_seq, max=20.0)

        A = -torch.exp(A_log)
        
        current_h = state if state is not None else \
                    torch.zeros(b, self.num_scales, self.d_model, self.d_state, device=device, dtype=dtype)
        
        y_scales_list = []
        h_surprise_list = []
        
        for t in range(seq_len):
            dt_t = dt_all_seq[:, t]
            B_t = B_all[:, t]
            x_t = x[:, t]
            C_t = C_all[:, t]
            
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dBx = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(1).unsqueeze(-1)
            
            current_h = dA * current_h + dBx
            
            h_mean = current_h.mean(dim=-1, keepdim=True)
            h_std = torch.sqrt((current_h - h_mean).pow(2).mean(dim=-1) + 1e-6)
            h_surprise_list.append(h_std.mean(dim=1))
            
            y_t = torch.einsum('bkds,bs->bkd', current_h, C_t)
            y_scales_list.append(y_t)

        y_scales = torch.stack(y_scales_list, dim=1)
        h_surprise_seq = torch.stack(h_surprise_list, dim=1)
        
        y_out = y_scales.view(b, seq_len, -1)
        y_final = F.linear(y_out, out_proj_weight) + x * D
        
        return y_final, h_surprise_seq, current_h

    def forward(self, x, state=None):
        params = [
            self.A_log, self.x_proj.weight, self.B_mix,
            *[p.weight for p in self.dt_projs],
            *[p.bias for p in self.dt_projs],
            self.D, self.out_proj.weight
        ]
        
        if x.device.type == 'privateuseone' and x.requires_grad:
             return hybrid_apply(self._core_forward, params, x, *params, state)
        else:
             return self._core_forward(x, *params, state)