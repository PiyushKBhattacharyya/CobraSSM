import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSSM(nn.Module):
    """
    Multi-Scale Selective State Space Model backbone for CobraSSM.
    Stable Hybrid Recurrence version for DirectML/Numerical Stability.
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

    def forward(self, x, state=None):
        b, seq_len, d = x.shape
        device = x.device
        dtype = x.dtype

        # Projections
        x_proj_out = self.x_proj(x)
        dt_inp, b_vec, C_all = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        B_all = b_vec.unsqueeze(2) * self.B_mix.unsqueeze(0).unsqueeze(0)
        
        # dts: (b, L, scales, d_model)
        dts_list = [F.softplus(proj(dt_inp)) for proj in self.dt_projs]
        dt_all_seq = torch.stack(dts_list, dim=2)
        dt_all_seq = torch.clamp(dt_all_seq, max=20.0)

        # A: (scales, d_model, d_state)
        A = -torch.exp(self.A_log)
        
        current_h = state if state is not None else \
                    torch.zeros(b, self.num_scales, self.d_model, self.d_state, device=device, dtype=dtype)
        
        from .ops.parallel_scan import selective_scan_dispatch
        
        # dA: (b, L, K, D, S), dBx: (b, L, K, D, S)
        # We need to broadcast and compute dA and dBx for the whole sequence at once
        # dt_all_seq: (b, L, K, D), A: (K, D, S)
        dA = torch.exp(dt_all_seq.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # B_all: (b, L, D, S), x: (b, L, D)
        dBx = dt_all_seq.unsqueeze(-1) * B_all.unsqueeze(2) * x.unsqueeze(2).unsqueeze(-1)
        
        # Parallel Selective Scan
        h_all = selective_scan_dispatch(dA, dBx, state) # (b, L, K, D, S)
        
        # Surprise (Parallel)
        # h_all: (b, L, K, D, S)
        h_mean = h_all.mean(dim=-1, keepdim=True)
        h_std = torch.sqrt((h_all - h_mean).pow(2).mean(dim=-1) + 1e-6)
        h_surprise_seq = h_std.mean(dim=2) # (b, L, D) - Average over scales K
        
        # Output y_t = sum_k C_all h_{k}
        # h_all: (b, L, K, D, S), C_all: (b, L, S)
        y_scales = torch.einsum('blkds,bls->blkd', h_all, C_all) # (b, L, K, D)
        
        current_h = h_all[:, -1] # Last state for next step
        
        y_out = y_scales.view(b, seq_len, -1)
        y_final = self.out_proj(y_out) + x * self.D
        
        return y_final, h_surprise_seq, current_h

    def forward_step(self, x_t, state):
        """
        Single-token recurrence for O(1) generation.
        x_t   : (b, 1, d)
        state : (b, num_scales, d_model, d_state)
        Returns: y_t (b, 1, d), h_surprise_t (b, 1, d), new_state
        """
        b, _, d = x_t.shape
        x_proj_out = self.x_proj(x_t)  # (b, 1, proj_dim)
        dt_inp, b_vec, C_t = torch.split(
            x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # B: (b, 1, d, s)
        B_t = b_vec.unsqueeze(2) * self.B_mix.unsqueeze(0).unsqueeze(0)
        # dts: (b, 1, scales, d)
        dts = [F.softplus(proj(dt_inp)) for proj in self.dt_projs]
        dt_t = torch.clamp(torch.stack(dts, dim=2), max=20.0)  # (b, 1, K, D)

        A = -torch.exp(self.A_log)  # (K, D, S)

        # Squeeze time dim for the recurrence
        dt_1 = dt_t[:, 0]    # (b, K, D)
        B_1  = B_t[:, 0]     # (b, D, S)
        x_1  = x_t[:, 0]     # (b, D)
        C_1  = C_t[:, 0]     # (b, S)

        dA  = torch.exp(dt_1.unsqueeze(-1) * A.unsqueeze(0))        # (b, K, D, S)
        dBx = dt_1.unsqueeze(-1) * B_1.unsqueeze(1) * x_1.unsqueeze(1).unsqueeze(-1)

        new_state = dA * state + dBx

        # Surprise
        h_mean = new_state.mean(dim=-1, keepdim=True)
        h_std  = torch.sqrt((new_state - h_mean).pow(2).mean(dim=-1) + 1e-6)
        h_surprise = h_std.mean(dim=1).unsqueeze(1)  # (b, 1, D)

        # Output
        y_t = torch.einsum('bkds,bs->bkd', new_state, C_1)  # (b, K, D)
        y_out = y_t.unsqueeze(1).reshape(b, 1, -1)           # (b, 1, K*D)
        y_final = self.out_proj(y_out) + x_t * self.D

        return y_final, h_surprise, new_state