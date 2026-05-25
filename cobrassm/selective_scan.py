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

    def forward(self, x, state=None, bidirectional=False):
        b, seq_len, d = x.shape
        device = x.device
        dtype = x.dtype

        def _scan(x_in, state_in):
            # Projections
            x_proj_out = self.x_proj(x_in)
            dt_inp, b_vec, C_all = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)

            B_all = b_vec.unsqueeze(2) * self.B_mix.unsqueeze(0).unsqueeze(0)
            
            # dts: (b, L, scales, d_model)
            dts_list = [F.softplus(proj(dt_inp)) for proj in self.dt_projs]
            dt_all_seq = torch.stack(dts_list, dim=2)
            dt_all_seq = torch.clamp(dt_all_seq, max=20.0)

<<<<<<< HEAD
            # A: (scales, d_model, d_state)
            A = -torch.exp(self.A_log)
            
            current_h = state_in if state_in is not None else \
                        torch.zeros(b, self.num_scales, self.d_model, self.d_state, device=device, dtype=dtype)
            
            y_scales_list = []
            h_surprise_list = []
            
            # We use a stable recurrent loop over L (fast enough for Phase 4)
            for t in range(seq_len):
                dt_t = dt_all_seq[:, t] # (b, scales, d)
                B_t = B_all[:, t]       # (b, d, s)
                x_t = x_in[:, t]        # (b, d)
                C_t = C_all[:, t]       # (b, s)
                
                # dA = exp(dt * A): (b, scales, d, s)
                dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
                # dBx = (dt * B * x): (b, scales, d, s)
                dBx = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(1).unsqueeze(-1)
                
                # Recurrence: h_t = dA * h_{t-1} + dBx
                current_h = dA * current_h + dBx
                
                # Surprise
                h_mean = current_h.mean(dim=-1, keepdim=True)
                h_std = torch.sqrt((current_h - h_mean).pow(2).mean(dim=-1) + 1e-6)
                h_surprise_list.append(h_std.mean(dim=1)) # (b, d)
                
                # Output y_t = sum_k C_t h_{t,k}
                # current_h: (b, scales, d, s), C_t: (b, s)
                y_t = torch.einsum('bkds,bs->bkd', current_h, C_t) # Result: (b, scales, d)
                y_scales_list.append(y_t)

            y_scales = torch.stack(y_scales_list, dim=1) # (b, L, scales, d)
            h_surprise_seq = torch.stack(h_surprise_list, dim=1)
            
            y_out = y_scales.view(b, seq_len, -1)
            y_final = self.out_proj(y_out) + x_in * self.D
            
            return y_final, h_surprise_seq, current_h

        y_fwd, surp_fwd, h_fwd = _scan(x, state)

        if not bidirectional:
            return y_fwd, surp_fwd, h_fwd
            
        x_bwd = torch.flip(x, dims=[1])
        y_bwd, surp_bwd, h_bwd = _scan(x_bwd, state)

        y_bwd = torch.flip(y_bwd, dims=[1])
        surp_bwd = torch.flip(surp_bwd, dims=[1])

        return y_fwd + y_bwd, (surp_fwd + surp_bwd) / 2.0, h_fwd + h_bwd
=======
        # A: (scales, d_model, d_state)
        A = -torch.exp(self.A_log)
        
        current_h = state if state is not None else \
                    torch.zeros(b, self.num_scales, self.d_model, self.d_state, device=device, dtype=dtype)
        
        from .ops.parallel_scan import selective_scan_dispatch
        
        chunk_size = 1024
        y_scales_list = []
        h_surprise_list = []
        curr_h = state if state is not None else \
                 torch.zeros(b, self.num_scales, self.d_model, self.d_state, device=device, dtype=dtype)
        
        # Sequence-level chunking for memory efficiency
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            
            dt_chunk = dt_all_seq[:, i:end] # (b, chunk, K, D)
            B_chunk = B_all[:, i:end]       # (b, chunk, D, S)
            x_chunk = x[:, i:end]           # (b, chunk, D)
            C_chunk = C_all[:, i:end]       # (b, chunk, S)
            
            # Discretization for this chunk only
            # dA_chunk: (b, L_c, K, D, S)
            dA_chunk = torch.exp(dt_chunk.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            dBx_chunk = dt_chunk.unsqueeze(-1) * B_chunk.unsqueeze(2) * x_chunk.unsqueeze(2).unsqueeze(-1)
            
            # Parallel Scan for this chunk
            h_chunk = selective_scan_dispatch(dA_chunk, dBx_chunk, curr_h) # (b, L_c, K, D, S)
            
            # Surprise (Chunk)
            h_mean = h_chunk.mean(dim=-1, keepdim=True)
            h_std = torch.sqrt((h_chunk - h_mean).pow(2).mean(dim=-1) + 1e-6)
            h_surprise_list.append(h_std.mean(dim=2)) # (b, L_c, D)
            
            # Output (Chunk)
            y_c = torch.einsum('blkds,bls->blkd', h_chunk, C_chunk)
            y_scales_list.append(y_c)
            
            # Pass state to next chunk
            curr_h = h_chunk[:, -1]

        y_scales = torch.cat(y_scales_list, dim=1) # (b, L, K, D)
        h_surprise_seq = torch.cat(h_surprise_list, dim=1)
        current_h = curr_h
        
        y_out = y_scales.view(b, seq_len, -1)
        y_final = self.out_proj(y_out) + x * self.D
        
        return y_final, h_surprise_seq, current_h
>>>>>>> c74a6bb81eacf3a794ecc9cba36826f59ab6c02b

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