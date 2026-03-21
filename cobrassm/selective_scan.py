import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSSM(nn.Module):
    """
    Multi-Scale Selective State Space Model backbone for CobraSSM.

    B is per-feature but implemented as low-rank:
        b_vec : (b, L, d_state)        -- input-dependent, projected from x
        B_mix : (d_model, d_state)     -- learned per-feature mixing weights
        B_t   = b_vec.unsqueeze(2) * B_mix   -> (b, d_model, d_state)

    This keeps per-feature selectivity without the 128×16=2048-dim x_proj
    blowup that caused token-0 collapse in the previous version.
    x_proj stays small: d_model -> dt_rank + d_state + d_state.
    """

    def __init__(self, d_model, d_state=16, num_scales=4, dt_rank="auto"):
        super().__init__()
        self.d_model    = d_model
        self.d_state    = d_state
        self.num_scales = num_scales
        self.dt_rank    = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # A matrices: evenly spaced log-decays, fast (k=0) to slow (k=N-1)
        A_logs = []
        for k in range(num_scales):
            base    = 0.05 + (3.0 - 0.05) * k / max(num_scales - 1, 1)
            A_log_k = torch.full((d_model, d_state), math.log(base))
            A_log_k = A_log_k + torch.randn(d_model, d_state) * 0.01
            A_logs.append(A_log_k)
        self.A_log = nn.Parameter(torch.stack(A_logs))  # (scales, d_model, d_state)

        # Small input projection: dt_rank + d_state (b_vec) + d_state (C)
        self.x_proj = nn.Linear(
            d_model,
            self.dt_rank + d_state + d_state,
            bias=False,
        )

        # Per-feature B mixing matrix — expands b_vec to (d_model, d_state)
        # Init to ones: at init B_t = b_vec broadcast, identical to old scalar B
        self.B_mix = nn.Parameter(torch.ones(d_model, d_state))

        # Scale-specific dt projections
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, d_model, bias=True)
            for _ in range(num_scales)
        ])
        for proj in self.dt_projs:
            nn.init.constant_(proj.bias, math.log(math.expm1(1.0)))

        # D skip connection (Mamba-style direct passthrough)
        self.D = nn.Parameter(torch.ones(d_model))

        # Output aggregation across scales
        self.out_proj = nn.Linear(d_model * num_scales, d_model, bias=False)

    def forward(self, x, state=None):
        """
        x     : (batch, seq_len, d_model)
        state : (batch, num_scales, d_model, d_state) or None
        Returns: y_final, h_seq, current_state
        """
        b, seq_len, d = x.shape

        # Single small projection for all scales
        x_proj_out = self.x_proj(x)  # (b, L, dt_rank + d_state + d_state)
        dt_inp, b_vec, C = torch.split(
            x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Low-rank per-feature B: (b, L, d_model, d_state)
        # b_vec: (b, L, 1, d_state) * B_mix: (1, 1, d_model, d_state)
        B = b_vec.unsqueeze(2) * self.B_mix.unsqueeze(0).unsqueeze(0)

        # Per-scale dt: list of (b, L, d_model)
        dts = [F.softplus(proj(dt_inp)) for proj in self.dt_projs]

        # A always negative
        A = -torch.exp(self.A_log)  # (scales, d_model, d_state)

        if state is None:
            state = torch.zeros(
                b, self.num_scales, self.d_model, self.d_state,
                device=x.device, dtype=x.dtype,
            )
        current_state = state.clone()

        y_ssm = []
        h_seq = []

        for t in range(seq_len):
            x_t = x[:, t, :]       # (b, d_model)
            B_t = B[:, t, :, :]    # (b, d_model, d_state)
            C_t = C[:, t, :]       # (b, d_state)

            y_t_scales      = []
            next_state_list = []

            for k in range(self.num_scales):
                dt_tk = torch.clamp(dts[k][:, t, :], max=20.0)  # (b, d_model)
                A_k   = A[k]                                      # (d_model, d_state)

                # ZOH discretisation
                dA_k  = torch.exp(dt_tk.unsqueeze(-1) * A_k)     # (b, d_model, d_state)
                dB_k  = dt_tk.unsqueeze(-1) * B_t                 # (b, d_model, d_state)

                h_k     = current_state[:, k]                     # (b, d_model, d_state)
                h_k_new = dA_k * h_k + dB_k * x_t.unsqueeze(-1)
                next_state_list.append(h_k_new)

                y_tk = torch.einsum('bds,bs->bd', h_k_new, C_t)  # (b, d_model)
                y_t_scales.append(y_tk)

            current_state = torch.stack(next_state_list, dim=1)
            y_ssm.append(torch.cat(y_t_scales, dim=-1))          # (b, d*scales)
            h_seq.append(current_state)

        y_out   = torch.stack(y_ssm, dim=1)   # (b, L, d*scales)
        y_final = self.out_proj(y_out)         # (b, L, d_model)
        y_final = y_final + x * self.D        # D skip connection

        h_seq = torch.stack(h_seq, dim=1)     # (b, L, scales, d_model, d_state)

        return y_final, h_seq, current_state