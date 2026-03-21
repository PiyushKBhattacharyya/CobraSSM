import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSSM(nn.Module):
    """
    Multi-Scale Selective State Space Model backbone for CobraSSM.
    Uses multiple A matrices with different decay rates to significantly improve over baseline single Mamba models.
    Maintains O(n) continuous hidden states with input-dependent parameterization.
    """
    def __init__(self, d_model, d_state=16, num_scales=4, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_scales = num_scales
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Multi-scale A matrices: initialized with different timescales (e.g., from small to large decays)
        # Shape: (num_scales, d_model, d_state)
        A_init = torch.stack([
            torch.randn(d_model, d_state) * (0.1 ** i) for i in range(num_scales)
        ])
        # We enforce A to be negative for stability
        self.A_log = nn.Parameter(torch.log(torch.abs(A_init) + 1e-4))
        
        # Input-dependent projections
        # Projects to Delta (step size), B, and C
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        
        # Scale-specific Delta projections
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, d_model, bias=True) for _ in range(num_scales)
        ])
        
        # For output aggregation
        self.out_proj = nn.Linear(d_model * num_scales, d_model, bias=False)

    def forward(self, x, state=None):
        """
        x: (batch, seq_len, d_model)
        state: optional initial state (batch, num_scales, d_model, d_state)
        returns: (y, state_out)
            y: (batch, seq_len, d_model)
            state_out: latest hidden state
        """
        b, seq_len, d = x.shape
        
        # Calculate Delta, B, C from input
        # x_proj_out: (batch, seq_len, dt_rank + 2 * d_state)
        x_proj_out = self.x_proj(x)
        dt_inp, B, C = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Compute multi-scale Deltas
        # List of (batch, seq_len, d_model)
        dts = [F.softplus(proj(dt_inp)) for proj in self.dt_projs]
        
        # Reconstruct A from log parameters
        A = -torch.exp(self.A_log) # (num_scales, d_model, d_state)
        
        y_ssm = []
        h_seq = []
        
        if state is None:
            # (batch, num_scales, d_model, d_state)
            state = torch.zeros(b, self.num_scales, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
            
        current_state = state.clone()
        
        # Recurrent scan over sequence (placeholder for efficient parallel scan later)
        # A true parallel associative scan would replace this loop for full speed.
        for t in range(seq_len):
            x_t = x[:, t, :] # (b, d)
            B_t = B[:, t, :] # (b, d_state)
            C_t = C[:, t, :] # (b, d_state)
            
            y_t_scales = []
            
            for k in range(self.num_scales):
                dt_tk = dts[k][:, t, :] # (b, d)
                A_k = A[k] # (d, d_state)
                
                # Discretize A and B (Zero-order hold approximation)
                # dA: (b, d, d_state)
                dA_k = torch.exp(dt_tk.unsqueeze(-1) * A_k) 
                # dB: (b, d, d_state)
                dB_k = dt_tk.unsqueeze(-1) * B_t.unsqueeze(1) 
                
                # Update hidden state
                h_k = current_state[:, k] # (b, d, d_state)
                h_k_new = dA_k * h_k + dB_k * x_t.unsqueeze(-1)
                current_state[:, k] = h_k_new
                
                # Compute output for this scale
                # h_k_new: (b, d, d_state), C_t: (b, d_state)
                y_tk = torch.einsum('bds,bs->bd', h_k_new, C_t)
                y_t_scales.append(y_tk)
                
            # Concat outputs from all scales and project back to d_model
            # Concat along feature dim: (b, d * num_scales)
            y_t_concat = torch.cat(y_t_scales, dim=-1) 
            y_ssm.append(y_t_concat)
            h_seq.append(current_state.clone())
            
        # (batch, seq_len, d_model * num_scales)
        y_out = torch.stack(y_ssm, dim=1)
        y_final = self.out_proj(y_out) # (batch, seq_len, d_model)
        
        h_seq = torch.stack(h_seq, dim=1) # (batch, seq_len, num_scales, d_model, d_state)
        
        return y_final, h_seq, current_state
