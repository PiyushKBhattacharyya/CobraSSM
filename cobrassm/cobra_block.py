import torch
import torch.nn as nn
import torch.nn.functional as F
from .selective_scan import MultiScaleSSM
from .event_detector import EventDetector
from .memory_buffer import DifferentiableMemoryBuffer

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm.type_as(x) * self.weight

class CobraBlock(nn.Module):
    """
    Integrates the Continuous SSM path and the Symbolic Memory path.
    Applies the "Strike" mechanism for soft reading from the memory buffer.
    """
    def __init__(self, d_model, d_state=16, num_scales=4, num_slots=64):
        super().__init__()
        self.d_model = d_model
        
        self.norm_ssm = RMSNorm(d_model)
        self.ssm = MultiScaleSSM(d_model, d_state, num_scales)
        
        self.event_detector = EventDetector(d_model, d_state, num_scales)
        self.memory = DifferentiableMemoryBuffer(d_model, num_slots)
        
        # User explicitly requested RMSNorm before Q/K/V memory projections for stability
        self.norm_mem_q = RMSNorm(d_model)
        self.norm_mem_kv = RMSNorm(d_model)
        
        # Fusion gate
        self.fusion_gate = nn.Linear(d_model, d_model)
        self.mem_out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.mlp_norm = RMSNorm(d_model)
        # Standard SwiGLU MLP
        self.mlp_up = nn.Linear(d_model, d_model * 4 * 2)
        self.mlp_down = nn.Linear(d_model * 4, d_model)

    def forward(self, x, ssm_state=None, mem_state=None):
        """
        x: (batch, seq_len, d_model)
        returns:
            out: (batch, seq_len, d_model) - Strictly residual
            ssm_state_out
            mem_state_out
        """
        b, seq_len, d = x.shape
        
        residual = x
        
        # --- PATH 1: Continuous Flow (Multi-Scale SSM) ---
        x_norm = self.norm_ssm(x)
        # y_ssm: (b, seq_len, d_model), h_seq: (b, seq_len, scales, d, state_d)
        y_ssm, h_seq, ssm_state_out = self.ssm(x_norm, state=ssm_state)
        
        # --- PATH 2: Symbolic Flow / memory operations ---
        y_mem_list = []
        current_mem_state = mem_state
        
        # Step-by-step memory updates and strikes based on the event detector
        for t in range(seq_len):
            x_t = x_norm[:, t, :]
            h_t = h_seq[:, t]
            
            # 1. Event Detection (Soft gating)
            S_t = self.event_detector(x_t, h_t)
            
            # 2. Write to memory (attention-weighted slot overwrite)
            x_t_kv_norm = self.norm_mem_kv(x_t)
            current_mem_state = self.memory(x_t_kv_norm, S_t, current_mem_state)
            
            # 3. Read from memory (Sparse Attention "Strike")
            x_t_q_norm = self.norm_mem_q(x_t)
            y_mem_t = self.memory.read_buffer(x_t_q_norm, current_mem_state)
            
            y_mem_list.append(y_mem_t)
            
        y_mem = torch.stack(y_mem_list, dim=1) # (batch, seq_len, d_model)
        
        # --- FUSION ---
        # $y_block = y^{ssm} + g_t \odot \text{Linear}(y^{mem})$
        g_t = torch.sigmoid(self.fusion_gate(x_norm))
        y_block = y_ssm + g_t * self.mem_out_proj(y_mem)
        
        # Residual connection 1
        x = residual + y_block
        
        # MLP / FFN block (SwiGLU)
        residual = x
        x_mlp_norm = self.mlp_norm(x)
        up = self.mlp_up(x_mlp_norm)
        gate, val = up.chunk(2, dim=-1)
        mlp_out = self.mlp_down(F.silu(gate) * val)
        
        # Residual connection 2 (Strict residual path per user feedback)
        out = residual + mlp_out
        
        return out, ssm_state_out, current_mem_state
