import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableMemoryBuffer(nn.Module):
    """
    Linear Associative Memory with Multi-Head projections, Dynamic Decay, and SSM Fusion.
    Stable recurrent version.
    """

    def __init__(self, d_model, d_state=16, num_slots=64, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_model // num_heads

        # Multi-Head projections
        self.kq_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)

        self.log_temp = nn.Parameter(torch.tensor(1.386))  # exp(1.386) ≈ 4.0
        
        # Dynamic Decay Gate (Learned vector decay per channel)
        self.decay_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.decay_proj.bias, 3.0)
        nn.init.normal_(self.decay_proj.weight, std=0.01)

        # SSM-Memory Fusion: surprise signal projection to scale writing strength
        self.surprise_proj = nn.Linear(d_model, num_heads, bias=True)
        nn.init.constant_(self.surprise_proj.bias, 0.0)
        nn.init.normal_(self.surprise_proj.weight, std=0.01)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    @property
    def temp(self):
        return torch.exp(self.log_temp)

    def forward_batch(self, x, x_prev, S, h_surprise, memory_init=None):
        """
        x          : (b, seq_len, d_model)
        x_prev     : (b, seq_len, d_model)
        S          : (b, seq_len, 1)
        h_surprise : (b, seq_len, d_model)
        """
        b, seq_len, d = x.shape
        device = x.device
        dtype = x.dtype
        
        # Projections & Reshaping to Multi-Head format: (b, L, num_heads, head_dim)
        k_all = F.normalize(self.kq_proj(x_prev), dim=-1).view(b, seq_len, self.num_heads, self.head_dim)
        v_all = self.v_proj(x).view(b, seq_len, self.num_heads, self.head_dim)
        
        # Dynamic decay gate: (b, L, num_heads, head_dim)
        decay_all = torch.sigmoid(self.decay_proj(x_prev)).view(b, seq_len, self.num_heads, self.head_dim)
        
        # SSM-Memory Fusion surprise scale factor: (b, L, num_heads)
        surprise_scale_all = 1.0 + 4.0 * torch.sigmoid(self.surprise_proj(h_surprise))
        
        current_M = memory_init if memory_init is not None else \
                    torch.zeros(b, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
        
        M_seq_list = []
        for t in range(seq_len):
            k = k_all[:, t]                  # (b, num_heads, head_dim)
            v = v_all[:, t]                  # (b, num_heads, head_dim)
            decay = decay_all[:, t]          # (b, num_heads, head_dim)
            s = S[:, t]                      # (b, 1)
            surp = surprise_scale_all[:, t]  # (b, num_heads)
            
            # Combine event write score with surprise scale
            write_scale = (s * surp).unsqueeze(-1).unsqueeze(-1)  # (b, num_heads, 1, 1)
            
            # Write outer product: (b, num_heads, head_dim, head_dim)
            write = write_scale * (v.unsqueeze(-1) * k.unsqueeze(-2))
            
            # Recurrent update: decay applied elementwise to key/value slots
            current_M = decay.unsqueeze(-1) * current_M + write
            M_seq_list.append(current_M)
            
        M_seq = torch.stack(M_seq_list, dim=1) # (b, seq_len, num_heads, head_dim, head_dim)
        return M_seq, current_M

    def forward_step(self, x_t, x_prev, S_t, memory_state, h_surprise):
        """
        Single step forward for O(1) generation.
        x_t, x_prev: (b, d_model)
        S_t: (b, 1)
        memory_state: (b, num_heads, head_dim, head_dim)
        h_surprise: (b, d_model)
        """
        b = x_t.shape[0]
        k = F.normalize(self.kq_proj(x_prev), dim=-1).view(b, self.num_heads, self.head_dim)
        v = self.v_proj(x_t).view(b, self.num_heads, self.head_dim)
        decay = torch.sigmoid(self.decay_proj(x_prev)).view(b, self.num_heads, self.head_dim)
        
        surp = 1.0 + 4.0 * torch.sigmoid(self.surprise_proj(h_surprise))
        write_scale = (S_t * surp).unsqueeze(-1).unsqueeze(-1)
        
        write = write_scale * (v.unsqueeze(-1) * k.unsqueeze(-2))
        new_M = decay.unsqueeze(-1) * memory_state + write
        return new_M

    def forward(self, x_t, x_prev, h_t, S_t, memory_state=None, h_surprise=None):
        """Legacy single step forward wrapper."""
        return self.forward_step(x_t, x_prev, S_t, memory_state, h_surprise)

    def read_buffer_batch(self, x, M_seq):
        """
        x     : (b, seq_len, d_model)
        M_seq : (b, seq_len, num_heads, head_dim, head_dim)
        """
        b, seq_len, d = x.shape
        q = F.normalize(self.kq_proj(x), dim=-1).view(b, seq_len, self.num_heads, self.head_dim)
        temp_q = (self.temp * q).unsqueeze(-1) # (b, seq_len, num_heads, head_dim, 1)
        
        y_heads = torch.matmul(M_seq, temp_q).squeeze(-1) # (b, seq_len, num_heads, head_dim)
        y = y_heads.view(b, seq_len, d)
        return self.out_proj(y)

    def read_buffer(self, x_t, memory_state):
        """
        x_t          : (b, d_model)
        memory_state : (b, num_heads, head_dim, head_dim)
        """
        b = x_t.shape[0]
        q = F.normalize(self.kq_proj(x_t), dim=-1).view(b, self.num_heads, self.head_dim)
        temp_q = (self.temp * q).unsqueeze(-1) # (b, num_heads, head_dim, 1)
        
        y_heads = torch.matmul(memory_state, temp_q).squeeze(-1) # (b, num_heads, head_dim)
        y = y_heads.view(b, self.d_model)
        return self.out_proj(y)