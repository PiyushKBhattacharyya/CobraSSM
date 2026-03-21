import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableMemoryBuffer(nn.Module):
    """
    Bounded, differentiable structured KV memory buffer.
    Replaces hard LRU with an attention-weighted soft-overwrite slot mechanism.
    Incorporates positional decay to provide temporal bias (awareness of recency).
    """
    def __init__(self, d_model, num_slots=64):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        
        # In a differentiable slot setting, we can define fixed slot keys or learned ones
        self.slot_keys = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_t, S_t, memory_state=None):
        """
        Write operation. memory_state is now a tuple (mem_k, mem_v)
        """
        b = x_t.size(0)
        
        if memory_state is None:
            mem_k = torch.zeros(b, self.num_slots, self.d_model, device=x_t.device)
            mem_v = torch.zeros(b, self.num_slots, self.d_model, device=x_t.device)
        else:
            mem_k, mem_v = memory_state
            
        k_t = self.k_proj(x_t).unsqueeze(1) # (b, 1, d_model)
        v_t = self.v_proj(x_t).unsqueeze(1) # (b, 1, d_model)
        
        # Route token to slots based on fixed slot centroids
        write_logits = torch.matmul(k_t, self.slot_keys.transpose(-1, -2)) / (self.d_model ** 0.5)
        write_weights = torch.softmax(write_logits, dim=-1) # (b, 1, num_slots)
        
        # Calculate soft write strength
        write_strength = write_weights.transpose(-1, -2) * S_t.unsqueeze(-1) # (b, num_slots, 1)
        
        decay_factor = 1.0 - write_strength
        
        # Update both keys and values in the slots
        # To maintain magnitude safely, we interpolate
        updated_mem_k = mem_k * decay_factor + k_t * write_strength
        updated_mem_v = mem_v * decay_factor + v_t * write_strength
        
        return (updated_mem_k, updated_mem_v)

    def read_buffer(self, q_t, memory_state):
        mem_k, mem_v = memory_state
        q_t = q_t.unsqueeze(1) # (b, 1, d_model)
        
        # Standard attention over the dynamic memory keys, NOT the static slot centroid keys!
        attn_logits = torch.matmul(q_t, mem_k.transpose(-1, -2)) / (self.d_model ** 0.5) # (b, 1, num_slots)
        
        attn_weights = torch.softmax(attn_logits, dim=-1)
        
        # Retrieve values
        read_out = torch.matmul(attn_weights, mem_v) # (b, 1, d_model)
        
        return read_out.squeeze(1)
