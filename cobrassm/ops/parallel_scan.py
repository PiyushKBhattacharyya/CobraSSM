import torch
import torch.nn as nn
import torch.nn.functional as F

def pscan_selective_scan(dA, dBx, h0=None):
    """
    Vectorized Parallel Scan for linear recurrence: h_t = dA_t * h_{t-1} + dBx_t
    dA: (b, L, K, D, S)
    dBx: (b, L, K, D, S)
    h0: (b, K, D, S)
    Returns: h (b, L, K, D, S)
    """
    b, L, K, d, s = dA.shape
    
    # h_t = exp(cum_log_dA) * [h0 + cumsum(dBx * exp(-cum_log_dA))]
    log_dA = torch.log(dA.clamp(min=1e-9))
    cum_log_dA = torch.cumsum(log_dA, dim=1)
    
    exp_cum_dA = torch.exp(cum_log_dA)
    term = dBx * torch.exp(-cum_log_dA)
    cum_term = torch.cumsum(term, dim=1)
    
    h = exp_cum_dA * cum_term
    
    if h0 is not None:
        h = h + exp_cum_dA * h0.unsqueeze(1)
        
    return h

def chunked_selective_scan(dA, dBx, h0=None, chunk_size=1024):
    """
    Segmented parallel scan to save VRAM.
    """
    b, L, K, d, s = dA.shape
    device = dA.device
    dtype = dA.dtype
    
    if L <= chunk_size:
        return pscan_selective_scan(dA, dBx, h0)
    
    h_list = []
    curr_h0 = h0
    
    for i in range(0, L, chunk_size):
        end = min(i + chunk_size, L)
        dA_chunk = dA[:, i:end]
        dBx_chunk = dBx[:, i:end]
        
        h_chunk = pscan_selective_scan(dA_chunk, dBx_chunk, curr_h0)
        h_list.append(h_chunk)
        
        # Last state becomes h0 for next chunk
        curr_h0 = h_chunk[:, -1]
        
    return torch.cat(h_list, dim=1)

def selective_scan_dispatch(dA, dBx, h0=None, chunk_size=1024):
    """
    Hardware-aware dispatcher with memory-efficient chunking.
    """
    if torch.cuda.is_available():
        try:
            from .triton_scan import triton_selective_scan
            # Triton kernel also benefits from internal chunking
            return triton_selective_scan(dA, dBx, h0)
        except ImportError:
            return chunked_selective_scan(dA, dBx, h0, chunk_size)
    else:
        # Optimized for MPS/CPU memory limits
        return chunked_selective_scan(dA, dBx, h0, chunk_size)
