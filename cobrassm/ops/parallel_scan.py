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
    b, L, K, D, S = dA.shape
    device = dA.device
    
    # We use the Log-Space prefix sum trick:
    # h_t = exp(cumsum(log(dA))) * [h0 + cumsum(dBx * exp(-cumsum(log(dA))))]
    
    log_dA = torch.log(dA + 1e-9) # Stability epsilon
    cum_log_dA = torch.cumsum(log_dA, dim=1)
    
    # Adjust for initial state
    if h0 is not None:
        # h0 is effectively dBx at t=-1 with dA=1
        # To handle h0 correctly in a pure cumsum, we can prepend it or add it to the first term
        # But a more stable way for recurrent state is:
        # h_t = exp(cum_log_dA) * h0 + sum_{i=0}^t exp(cum_log_dA_t - cum_log_dA_i) * dBx_i
        pass

    # Efficient vectorized associative scan (simplified for now to log-space prefix sum)
    # Note: For production, we'd use a more robust Blelloch scan if L is massive
    
    # Pre-compute exponential cumulative factors
    exp_cum_dA = torch.exp(cum_log_dA)
    
    # Compute the weighted sum of inputs
    # term = dBx_t / exp_cum_dA_t
    term = dBx * torch.exp(-cum_log_dA)
    cum_term = torch.cumsum(term, dim=1)
    
    h = exp_cum_dA * cum_term
    
    if h0 is not None:
        h = h + exp_cum_dA * h0.unsqueeze(1)
        
    return h

def selective_scan_dispatch(dA, dBx, h0=None):
    """
    Hardware-aware dispatcher for Selective Scan.
    """
    if torch.cuda.is_available():
        try:
            from .triton_scan import triton_selective_scan
            return triton_selective_scan(dA, dBx, h0)
        except ImportError:
            return pscan_selective_scan(dA, dBx, h0)
    else:
        # Default to PyTorch Parallel Scan (Works on MPS/CPU)
        return pscan_selective_scan(dA, dBx, h0)
