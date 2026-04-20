import torch
import torch.nn as nn

# Only try to import Triton if on CUDA. 
# On Mac, this file will be imported but the functions will only be called if CUDA is detected.
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def selective_scan_fwd_kernel(
        dA_ptr, dBx_ptr, h_ptr,
        b_stride, l_stride, s_stride,
        L, S,
        BLOCK_SIZE: tl.constexpr
    ):
        # Parallelize over (Batch * Scales * Channels)
        bid = tl.program_id(0)
        
        # Pointers for this specific channel
        chan_dA_ptr = dA_ptr + bid * b_stride
        chan_dBx_ptr = dBx_ptr + bid * b_stride
        chan_h_ptr = h_ptr + bid * b_stride
        
        # Load h_state (accumulator)
        # For simplicity in this version, we assume seq_len fits in blocks
        # or use a global-memory carry for multi-block scans.
        
        h_curr = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for t in range(L):
            # Load dA_t and dBx_t for all S states in this channel
            # In a real high-perf kernel, we'd tile S as well.
            offs = tl.arange(0, BLOCK_SIZE)
            mask = offs < S
            
            da_t = tl.load(chan_dA_ptr + t * l_stride + offs, mask=mask)
            dbx_t = tl.load(chan_dBx_ptr + t * l_stride + offs, mask=mask)
            
            # Recurrence
            h_curr = da_t * h_curr + dbx_t
            
            # Store back
            tl.store(chan_h_ptr + t * l_stride + offs, h_curr, mask=mask)

def triton_selective_scan(dA, dBx, h0=None):
    """
    Triton implementation of Selective Scan.
    Handles (B, L, K, D, S) -> (B, L, K, D, S)
    """
    if not HAS_TRITON:
        raise ImportError("Triton not installed. Cannot use triton_selective_scan.")
        
    b, L, K, d, s = dA.shape
    # Flatten Batch, Scales, Channels for parallelization
    dA_f = dA.reshape(-1, L, s)
    dBx_f = dBx.reshape(-1, L, s)
    out = torch.empty_like(dA_f)
    
    grid = (dA_f.shape[0],) # One program per channel
    
    # We use a power-of-2 block size for the state dimension S
    BLOCK_SIZE = 32 if s <= 32 else 64 # CobraSSM typically uses 16 or 32
    
    selective_scan_fwd_kernel[grid](
        dA_f, dBx_f, out,
        dA_f.stride(0), dA_f.stride(1), dA_f.stride(2),
        L, s,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.reshape(b, L, K, d, s)
