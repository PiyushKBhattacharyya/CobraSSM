import torch
import torch.nn as nn
import torch.nn.functional as F

class EventDetector(nn.Module):
    """
    Event Detector for CobraSSM Strike mechanism.
    Computes soft probabilistic gating scores S_t based on input x_t and the 
    latest hidden state h_t representation (e.g. state variance or magnitude as surprise).
    """
    def __init__(self, d_model, d_state, num_scales):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_scales = num_scales
        
        # We project x_t and a summarized h_t to compute the significance score
        self.x_proj = nn.Linear(d_model, d_model // 2)
        # Summarize hidden state across scales and state dim
        self.h_proj = nn.Linear(d_model, d_model // 2)
        
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, x_t, h_t):
        """
        x_t: (batch, d_model)
        h_t: (batch, num_scales, d_model, d_state) - The hidden state from the SSM backbone
        returns: S_t (batch, 1) - Soft importance score in [0, 1]
        """
        # Feature 1: The input token information
        x_feat = self.x_proj(x_t)
        
        # Feature 2: Summarize the multi-scale hidden state
        # We use standard deviation across the state dimension as a "surprise" / "entropy" surrogate
        # h_std: (batch, num_scales, d_model) -> (batch, d_model) by taking mean across scales
        h_std = torch.std(h_t, dim=-1).mean(dim=1)
        h_feat = self.h_proj(h_std)
        
        # Concatenate features
        combined = torch.cat([x_feat, h_feat], dim=-1)
        
        # Compute soft gating score
        s_logit = self.scorer(combined)
        S_t = torch.sigmoid(s_logit)
        
        return S_t
