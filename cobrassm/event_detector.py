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
        
        # To keep the tensor entirely on the AMD GPU (DirectML doesn't support native torch.std), 
        # we compute the standard deviation manually using basic supported ops:
        h_mean = h_t.mean(dim=-1, keepdim=True)
        h_std_manual = torch.sqrt((h_t - h_mean).pow(2).mean(dim=-1) + 1e-6)
        
        # h_std_manual: (batch, num_scales)
        # Average the "surprise" across all multi-scale paths
        h_std = h_std_manual.mean(dim=1)
        
        h_feat = self.h_proj(h_std)
        
        # Concatenate features
        combined = torch.cat([x_feat, h_feat], dim=-1)
        
        # Compute soft gating score
        s_logit = self.scorer(combined)
        S_t = torch.sigmoid(s_logit)
        
        return S_t
