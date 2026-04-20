import torch
import torch.nn as nn
import torch.nn.functional as F

class EventDetector(nn.Module):
    """
    Event Detector for CobraSSM Strike mechanism.
    Updated to use compressed surprise signals (h_std) computed during scanning.
    """
    def __init__(self, d_model, d_state, num_scales):
        super().__init__()
        self.d_model = d_model
        
        self.x_proj = nn.Linear(d_model, d_model // 2)
        self.h_proj = nn.Linear(d_model, d_model // 2)
        
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, x, h_surprise, input_ids=None):
        """
        Standard PyTorch entry point. Handles both:
        - Single step: x (b, d), h_surprise (b, d)
        - Sequence: x (b, L, d), h_surprise (b, L, d)
        """
        x_feat = self.x_proj(x)
        h_feat = self.h_proj(h_surprise)
        combined = torch.cat([x_feat, h_feat], dim=-1)
        s_logit = self.scorer(combined)
        return torch.sigmoid(s_logit).to(dtype=x.dtype)
