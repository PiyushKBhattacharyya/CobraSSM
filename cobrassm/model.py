import torch
import torch.nn as nn
from .cobra_block import CobraBlock, RMSNorm

class CobraSSM(nn.Module):
    """
    Full CobraSSM Language Model.
    Designed for O(n) exact token recall and multi-scale continuous sequence modeling.
    """
    def __init__(self, vocab_size, d_model=512, n_layers=8, d_state=16, num_scales=4, num_slots=64):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            CobraBlock(d_model, d_state, num_scales, num_slots)
            for _ in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, ssm_states=None, mem_states=None):
        """
        input_ids: (batch, seq_len)
        Returns: logits (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)
        
        if ssm_states is None:
            ssm_states = [None] * self.n_layers
        if mem_states is None:
            mem_states = [None] * self.n_layers
            
        out_ssm_states = []
        out_mem_states = []
        
        for i, block in enumerate(self.blocks):
            x, s_ssm, s_mem = block(x, ssm_states[i], mem_states[i])
            out_ssm_states.append(s_ssm)
            out_mem_states.append(s_mem)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits, out_ssm_states, out_mem_states

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test forward pass with dummy data
    model = CobraSSM(vocab_size=10000, d_model=128, n_layers=2)
    x = torch.randint(0, 10000, (2, 32))
    logits, ssm, mem = model(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Total parameters: {model.parameter_count()/1e6:.2f} M")
