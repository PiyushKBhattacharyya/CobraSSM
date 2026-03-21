import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableMemoryBuffer(nn.Module):
    """
    Linear Associative Memory with tied key/query projection.

    Write:  M = decay * M + S_t * v_t ⊗ k_t
            k_t = normalize(kq_proj(x_prev))
            v_t = v_proj(x_t)

    Read:   y = M @ (temp * normalize(kq_proj(x_t)))

    Changes from previous version:
    - decay_logit init 2.0→3.0: sigmoid(3.0)=0.952 vs 0.881.
      With 7 pairs over 14 write steps, decay=0.88 attenuates the
      first pair to 0.88^14=0.17. decay=0.95 gives 0.95^14=0.49.
      The gradient d(sigmoid)/d(logit) is also larger at logit=3
      (0.095 vs 0.105) — slight improvement but mainly the init matters.
    """

    def __init__(self, d_model, d_state=16, num_slots=64):
        super().__init__()
        self.d_model = d_model

        self.kq_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)

        self.log_temp    = nn.Parameter(torch.tensor(1.386))  # exp(1.386) ≈ 4.0
        self.decay_logit = nn.Parameter(torch.tensor(3.0))    # sigmoid(3.0) ≈ 0.952

        self.state_proj = nn.Linear(d_model, d_model, bias=False)

    @property
    def temp(self):
        return torch.exp(self.log_temp)

    @property
    def decay(self):
        return torch.sigmoid(self.decay_logit)

    def forward(self, x_t, x_prev, h_t, S_t, memory_state=None):
        b = x_t.size(0)

        k_t = F.normalize(self.kq_proj(x_prev), dim=-1)
        v_t = self.v_proj(x_t)

        write = S_t.unsqueeze(-1) * torch.bmm(
            v_t.unsqueeze(2),
            k_t.unsqueeze(1),
        )

        if memory_state is None:
            M = torch.zeros(b, self.d_model, self.d_model,
                            device=x_t.device, dtype=x_t.dtype)
        else:
            M = memory_state

        return self.decay * M + write

    def read_buffer(self, x_t, memory_state):
        q = F.normalize(self.kq_proj(x_t), dim=-1)
        return torch.bmm(memory_state, (self.temp * q).unsqueeze(2)).squeeze(2)

    def _summarise_state(self, h_t):
        return self.state_proj(h_t.mean(dim=(1, 3)))