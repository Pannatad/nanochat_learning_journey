import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        # Store epsilon.
        self.eps = eps
        # Create the learned gamma vector.
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self.eps)
        normalized = x * inv_rms
        return normalized * self.weight
