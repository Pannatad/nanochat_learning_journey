import torch
from torch import nn

from llm_lab.model.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(n_head, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)
        self.d_model = d_model
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_layer: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head) for _ in range(n_layer)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
