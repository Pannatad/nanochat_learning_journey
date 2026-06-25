import torch
from torch import nn


def make_causal_mask(sequence_length: int) -> torch.Tensor:
    return torch.tril(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool)
    )


class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q = self.query(x)  # (B, T, d_model)
        k = self.key(x)  # (B, T, d_model)
        v = self.value(x)  # (B, T, d_model)
        scores = (q @ k.transpose(-2, -1)) / self.d_model**0.5
        mask = make_causal_mask(sequence_length=x.shape[1])
        scores = scores.masked_fill(~mask, float("-inf"))
        wei = scores.softmax(dim=-1)
        out = wei @ v

        if return_attention_weights:
            return out, wei
        return out


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head
        self.head = nn.ModuleList(
            [SingleHeadCausalSelfAttention(self.head_dim) for _ in range(n_head)]
        )
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_heads = x.split(self.head_dim, dim=-1)
        output = [head(x_part) for head, x_part in zip(self.head, x_heads)]
        out = torch.cat(output, dim=-1)
        out = self.output_projection(out)
        return out
