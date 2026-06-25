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
