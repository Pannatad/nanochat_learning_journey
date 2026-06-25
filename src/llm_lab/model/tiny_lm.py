import torch
from torch import nn


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)  # (B, T, d_model)
        logits = self.output_head(x)
        return logits
