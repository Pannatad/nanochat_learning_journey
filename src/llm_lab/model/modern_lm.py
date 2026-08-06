import torch
from torch import nn

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.factories import build_normalization
from llm_lab.model.modern_block import ModernTransformerBlock


class ModernLanguageModel(nn.Module):
    def __init__(self, config: ModernModelConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=config.block_size,
            embedding_dim=config.d_model,
        )

        self.blocks = nn.ModuleList(
            [ModernTransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.final_norm = build_normalization(
            kind=config.normalization,
            d_model=config.d_model,
        )

        self.output_head = nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            bias=False,
        )

        self.output_head.weight = self.token_embeddings.weight

    def _embed_inputs(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (B, T)")

        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")

        sequence_length = token_ids.shape[1]

        if sequence_length > self.config.block_size:
            raise ValueError("sequence length cannot exceed block_size")

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_embeddings = self.token_embeddings(
            token_ids,
        )
        position_embeddings = self.position_embeddings(
            positions,
        )

        return token_embeddings + position_embeddings

    def _run_blocks(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self._embed_inputs(token_ids)
        x = self._run_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
