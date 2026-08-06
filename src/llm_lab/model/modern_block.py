from collections import OrderedDict

import torch
from torch import nn

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.factories import (
    build_activation,
    build_attention,
    build_normalization,
)


class ModernFeedForward(nn.Module):
    def __init__(self, config: ModernModelConfig):
        super().__init__()
        hidden_dim = 4 * config.d_model

        self.net = nn.Sequential(
            OrderedDict(
                [
                    (
                        "up_projection",
                        nn.Linear(
                            in_features=config.d_model,
                            out_features=hidden_dim,
                        ),
                    ),
                    (
                        "activation",
                        build_activation(
                            kind=config.activation,
                        ),
                    ),
                    (
                        "down_projection",
                        nn.Linear(
                            in_features=hidden_dim,
                            out_features=config.d_model,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModernTransformerBlock(nn.Module):
    def __init__(self, config: ModernModelConfig):
        super().__init__()

        self.norm1 = build_normalization(
            kind=config.normalization,
            d_model=config.d_model,
        )

        self.attention = build_attention(
            kind=config.attention,
            d_model=config.d_model,
            n_head=config.n_head,
        )

        self.norm2 = build_normalization(
            kind=config.normalization,
            d_model=config.d_model,
        )

        self.mlp = ModernFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
