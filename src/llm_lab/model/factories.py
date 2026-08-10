from torch import nn

from llm_lab.model.attention import MultiHeadCausalSelfAttention
from llm_lab.model.normalization import RMSNorm


def build_normalization(kind: str, d_model: int) -> nn.Module:
    if kind == "layer_norm":
        return nn.LayerNorm(d_model)
    if kind == "rms_norm":
        return RMSNorm(d_model=d_model)

    raise ValueError(f"unsupported normalization: {kind}")


def build_activation(kind: str) -> nn.Module:
    if kind == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation: {kind}")


def build_attention(
    kind: str,
    d_model: int,
    n_head: int,
) -> nn.Module:
    if kind == "causal_mha":
        return MultiHeadCausalSelfAttention(
            n_head=n_head,
            d_model=d_model,
        )

    raise ValueError(f"unsupported attention: {kind}")
