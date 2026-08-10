import pytest
import torch
from torch import nn

from llm_lab.model.attention import MultiHeadCausalSelfAttention
from llm_lab.model.factories import (
    build_activation,
    build_attention,
    build_normalization,
)
from llm_lab.model.normalization import RMSNorm


def test_build_normalization_creates_layer_norm():
    normalization = build_normalization(
        kind="layer_norm",
        d_model=64,
    )

    assert isinstance(normalization, nn.LayerNorm)
    assert normalization.normalized_shape == (64,)


def test_build_normalization_creates_rms_norm():
    normalization = build_normalization(
        kind="rms_norm",
        d_model=64,
    )

    assert isinstance(normalization, RMSNorm)
    assert normalization.weight.shape == (64,)


def test_build_normalization_rejects_unknown_kind():
    with pytest.raises(ValueError, match="normalization"):
        build_normalization(
            kind="unknown_norm",
            d_model=64,
        )


def test_built_normalization_preserves_shape():
    normalization = build_normalization(
        kind="layer_norm",
        d_model=64,
    )
    x = torch.randn(2, 3, 64)

    output = normalization(x)

    assert output.shape == (2, 3, 64)
    assert torch.isfinite(output).all()


def test_build_activation_creates_relu():
    activation = build_activation(kind="relu")

    assert isinstance(activation, nn.ReLU)


def test_built_relu_applies_elementwise_activation():
    activation = build_activation(kind="relu")
    x = torch.tensor([[[-2.0, 0.0, 3.0]]])

    output = activation(x)

    expected = torch.tensor([[[0.0, 0.0, 3.0]]])

    assert output.shape == x.shape
    assert torch.equal(output, expected)


def test_build_activation_rejects_unknown_kind():
    with pytest.raises(ValueError, match="activation"):
        build_activation(kind="unknown_activation")


def test_build_attention_creates_multi_head_causal_attention():
    attention = build_attention(
        kind="causal_mha",
        d_model=64,
        n_head=4,
    )

    assert isinstance(attention, MultiHeadCausalSelfAttention)
    assert attention.d_model == 64
    assert attention.n_head == 4
    assert attention.head_dim == 16


def test_built_attention_preserves_hidden_shape():
    attention = build_attention(
        kind="causal_mha",
        d_model=64,
        n_head=4,
    )
    x = torch.randn(2, 3, 64)

    output = attention(x)

    assert output.shape == (2, 3, 64)
    assert torch.isfinite(output).all()


def test_build_attention_rejects_unknown_kind():
    with pytest.raises(ValueError, match="attention"):
        build_attention(
            kind="unknown_attention",
            d_model=64,
            n_head=4,
        )
