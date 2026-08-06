import pytest
import torch
from torch import nn

from llm_lab.model.attention import MultiHeadCausalSelfAttention
from llm_lab.model.config import ModernModelConfig
from llm_lab.model.modern_block import ModernFeedForward, ModernTransformerBlock


def test_modern_feed_forward_has_expected_components():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
    )

    feed_forward = ModernFeedForward(config)

    assert isinstance(feed_forward.net, nn.Sequential)

    assert isinstance(feed_forward.net.up_projection, nn.Linear)
    assert feed_forward.net.up_projection.in_features == 64
    assert feed_forward.net.up_projection.out_features == 256

    assert isinstance(feed_forward.net.activation, nn.ReLU)

    assert isinstance(feed_forward.net.down_projection, nn.Linear)
    assert feed_forward.net.down_projection.in_features == 256
    assert feed_forward.net.down_projection.out_features == 64


def test_modern_feed_forward_preserves_hidden_shape():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    feed_forward = ModernFeedForward(config)
    x = torch.randn(2, 3, 64)

    output = feed_forward(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_modern_transformer_block_has_expected_components():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
    )

    block = ModernTransformerBlock(config)

    assert isinstance(block.norm1, nn.LayerNorm)
    assert block.norm1.normalized_shape == (64,)

    assert isinstance(
        block.attention,
        MultiHeadCausalSelfAttention,
    )
    assert block.attention.d_model == 64
    assert block.attention.n_head == 4

    assert isinstance(block.norm2, nn.LayerNorm)
    assert block.norm2.normalized_shape == (64,)

    assert block.norm1 is not block.norm2
    assert isinstance(block.mlp, ModernFeedForward)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length"),
    [
        (1, 1),
        (2, 3),
        (4, 5),
    ],
)
def test_modern_transformer_block_preserves_hidden_shape(
    batch_size: int,
    sequence_length: int,
):
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    block = ModernTransformerBlock(config)
    x = torch.randn(
        batch_size,
        sequence_length,
        64,
    )
    original_x = x.clone()

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.equal(x, original_x)


def test_modern_transformer_block_supports_backpropagation():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    block = ModernTransformerBlock(config)

    x = torch.randn(
        2,
        3,
        64,
        requires_grad=True,
    )

    output = block(x)
    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    parameter_gradients = [
        parameter.grad for parameter in block.parameters() if parameter.requires_grad
    ]

    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in parameter_gradients
        if gradient is not None
    )
