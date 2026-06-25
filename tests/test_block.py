import torch

from llm_lab.model.block import FeedForward, TransformerBlock, TransformerStack


def test_feed_forward_preserves_shape():
    x = torch.randn(2, 3, 8)
    feedforward = FeedForward(d_model=8)
    assert x.shape == feedforward(x).shape


def test_transformer_block_preserves_shape():
    x = torch.randn(2, 3, 8)
    transformer = TransformerBlock(d_model=8, n_head=2)
    assert x.shape == transformer(x).shape


def test_transformer_block_output_is_finite():
    x = torch.randn(2, 3, 8)
    transformer = TransformerBlock(d_model=8, n_head=2)
    out = transformer(x)
    assert torch.isfinite(out).all()


def test_transformer_stack_preserves_shape():
    x = torch.randn(2, 3, 8)
    transformer = TransformerStack(d_model=8, n_head=2, n_layer=3)
    out = transformer(x)
    assert x.shape == out.shape


def test_transformer_stack_output_is_finite():
    x = torch.randn(2, 3, 8)
    transformer = TransformerStack(d_model=8, n_head=2, n_layer=3)
    out = transformer(x)
    assert torch.isfinite(out).all()


def test_transformer_stack_has_expected_number_of_layers():
    transformer = TransformerStack(d_model=8, n_head=2, n_layer=3)

    assert len(transformer.blocks) == 3
