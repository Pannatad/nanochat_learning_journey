import pytest
import torch
from torch import nn

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.modern_block import ModernTransformerBlock
from llm_lab.model.modern_lm import ModernLanguageModel
from llm_lab.model.normalization import RMSNorm


def test_modern_language_model_has_expected_components():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
    )

    model = ModernLanguageModel(config)

    assert isinstance(model.token_embeddings, nn.Embedding)
    assert model.token_embeddings.num_embeddings == 512
    assert model.token_embeddings.embedding_dim == 64

    assert isinstance(model.position_embeddings, nn.Embedding)
    assert model.position_embeddings.num_embeddings == 128
    assert model.position_embeddings.embedding_dim == 64

    assert isinstance(model.blocks, nn.ModuleList)
    assert len(model.blocks) == 3
    assert all(isinstance(block, ModernTransformerBlock) for block in model.blocks)
    assert len({id(block) for block in model.blocks}) == 3

    assert isinstance(model.final_norm, nn.LayerNorm)
    assert model.final_norm.normalized_shape == (64,)

    assert isinstance(model.output_head, nn.Linear)
    assert model.output_head.in_features == 64
    assert model.output_head.out_features == 512
    assert model.output_head.bias is None


def test_modern_language_model_combines_token_and_position_embeddings():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)
    token_ids = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60],
        ]
    )

    hidden = model._embed_inputs(token_ids)

    positions = torch.arange(3)
    expected = model.token_embeddings(token_ids) + model.position_embeddings(positions)

    assert hidden.shape == (2, 3, 64)
    assert torch.isfinite(hidden).all()
    torch.testing.assert_close(hidden, expected)


def test_embed_inputs_requires_batch_and_sequence_dimensions():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=4,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)
    token_ids = torch.tensor([10, 20, 30])

    with pytest.raises(ValueError, match=r"\(B, T\)"):
        model._embed_inputs(token_ids)


def test_embed_inputs_requires_long_token_ids():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=4,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)
    token_ids = torch.tensor([[10.0, 20.0, 30.0]])

    with pytest.raises(TypeError, match="torch.long"):
        model._embed_inputs(token_ids)


def test_embed_inputs_rejects_sequence_longer_than_block_size():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=4,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)
    token_ids = torch.zeros(
        (1, 5),
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="block_size"):
        model._embed_inputs(token_ids)


def test_embed_inputs_accepts_sequence_at_block_size():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=4,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)
    token_ids = torch.zeros(
        (1, 4),
        dtype=torch.long,
    )

    hidden = model._embed_inputs(token_ids)

    assert hidden.shape == (1, 4, 64)
    assert torch.isfinite(hidden).all()


def test_modern_language_model_runs_hidden_states_through_blocks():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=8,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)
    token_ids = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60],
        ],
        dtype=torch.long,
    )

    embedded = model._embed_inputs(token_ids)
    original_embedded = embedded.clone()

    contextual = model._run_blocks(embedded)

    assert contextual.shape == embedded.shape
    assert torch.isfinite(contextual).all()
    assert torch.equal(embedded, original_embedded)

    expected = original_embedded
    for block in model.blocks:
        expected = block(expected)

    torch.testing.assert_close(contextual, expected)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length"),
    [
        (1, 1),
        (2, 3),
        (4, 5),
        (1, 8),
    ],
)
def test_modern_language_model_returns_vocabulary_logits(
    batch_size: int,
    sequence_length: int,
):
    config = ModernModelConfig(
        vocab_size=512,
        block_size=8,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)

    token_ids = torch.arange(
        batch_size * sequence_length,
        dtype=torch.long,
    ).reshape(
        batch_size,
        sequence_length,
    )

    logits = model(token_ids)

    assert logits.shape == (
        batch_size,
        sequence_length,
        config.vocab_size,
    )
    assert torch.is_floating_point(logits)
    assert torch.isfinite(logits).all()


def test_modern_language_model_forward_matches_explicit_pipeline():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=8,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)
    token_ids = torch.tensor(
        [[10, 20, 30]],
        dtype=torch.long,
    )

    logits = model(token_ids)

    x = model._embed_inputs(token_ids)
    x = model._run_blocks(x)
    x = model.final_norm(x)
    expected = model.output_head(x)

    torch.testing.assert_close(logits, expected)


def test_modern_language_model_ties_embedding_and_output_weights():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=8,
        d_model=64,
        n_head=4,
        n_layer=3,
    )
    model = ModernLanguageModel(config)

    assert model.output_head.weight is model.token_embeddings.weight

    assert (
        model.output_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr()
    )


def test_modern_language_model_cannot_use_future_tokens():
    config = ModernModelConfig(
        vocab_size=128,
        block_size=8,
        d_model=32,
        n_head=4,
        n_layer=2,
    )
    model = ModernLanguageModel(config)
    model.eval()

    sequence_a = torch.tensor(
        [[10, 20, 30, 40]],
        dtype=torch.long,
    )
    sequence_b = torch.tensor(
        [[10, 20, 99, 88]],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits_a = model(sequence_a)
        logits_b = model(sequence_b)

    shared_prefix_length = 2

    torch.testing.assert_close(
        logits_a[:, :shared_prefix_length, :],
        logits_b[:, :shared_prefix_length, :],
    )


def test_modern_language_model_uses_rms_norm_when_configured():
    config = ModernModelConfig(
        vocab_size=128,
        block_size=8,
        d_model=32,
        n_head=4,
        n_layer=2,
        normalization="rms_norm",
    )
    model = ModernLanguageModel(config)

    assert isinstance(model.final_norm, RMSNorm)
    assert all(
        isinstance(block.norm1, RMSNorm) and isinstance(block.norm2, RMSNorm)
        for block in model.blocks
    )

    token_ids = torch.tensor(
        [[10, 20, 30, 40]],
        dtype=torch.long,
    )

    logits = model(token_ids)

    assert logits.shape == (1, 4, config.vocab_size)
    assert torch.isfinite(logits).all()
