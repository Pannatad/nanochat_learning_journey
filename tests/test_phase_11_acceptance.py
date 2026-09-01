import torch

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.inspection import build_parameter_report
from llm_lab.model.modern_lm import ModernLanguageModel
from llm_lab.model.rope import apply_rotary_embedding, build_rope_cache
from llm_lab.training.loop import compute_loss


def test_phase_11_rope_model_end_to_end():
    config = ModernModelConfig(
        vocab_size=32,
        block_size=8,
        d_model=16,
        n_head=4,
        n_layer=2,
        positional_embedding="rope",
    )
    model = ModernLanguageModel(config)
    token_ids = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])

    logits = model(token_ids)
    loss = compute_loss(logits, targets)
    loss.backward()

    assert logits.shape == (1, 4, config.vocab_size)
    assert torch.isfinite(loss)
    assert model.position_embeddings is None
    assert all(head.use_rope for block in model.blocks for head in block.attention.head)

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_phase_11_attention_interaction_depends_on_relative_position():
    sequence_length = 5
    head_dim = 4
    query = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(
        1,
        sequence_length,
        1,
    )
    key = torch.tensor([4.0, 3.0, 2.0, 1.0]).repeat(
        1,
        sequence_length,
        1,
    )
    cosine, sine = build_rope_cache(sequence_length, head_dim)

    rotated_query = apply_rotary_embedding(query, cosine, sine)
    rotated_key = apply_rotary_embedding(key, cosine, sine)
    interactions = rotated_query @ rotated_key.transpose(-2, -1)

    # Both pairs have n - m = 2, despite different absolute positions.
    torch.testing.assert_close(interactions[0, 0, 2], interactions[0, 1, 3])
    torch.testing.assert_close(interactions[0, 1, 3], interactions[0, 2, 4])


def test_phase_11_rope_removes_learned_position_parameters():
    shared_config = {
        "vocab_size": 32,
        "block_size": 8,
        "d_model": 16,
        "n_head": 4,
        "n_layer": 2,
    }
    learned_model = ModernLanguageModel(
        ModernModelConfig(**shared_config, positional_embedding="learned")
    )
    rope_model = ModernLanguageModel(
        ModernModelConfig(**shared_config, positional_embedding="rope")
    )

    learned_report = build_parameter_report(learned_model)
    rope_report = build_parameter_report(rope_model)

    expected_reduction = shared_config["block_size"] * shared_config["d_model"]
    assert learned_report.total - rope_report.total == expected_reduction
