import torch

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.modern_lm import ModernLanguageModel
from llm_lab.training.loop import compute_loss
from llm_lab.model.inspection import build_parameter_report


def test_phase_10_rms_norm_model_end_to_end():
    config = ModernModelConfig(
        vocab_size=32,
        block_size=8,
        d_model=16,
        n_head=4,
        n_layer=2,
        normalization="rms_norm",
    )
    model = ModernLanguageModel(config)

    token_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[2, 3, 4, 5]],
        dtype=torch.long,
    )

    logits = model(token_ids)
    loss = compute_loss(logits, targets)
    loss.backward()

    assert logits.shape == (
        1,
        4,
        config.vocab_size,
    )
    assert torch.isfinite(loss)

    parameter_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]

    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in parameter_gradients
        if gradient is not None
    )


def test_phase_10_rms_norm_parameter_count_against_layer_norm():
    layer_norm_config = ModernModelConfig(
        vocab_size=32,
        block_size=8,
        d_model=16,
        n_head=4,
        n_layer=2,
        normalization="layer_norm",
    )
    rms_norm_config = ModernModelConfig(
        vocab_size=32,
        block_size=8,
        d_model=16,
        n_head=4,
        n_layer=2,
        normalization="rms_norm",
    )

    layer_norm_model = ModernLanguageModel(layer_norm_config)
    rms_norm_model = ModernLanguageModel(rms_norm_config)

    layer_norm_report = build_parameter_report(layer_norm_model)
    rms_norm_report = build_parameter_report(rms_norm_model)

    expected_reduction = (2 * rms_norm_config.n_layer + 1) * rms_norm_config.d_model

    assert layer_norm_report.total - rms_norm_report.total == expected_reduction
