import torch

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.inspection import build_parameter_report
from llm_lab.model.modern_lm import ModernLanguageModel
from llm_lab.training.loop import compute_loss


def test_phase_09_modern_baseline_end_to_end():
    config = ModernModelConfig(
        vocab_size=32,
        block_size=8,
        d_model=16,
        n_head=4,
        n_layer=2,
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

    report = build_parameter_report(model)

    assert logits.shape == (
        1,
        4,
        config.vocab_size,
    )
    assert torch.isfinite(loss)

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert model.output_head.weight is model.token_embeddings.weight

    assert report.total > 0
    assert report.trainable == report.total
    assert report.frozen == 0
    assert sum(report.by_component.values()) == report.total
