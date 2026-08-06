import torch

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.modern_lm import ModernLanguageModel
from llm_lab.training.loop import (
    compute_loss,
    make_optimizer,
    train_step,
    validation_step,
)


def test_modern_language_model_supports_external_cross_entropy():
    config = ModernModelConfig(
        vocab_size=128,
        block_size=8,
        d_model=32,
        n_head=4,
        n_layer=2,
    )
    model = ModernLanguageModel(config)

    x = torch.tensor(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [20, 30, 40, 50],
            [60, 70, 80, 90],
        ],
        dtype=torch.long,
    )

    logits = model(x)
    loss = compute_loss(logits, targets)

    assert logits.shape == (
        2,
        4,
        config.vocab_size,
    )
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_modern_language_model_receives_end_to_end_gradients():
    config = ModernModelConfig(
        vocab_size=128,
        block_size=8,
        d_model=32,
        n_head=4,
        n_layer=2,
    )
    model = ModernLanguageModel(config)

    x = torch.tensor(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [20, 30, 40, 50],
            [60, 70, 80, 90],
        ],
        dtype=torch.long,
    )

    logits = model(x)
    loss = compute_loss(logits, targets)
    loss.backward()

    assert model.token_embeddings.weight.grad is not None
    assert torch.isfinite(model.token_embeddings.weight.grad).all()

    assert model.position_embeddings.weight.grad is not None
    assert torch.isfinite(model.position_embeddings.weight.grad).all()

    assert model.final_norm.weight.grad is not None
    assert torch.isfinite(model.final_norm.weight.grad).all()

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

    assert model.output_head.weight is model.token_embeddings.weight


def test_modern_language_model_supports_adamw_training_step():
    config = ModernModelConfig(
        vocab_size=128,
        block_size=8,
        d_model=32,
        n_head=4,
        n_layer=2,
    )
    model = ModernLanguageModel(config)

    optimizer = make_optimizer(
        model,
        lr=1e-2,
        weight_decay=0.0,
    )

    x = torch.tensor(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [20, 30, 40, 50],
            [60, 70, 80, 90],
        ],
        dtype=torch.long,
    )

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    shared_weight_before = model.token_embeddings.weight.detach().clone()

    loss = train_step(
        model,
        optimizer,
        x,
        targets,
    )

    parameters_after = list(model.parameters())

    at_least_one_parameter_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
            strict=True,
        )
    )

    assert torch.isfinite(loss)
    assert at_least_one_parameter_changed
    assert not torch.equal(
        shared_weight_before,
        model.token_embeddings.weight,
    )
    assert model.output_head.weight is model.token_embeddings.weight
    assert model.training


def test_modern_language_model_supports_validation_step():
    config = ModernModelConfig(
        vocab_size=128,
        block_size=8,
        d_model=32,
        n_head=4,
        n_layer=2,
    )
    model = ModernLanguageModel(config)

    x = torch.tensor(
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [20, 30, 40, 50],
            [60, 70, 80, 90],
        ],
        dtype=torch.long,
    )

    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = validation_step(
        model,
        x,
        targets,
    )

    parameters_after = list(model.parameters())

    parameters_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
            strict=True,
        )
    )

    assert torch.isfinite(loss)
    assert parameters_unchanged
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.output_head.weight is model.token_embeddings.weight
