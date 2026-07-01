import torch

from llm_lab.model.tiny_lm import TinyLanguageModel
from llm_lab.training.loop import (
    compute_loss,
    generate,
    make_optimizer,
    train_step,
    validation_step,
)


def test_compute_loss_returns_finite_scalar():
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor(
        [
            [1, 2, 3],
            [0, 4, 1],
        ]
    )
    loss = compute_loss(logits, targets)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_make_optimizer_returns_adamw():
    model = TinyLanguageModel(vocab_size=256, d_model=16)

    optimizer = make_optimizer(model, lr=1e-3, weight_decay=0.01)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_train_step_updates_model_parameters():
    model = TinyLanguageModel(vocab_size=10, d_model=8)
    optimizer = make_optimizer(model, lr=1e-2, weight_decay=0.0)

    x = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    y = torch.tensor(
        [
            [2, 3, 4],
            [5, 6, 7],
        ]
    )

    before = [param.detach().clone() for param in model.parameters()]

    loss = train_step(model, optimizer, x, y)

    after = list(model.parameters())
    changed = any(
        not torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, after)
    )

    assert torch.isfinite(loss)
    assert changed


def test_validation_step_returns_finite_scalar_without_updating_parameters():
    model = TinyLanguageModel(vocab_size=10, d_model=8)
    x = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    y = torch.tensor(
        [
            [2, 3, 4],
            [5, 6, 7],
        ]
    )

    before = [param.detach().clone() for param in model.parameters()]

    loss = validation_step(model, x, y)

    after = list(model.parameters())
    unchanged = all(
        torch.equal(before_param, after_param)
        for before_param, after_param in zip(before, after)
    )

    assert torch.isfinite(loss)
    assert unchanged


def test_generate_appends_requested_number_of_tokens():
    model = TinyLanguageModel(vocab_size=10, d_model=8)
    start_token_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    generated = generate(model, start_token_ids, max_tokens=2)

    assert generated.shape == (2, 5)
    assert torch.equal(generated[:, :3], start_token_ids)
