import torch
import pytest

from llm_lab.model.normalization import RMSNorm


def test_rms_norm_matches_manual_calculation():
    normalization = RMSNorm(
        d_model=2,
        eps=0.0,
    )

    x = torch.tensor(
        [[[3.0, 4.0]]],
    )

    output = normalization(x)

    # Calculate sqrt((3² + 4²) / 2).
    expected_rms = torch.sqrt(torch.tensor(25.0 / 2.0))

    expected = torch.tensor(
        [[[3.0 / expected_rms, 4.0 / expected_rms]]],
    )

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "d_model"),
    [
        (1, 1, 2),
        (2, 3, 4),
        (4, 5, 8),
    ],
)
def test_rms_norm_preserves_shape(
    batch_size: int,
    sequence_length: int,
    d_model: int,
):
    normalization = RMSNorm(d_model=d_model)

    x = torch.randn(
        batch_size,
        sequence_length,
        d_model,
    )

    output = normalization(x)

    # Assert that output has exactly the same shape as x.
    assert output.shape == x.shape

    # Assert that every output value is finite.
    assert torch.isfinite(output).all()


def test_rms_norm_handles_zero_input():
    normalization = RMSNorm(d_model=4)

    x = torch.zeros(
        2,
        3,
        4,
    )

    output = normalization(x)

    assert torch.isfinite(output).all()
    expected = torch.zeros_like(x)

    torch.testing.assert_close(output, expected)


def test_rms_norm_applies_learned_weight():
    normalization = RMSNorm(
        d_model=3,
        eps=0.0,
    )

    with torch.no_grad():
        normalization.weight.copy_(
            torch.tensor([2.0, 0.5, -1.0]),
        )

    x = torch.ones(
        1,
        1,
        3,
    )

    output = normalization(x)

    expected = torch.tensor(
        [[[2.0, 0.5, -1.0]]],
    )

    torch.testing.assert_close(output, expected)


def test_rms_norm_supports_backpropagation():
    normalization = RMSNorm(d_model=4)

    x = torch.randn(
        2,
        3,
        4,
        requires_grad=True,
    )

    output = normalization(x)

    loss = output.square().mean()
    loss.backward()

    # Verify that x received a gradient.
    assert x.grad is not None
    # Verify that weight received a gradient.
    assert normalization.weight.grad is not None
    # Verify that both gradients contain only finite values.
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(normalization.weight.grad).all()
