import pytest
import torch

from llm_lab.model.rope import apply_rotary_embedding, build_rope_cache


def test_build_rope_cache_matches_manual_angles():
    cosine, sine = build_rope_cache(
        sequence_length=3,
        head_dim=4,
    )

    frequencies = torch.tensor([1.0, 0.01])
    positions = torch.tensor([0.0, 1.0, 2.0])
    expected_angles = positions[:, None] * frequencies[None, :]

    torch.testing.assert_close(cosine, expected_angles.cos())
    torch.testing.assert_close(sine, expected_angles.sin())


def test_build_rope_cache_has_one_rotation_per_feature_pair():
    cosine, sine = build_rope_cache(
        sequence_length=5,
        head_dim=8,
    )

    assert cosine.shape == (5, 4)
    assert sine.shape == (5, 4)
    assert cosine.dtype == torch.float32
    assert sine.dtype == torch.float32


def test_build_rope_cache_position_zero_is_identity_rotation():
    cosine, sine = build_rope_cache(
        sequence_length=4,
        head_dim=6,
    )

    torch.testing.assert_close(cosine[0], torch.ones(3))
    torch.testing.assert_close(sine[0], torch.zeros(3))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"sequence_length": 0, "head_dim": 4}, "sequence_length"),
        ({"sequence_length": 2, "head_dim": 0}, "head_dim"),
        ({"sequence_length": 2, "head_dim": 3}, "even head dimension"),
        ({"sequence_length": 2, "head_dim": 4, "base": 0.0}, "base"),
    ],
)
def test_build_rope_cache_rejects_invalid_inputs(arguments, message):
    with pytest.raises(ValueError, match=message):
        build_rope_cache(**arguments)


def test_apply_rotary_embedding_matches_manual_pair_rotations():
    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
            ]
        ]
    )
    cosine = torch.tensor([[0.0, -1.0]])
    sine = torch.tensor([[1.0, 0.0]])

    output = apply_rotary_embedding(x, cosine, sine)

    expected = torch.tensor(
        [
            [
                [-2.0, 1.0, -3.0, -4.0],
            ]
        ]
    )
    torch.testing.assert_close(output, expected)


def test_apply_rotary_embedding_preserves_shape_and_pair_magnitudes():
    x = torch.randn(2, 5, 8)
    cosine, sine = build_rope_cache(sequence_length=5, head_dim=8)

    output = apply_rotary_embedding(x, cosine, sine)

    assert output.shape == x.shape
    input_pair_magnitudes = x.reshape(2, 5, 4, 2).square().sum(dim=-1)
    output_pair_magnitudes = output.reshape(2, 5, 4, 2).square().sum(dim=-1)
    torch.testing.assert_close(output_pair_magnitudes, input_pair_magnitudes)


def test_apply_rotary_embedding_leaves_position_zero_unchanged():
    x = torch.randn(3, 1, 6)
    cosine, sine = build_rope_cache(sequence_length=1, head_dim=6)

    output = apply_rotary_embedding(x, cosine, sine)

    torch.testing.assert_close(output, x)


def test_apply_rotary_embedding_broadcasts_cache_across_batches():
    first_batch = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    x = torch.cat((first_batch, first_batch * 2), dim=0)
    cosine = torch.tensor([[0.0, -1.0]])
    sine = torch.tensor([[1.0, 0.0]])

    output = apply_rotary_embedding(x, cosine, sine)

    torch.testing.assert_close(output[1], output[0] * 2)


@pytest.mark.parametrize(
    ("x", "cosine", "sine", "message"),
    [
        (torch.randn(2, 4), torch.randn(2, 2), torch.randn(2, 2), "x must"),
        (
            torch.randn(1, 2, 3),
            torch.randn(2, 1),
            torch.randn(2, 1),
            "even head dimension",
        ),
        (
            torch.randn(1, 2, 4),
            torch.randn(1, 2),
            torch.randn(2, 2),
            "cosine must",
        ),
        (
            torch.randn(1, 2, 4),
            torch.randn(2, 2),
            torch.randn(1, 2),
            "sine must",
        ),
    ],
)
def test_apply_rotary_embedding_rejects_invalid_shapes(
    x,
    cosine,
    sine,
    message,
):
    with pytest.raises(ValueError, match=message):
        apply_rotary_embedding(x, cosine, sine)
