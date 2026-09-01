import pytest
import torch

import llm_lab.model.attention as attention_module
from llm_lab.model.attention import (
    MultiHeadCausalSelfAttention,
    SingleHeadCausalSelfAttention,
    make_causal_mask,
)


def test_causal_mask():
    tril = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )
    assert torch.equal(tril, make_causal_mask(3))


def test_causal_mask_shape_changes_with_sequence_length():
    mask = make_causal_mask(4)

    assert mask.shape == (4, 4)
    assert not mask[0, 1]
    assert mask[3, 0]


def test_single_head_attention_output_shape():
    x = torch.rand(2, 3, 4)
    attention = SingleHeadCausalSelfAttention(d_model=4)
    output = attention(x)
    assert output.shape == x.shape


def test_single_head_attention_rope_output_is_finite_and_preserves_shape():
    x = torch.rand(2, 3, 4)
    attention = SingleHeadCausalSelfAttention(d_model=4, use_rope=True)

    output = attention(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_single_head_attention_rotates_projected_query_and_key_separately(
    monkeypatch,
):
    x = torch.rand(1, 3, 4)
    attention = SingleHeadCausalSelfAttention(d_model=4, use_rope=True)
    projected_query = attention.query(x)
    projected_key = attention.key(x)
    rotated_inputs = []

    def record_rotation(tensor, cosine, sine):
        rotated_inputs.append(tensor)
        return tensor

    monkeypatch.setattr(
        attention_module,
        "apply_rotary_embedding",
        record_rotation,
    )

    attention(x)

    assert len(rotated_inputs) == 2
    torch.testing.assert_close(rotated_inputs[0], projected_query)
    torch.testing.assert_close(rotated_inputs[1], projected_key)


def test_single_head_attention_rope_requires_even_head_dimension():
    with pytest.raises(ValueError, match="even head dimension"):
        SingleHeadCausalSelfAttention(d_model=3, use_rope=True)


def test_single_head_attention_cannot_attend_to_future_tokens():
    x = torch.rand(1, 3, 4)
    attention = SingleHeadCausalSelfAttention(d_model=4)

    _, weights = attention(x, return_attention_weights=True)

    assert weights.shape == (1, 3, 3)
    assert torch.all(weights[0].triu(diagonal=1) == 0)


def test_multi_head_attention_output_shape():
    x = torch.rand(1, 3, 8)
    attention = MultiHeadCausalSelfAttention(n_head=2, d_model=8)
    out = attention(x)
    assert x.shape == out.shape


def test_multi_head_attention_requires_even_head_split():
    with pytest.raises(ValueError):
        MultiHeadCausalSelfAttention(d_model=10, n_head=3)
