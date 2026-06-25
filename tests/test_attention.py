import torch

from llm_lab.model.attention import SingleHeadCausalSelfAttention, make_causal_mask


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


def test_single_head_attention_cannot_attend_to_future_tokens():
    x = torch.rand(1, 3, 4)
    attention = SingleHeadCausalSelfAttention(d_model=4)

    _, weights = attention(x, return_attention_weights=True)

    assert weights.shape == (1, 3, 3)
    assert torch.all(weights[0].triu(diagonal=1) == 0)
