# Phase 03: Manual Attention Learning

## What We Built

We added a causal mask and a single-head causal self-attention module. The
attention module creates query, key, and value vectors, computes attention
scores, blocks future tokens with a causal mask, applies softmax, and mixes the
value vectors.

## What I Learned

- I learned that `torch.equal(x, y)` can be used to compare two tensors.
- I learned that `torch.tril()` can create the lower-triangular mask used for
  causal attention.
- I revised the attention workflow: the shapes of `q`, `k`, and `v`, the order
  of multiplication, masking, softmax, and value mixing.

## High-Level Understanding

Causal self-attention lets each token look at previous tokens and itself, but
not future tokens. This is what makes the model predict left to right.

The attention flow is:

```text
x -> q, k, v -> scores -> causal mask -> softmax -> weighted value sum
```

## Intuition / Small Example

For a sequence of length 3, the causal mask is:

```python
[
    [True, False, False],
    [True, True, False],
    [True, True, True],
]
```

This means:

- token 0 can only look at token 0;
- token 1 can look at token 0 and token 1;
- token 2 can look at token 0, token 1, and token 2.

After masking and softmax, the attention weights might look like:

```python
[
    [1.0, 0.0, 0.0],
    [0.4, 0.6, 0.0],
    [0.2, 0.3, 0.5],
]
```

The upper-right values are zero because those are future-token positions.

## Detailed Explanation

The input to attention has shape:

```text
(B, T, d_model)
```

The query, key, and value projections keep the same shape:

```text
q: (B, T, d_model)
k: (B, T, d_model)
v: (B, T, d_model)
```

Attention scores are computed with:

```python
scores = q @ k.transpose(-2, -1)
```

The shape becomes:

```text
(B, T, T)
```

Each token now has one score for every token position. The causal mask replaces
future-token scores with `-inf`, so softmax turns those positions into zero
probability.

Then:

```python
weights = scores.softmax(dim=-1)
out = weights @ v
```

The final output returns to:

```text
(B, T, d_model)
```

## Experiments To Try

- Print the attention weights for a sequence of length 3.
- Check that `weights[0].triu(diagonal=1)` is all zeros.
- Change `d_model` and confirm the output shape still matches the input shape.

## Tests / Checks

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

Expected result:

- causal mask test passes;
- attention output shape test passes;
- future-token masking test passes;
- Ruff reports all checks passed.
