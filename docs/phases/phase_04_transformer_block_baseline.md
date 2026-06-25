# Phase 04: Transformer Block Baseline

## What We Built

We built the baseline Transformer block. It combines multi-head causal
attention, an MLP, LayerNorm, residual connections, and a stack of repeated
blocks.

## What I Learned

- I learned the full Transformer block structure:
  `LayerNorm -> Attention -> residual add -> LayerNorm -> MLP -> residual add`.
- I learned that residual connections add the branch output back to the main
  stream.
- I learned how multi-head attention splits the channel dimension into several
  heads, runs attention per head, concatenates the outputs, and uses a final
  linear layer to blend the concatenated information.
- I learned that the MLP expands from `d_model` to `4 * d_model`, applies
  `ReLU`, then projects back to `d_model`.
- I learned that `nn.LayerNorm(d_model)` normalizes the last dimension of the
  tensor.
- I learned how `nn.ModuleList` can store `N` Transformer blocks for a stack.

## High-Level Understanding

A Transformer block updates the residual stream while keeping the tensor shape
the same. Attention mixes information across token positions. The MLP updates
each token's feature vector. Residual connections keep the original stream
available and make stacking blocks possible.

## Intuition / Small Example

The block flow is:

```text
x
|
|---- LayerNorm -> Multi-head attention ----|
|                                           |
+ <-----------------------------------------|
|
|---- LayerNorm -> MLP ---------------------|
|                                           |
+ <-----------------------------------------|
|
output
```

With stacked blocks:

```text
x -> block 1 -> block 2 -> ... -> block N -> output
```

Each block preserves:

```text
(B, T, C) -> (B, T, C)
```

## Detailed Explanation

### Multi-Head Attention

The input has shape:

```text
(B, T, C)
```

For `n_head` heads, each head receives:

```text
(B, T, C / n_head)
```

Each head runs causal self-attention. The outputs are concatenated back into:

```text
(B, T, C)
```

Then a final linear projection blends the concatenated head information.

### MLP

The MLP works on the last dimension:

```text
(B, T, d_model)
-> Linear(d_model, 4 * d_model)
-> ReLU
-> Linear(4 * d_model, d_model)
-> (B, T, d_model)
```

### LayerNorm

`nn.LayerNorm(d_model)` normalizes the last dimension. For example:

```text
(2, 4, 6, 8)
```

means LayerNorm would normalize each vector of length `8`.

### Transformer Block

The block uses pre-norm structure:

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

The residual additions require every branch to return the same shape as `x`.

## Experiments To Try

- Change `n_head` and confirm `d_model` must divide evenly by `n_head`.
- Change the number of stacked blocks and confirm the output shape is unchanged.
- Print the tensor shape after attention, MLP, and each residual add.

## Tests / Checks

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

Expected result:

- feedforward shape test passes;
- multi-head attention shape test passes;
- Transformer block shape and finite-output tests pass;
- Transformer stack shape, finite-output, and layer-count tests pass;
- Ruff reports all checks passed.
