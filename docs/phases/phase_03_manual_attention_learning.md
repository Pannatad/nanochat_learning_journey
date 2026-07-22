# Phase 03: Manual Attention Learning

This phase introduced causal self-attention. Attention is the mechanism that
lets each token mix information from earlier tokens in the same sequence.

The attention flow is:

$$
x
\rightarrow (q, k, v)
\rightarrow \text{scores}
\rightarrow \text{causal mask}
\rightarrow \text{softmax}
\rightarrow \text{weighted value sum}
$$

## Causal Attention

Causal attention means a token can look backward, but not forward.

For a sequence of length `3`, the causal mask is:

$$
\begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$

This means:

$$
\begin{aligned}
\text{token }0 &\rightarrow \{0\} \\
\text{token }1 &\rightarrow \{0, 1\} \\
\text{token }2 &\rightarrow \{0, 1, 2\}
\end{aligned}
$$

Future positions are blocked. This is required for next-token prediction
because the model should not cheat by seeing tokens that come later.

`torch.tril()` creates the lower-triangular mask:

```python
torch.tril(torch.ones((T, T), dtype=torch.bool))
```

## Query, Key, And Value

Attention creates three projections from the input:

```text
q = query vectors
k = key vectors
v = value vectors
```

The input shape is:

$$
(B, T, d_{\text{model}})
$$

The query, key, and value shapes are:

$$
q, k, v \in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

The query asks what a token is looking for. The key describes what each token
offers. The value is the information that gets mixed together.

## Attention Scores

Scores are computed with:

$$
\text{scores} = qk^\top
$$

Shape:

$$
(B, T, d_{\text{model}})
\times
(B, d_{\text{model}}, T)
\rightarrow
(B, T, T)
$$

Each token now has one score for every token position.

The scores are scaled by:

$$
\sqrt{d_{\text{model}}}
$$

Scaling keeps the scores from becoming too large before softmax.

## Masking And Softmax

Future-token scores are replaced with `-inf`.

After softmax, those positions become probability `0`.

Example attention weights for length `3`:

$$
\begin{bmatrix}
1.0 & 0.0 & 0.0 \\
0.4 & 0.6 & 0.0 \\
0.2 & 0.3 & 0.5
\end{bmatrix}
$$

The upper-right triangle is zero because those are future-token positions.

## Mixing Values

The final attention output is:

$$
\text{out} = \text{weights} \cdot v
$$

Shape:

$$
(B, T, T)
\times
(B, T, d_{\text{model}})
\rightarrow
(B, T, d_{\text{model}})
$$

So attention starts and ends with the same shape:

$$
(B, T, d_{\text{model}})
\rightarrow
(B, T, d_{\text{model}})
$$

This matters because the attention output can be added back to the residual
stream in later Transformer blocks.

## Small Example

For token 2 in a length-3 sequence, the model can combine information from:

```text
token 0
token 1
token 2
```

If its attention weights are:

$$
[0.2, 0.3, 0.5]
$$

then its output is roughly:

$$
0.2v_0 + 0.3v_1 + 0.5v_2
$$

The token is not just copied forward. It becomes a weighted mixture of the
allowed context.
