# Phase 04: Transformer Block Baseline

This phase combined attention with the other core pieces of a Transformer
block: multi-head attention, an MLP, LayerNorm, residual connections, and a
stack of repeated blocks.

The block structure is:

$$
x
\rightarrow \text{LayerNorm}
\rightarrow \text{multi-head causal attention}
\rightarrow \text{residual add}
\rightarrow \text{LayerNorm}
\rightarrow \text{MLP}
\rightarrow \text{residual add}
$$

## Residual Stream

The residual stream is the main tensor that flows through the Transformer.

Shape:

$$
(B, T, d_{\text{model}})
$$

Each block updates this stream but keeps the same shape. That makes it possible
to stack many blocks.

Residual connections add a branch output back to the main stream:

$$
x \leftarrow x + \text{attention}(\text{norm}_1(x))
$$

$$
x \leftarrow x + \text{MLP}(\text{norm}_2(x))
$$

The branch output must have the same shape as `x`.

## Multi-Head Attention

Single-head attention computes one attention pattern. Multi-head attention
splits the channel dimension into multiple smaller heads.

Example:

$$
d_{\text{model}} = 8,\quad n_{\text{head}} = 2,\quad d_{\text{head}} = 4
$$

The input is split from:

$$
(B, T, 8)
$$

into two heads:

$$
(B, T, 4),\quad (B, T, 4)
$$

Each head runs causal self-attention. The outputs are concatenated back into:

$$
(B, T, 8)
$$

A final linear projection blends the concatenated heads.

`d_model` must divide evenly by `n_head`, otherwise the channels cannot be split
equally.

## MLP

The MLP updates each token vector independently. It does not mix positions like
attention does.

The shape flow is:

$$
(B, T, d_{\text{model}})
\rightarrow \text{Linear}(d_{\text{model}}, 4d_{\text{model}})
\rightarrow \text{ReLU}
\rightarrow \text{Linear}(4d_{\text{model}}, d_{\text{model}})
\rightarrow (B, T, d_{\text{model}})
$$

The expansion to `4 * d_model` gives the block more capacity to transform each
token's features before projecting back to the residual stream size.

## LayerNorm

`nn.LayerNorm(d_model)` normalizes the last dimension.

Example shape:

$$
(2, 4, 6, 8)
$$

LayerNorm normalizes each vector of length `8`.

This phase uses pre-norm structure, meaning normalization happens before the
attention or MLP branch:

$$
\text{LayerNorm}
\rightarrow \text{branch}
\rightarrow \text{residual add}
$$

Pre-norm is common because it makes deeper Transformer stacks easier to train.

## Transformer Stack

A stack repeats the same block structure multiple times:

$$
x
\rightarrow \text{block}_1
\rightarrow \text{block}_2
\rightarrow \text{block}_3
\rightarrow \text{output}
$$

`nn.ModuleList` stores the blocks so PyTorch can track their parameters.

Every block preserves shape:

$$
(B, T, d_{\text{model}})
\rightarrow
(B, T, d_{\text{model}})
$$

This is the main contract that makes the stack simple.

## Small Example

With:

$$
B = 2,\quad T = 3,\quad d_{\text{model}} = 8,\quad n_{\text{head}} = 2
$$

the attention heads each receive:

$$
(2, 3, 4)
$$

After attention, concatenation, projection, MLP, and residual adds, the block
still returns:

$$
(2, 3, 8)
$$

The block changes the values, not the outer shape.
