# Phase 09: nanochat-Style Architecture Baseline

This phase added a configurable decoder-only language-model baseline while
preserving the smaller models used in earlier learning phases. The new model is
still intentionally simple, but its architecture is divided into clear
components so later phases can change normalization, positional encoding,
activation functions, and attention independently.

The complete flow is:

$$
\text{token IDs }(B,T)
\rightarrow \text{token + position embeddings }(B,T,C)
\rightarrow N\text{ Transformer blocks }(B,T,C)
\rightarrow \text{final normalization }(B,T,C)
\rightarrow \text{logits }(B,T,V)
$$

Here, $B$ is batch size, $T$ is sequence length, $C$ is model width, and $V$
is vocabulary size.

## A Separate Modern Model Configuration

`ModernModelConfig` stores the architectural choices required to construct the
modern model:

```text
vocab_size
block_size
d_model
n_head
n_layer
normalization
activation
attention
```

The configuration validates impossible values early. Sizes must be positive,
`d_model` must be divisible by `n_head`, and component names must be supported.
This prevents a malformed architecture from surviving until a later tensor
operation fails with a less useful error.

The Python model configuration has a different responsibility from the YAML
experiment configuration introduced in Phase 6. YAML describes a reproducible
run and is loaded as external data. `ModernModelConfig` is the validated Python
contract used by model constructors. A future integration layer may translate
a resolved YAML section into this dataclass, but Phase 9 does not couple those
systems yet.

## Component Factories

Factories translate a configuration choice into an executable PyTorch module:

```text
"layer_norm"  -> nn.LayerNorm
"relu"        -> nn.ReLU
"causal_mha"  -> MultiHeadCausalSelfAttention
```

This separates two decisions:

1. The configuration says which component should be used.
2. The factory knows how to construct that component.

The Transformer block therefore depends on stable builder functions rather
than containing selection logic for every possible implementation. Later
phases can add an option to a focused factory without rewriting the complete
model.

## Feed-Forward Network and Transformer Block

The feed-forward network expands and contracts the final dimension:

$$
(B,T,C) \rightarrow (B,T,4C) \rightarrow (B,T,C)
$$

Its named sequential pipeline is:

```text
up_projection -> ReLU -> down_projection
```

The Transformer block uses pre-normalized residual connections:

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

Layer normalization receives `d_model` because it normalizes the last
dimension of every token independently. Given `(B,T,C)`, it treats the tensor
as `B * T` separate vectors of length `C`; it does not normalize across tokens
or batches.

The residual path preserves the `(B,T,C)` shape and gives gradients a direct
route through the block. Tests verify shape preservation, causal behavior, and
gradient flow through every trainable parameter.

## Token and Position Embeddings

Token IDs begin with shape `(B,T)`. Token embeddings map each ID to a vector,
producing `(B,T,C)`.

Learned position embeddings provide one vector for each sequence position.
For a sequence of length `T`, `torch.arange(T)` creates:

```text
[0, 1, 2, ..., T - 1]
```

Looking up those positions produces `(T,C)`. Broadcasting adds the same
position vectors to every batch:

```text
token embeddings:    (B,T,C)
position embeddings:   (T,C)
result:              (B,T,C)
```

Token embeddings tell the model which tokens are present. Position embeddings
distinguish where those tokens occur.

## Complete Modern Language Model

The model stores Transformer blocks in a `ModuleList` and executes them in
order. Each block reads the previous hidden state and returns an updated tensor
with the same shape. A final LayerNorm stabilizes the representation after the
last residual block, and the output head maps each token vector from `C` to
`V` logits.

The model returns raw logits with shape `(B,T,V)`. Cross-entropy remains outside
the model so the same forward method can support training, validation, and
future inference code.

## Weight Tying

The output projection and token embedding use the same parameter object:

```python
output_head.weight = token_embeddings.weight
```

Both operations need a matrix with shape `(V,C)`. The embedding reads rows from
it, while the output head compares hidden vectors against it to produce
vocabulary logits. Sharing the matrix reduces parameter count and keeps the
input and output token representations connected.

Weight tying is an identity relationship, not merely equal tensor values:

```python
model.output_head.weight is model.token_embeddings.weight
```

Because the parameter is shared, it must be counted and optimized once.

## Causal and Training Compatibility

The attention mask prevents an output position from depending on future
tokens. Whole-model causality tests change later input tokens and verify that
earlier logits remain unchanged.

The modern model also works with the existing training utilities:

```text
logits (B,T,V)
-> flatten logits to (B*T,V)
-> flatten targets to (B*T)
-> cross-entropy loss
-> backward
-> AdamW update
```

Tests cover finite loss, end-to-end gradients, optimizer updates, and
validation without parameter mutation. The original `TinyLanguageModel`
continues to pass its existing model and training tests.

## Parameter Reporting

`ParameterReport` summarizes:

```text
total
trainable
frozen
by_component
```

Parameter identity is tracked so tied weights are counted once. In the
component breakdown, the shared matrix is assigned to `token_embeddings`, the
first top-level component that exposes it, while `output_head` contributes zero
additional unique parameters.

The scaling tests demonstrate three useful relationships:

```text
vocabulary increase:  delta parameters = delta vocab_size * d_model
context increase:     delta parameters = delta block_size * d_model
depth increase:       delta parameters = added layers * parameters per block
```

Vocabulary growth is counted once because of weight tying. Context length adds
learned positional vectors. Increasing depth adds independent Transformer
blocks with the same architecture.

## Why This Baseline Matters

The early learning models remain useful because their small amount of code
makes the fundamentals visible. The modern baseline serves a different
purpose: it provides stable architectural seams for controlled experiments.

Future phases can replace one choice at a time, such as LayerNorm with RMSNorm
or learned positions with RoPE, while retaining the same surrounding model and
tests. This makes any behavioral or parameter-count difference easier to
attribute to the feature being studied.

## Educational Limitations

This baseline prioritizes clarity over production throughput. Attention uses
the educational multi-head implementation from earlier phases rather than a
fused kernel. It has learned positional embeddings, ReLU, LayerNorm, equal
query/key/value head counts, and tied embeddings as fixed Phase 9 defaults.
Later phases introduce these alternatives individually so their effects remain
observable.
