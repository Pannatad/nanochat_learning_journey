# Phase 10: RMSNorm

This phase adds RMSNorm as an optional normalization method for the modern
language model. LayerNorm remains the default so the Phase 9 baseline continues
to behave as before.

The main goal is to change one architectural component at a time:

```text
same model, data, optimizer, and training steps
                    ↓
        LayerNorm versus RMSNorm
```

## LayerNorm and RMSNorm

LayerNorm recenters and rescales a hidden vector. Before applying its learned
scale and bias, it subtracts the mean and divides by the standard deviation:

$$
\mathrm{LayerNorm}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
$$

RMSNorm does not subtract the mean. It controls only the vector's overall
magnitude:

$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{C}\sum_{i=1}^{C}x_i^2+\epsilon}}
$$

Here, $C$ is `d_model` and $\gamma$ is the learned feature weight. Dividing by
the RMS preserves the vector's direction before $\gamma$ is applied. Because
$\gamma$ contains a separate value for every feature, it can then change the
relative feature scales.

## Tensor Contract

Given a hidden-state tensor:

```text
x: (B,T,C)
```

RMSNorm processes every `(B,T)` token position independently:

```text
x squared:                         (B,T,C)
mean over the final dimension:     (B,T,1)
inverse RMS:                       (B,T,1)
learned weight gamma:                  (C,)
output:                            (B,T,C)
```

`keepdim=True` retains the final dimension as size one, allowing the inverse
RMS to broadcast across the token's $C$ features. The learned weight has shape
`(C,)` because the same feature scales are shared across every batch and token
position.

## Standalone RMSNorm Module

The module owns one learned parameter:

```python
self.weight = nn.Parameter(torch.ones(d_model))
```

The weight represents $\gamma$. Initializing it to ones means the module begins
as pure RMS normalization before training learns different feature scales.

The forward calculation is:

```python
mean_square = x.pow(2).mean(dim=-1, keepdim=True)
inverse_rms = torch.rsqrt(mean_square + eps)
normalized = x * inverse_rms
output = normalized * weight
```

Epsilon prevents division by zero. For an all-zero vector, the inverse RMS is
large but finite, and multiplying it by the zero input still produces a finite
zero output.

## Small Numerical Example

For:

$$
x=[3,4]
$$

the mean square and RMS are:

$$
\frac{3^2+4^2}{2}=12.5
$$

$$
\mathrm{RMS}(x)=\sqrt{12.5}
$$

With `eps=0` and $\gamma=[1,1]$:

$$
\mathrm{RMSNorm}(x) = \left[\frac{3}{\sqrt{12.5}}, \frac{4}{\sqrt{12.5}}\right]
$$

If $\gamma=[2,0.5]$, the first normalized feature is amplified while the
second is reduced.

## Configuration and Factory Integration

The modern configuration now accepts:

```text
normalization="layer_norm"
normalization="rms_norm"
```

The configuration validates the name, while the factory constructs the
selected module:

```text
ModernModelConfig
        ↓
build_normalization(kind, d_model)
        ↓
nn.LayerNorm or RMSNorm
```

Keeping selection logic in the factory prevents the Transformer block from
needing to understand each normalization implementation.

## Transformer Integration

Each modern Transformer block contains two independent normalization modules:

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

The complete language model also contains a final normalization before the
output head. A model with $L$ layers therefore has:

$$
2L+1
$$

normalization sites. Each site has its own learned weight.

## What the Tests Prove

The focused tests verify:

- agreement with a manually calculated RMSNorm result;
- shape preservation across several `(B,T,C)` inputs;
- finite zero output for an all-zero input;
- feature-wise behavior of the learned weight;
- finite gradients for both the input and learned weight;
- config validation and factory construction;
- independent RMSNorm instances inside each Transformer block;
- RMSNorm use in every block and the model's final normalization;
- finite logits, loss, and model-wide gradients in an end-to-end pass.

A shape test alone is insufficient because an incorrect formula can still
return a tensor with shape `(B,T,C)`. The manual numerical test proves the
actual calculation.

## Parameter-Count Comparison

For every normalization site:

```text
LayerNorm: gamma + beta = 2C parameters
RMSNorm:   gamma        =  C parameters
```

RMSNorm therefore removes $C$ parameters per site. Across the complete model:

$$
\Delta_{\mathrm{params}}=(2L+1)C
$$

For the comparison model with `n_layer=2` and `d_model=16`:

```text
normalization sites: 5
expected reduction:  5 * 16 = 80 parameters

LayerNorm model: 6080 parameters
RMSNorm model:   6000 parameters
observed reduction: 80 parameters
```

## Controlled Training Comparison

Both models used the same seed, model dimensions, token batch, targets,
optimizer, learning rate, weight decay, and number of steps. Only the
normalization method changed.

```text
seed:          0
batch shape:   (2,4)
optimizer:     AdamW
learning rate: 0.01
weight decay:  0.0
steps:         5
```

| Normalization | Parameters | Initial loss | Final loss |
|---|---:|---:|---:|
| LayerNorm | 6080 | 12.5702 | 3.4722 |
| RMSNorm | 6000 | 12.7885 | 3.5088 |

Both paths produced finite losses and substantially reduced loss on the
repeated batch. This demonstrates that both normalization choices support the
training path.

It does not establish which method is better. The comparison uses one seed,
one tiny repeated batch, five steps, and no held-out validation data.

## What I Learned

- normalization statistics are calculated separately for every token vector;
- the learned weight is shared by feature position across all batches and
  tokens;
- RMSNorm controls magnitude without explicitly centering the vector;
- epsilon is required for numerical safety around zero-magnitude inputs;
- a factory makes architectural substitutions possible without rewriting the
  Transformer block;
- component tests and end-to-end tests answer different questions;
- controlled experiments must keep every unrelated variable fixed;
- a successful smoke comparison proves execution, not generalization.

## Experiments To Try

- repeat the comparison with several random seeds;
- use separate training and validation batches;
- compare different model widths and depths;
- compare explicit epsilon values;
- record loss curves over more steps;
- measure runtime only after using a proper benchmark setup.

## Open Questions

- How sensitive is a small model to the epsilon value?
- Does either normalization converge more consistently across several seeds?
- Does the parameter reduction matter at this model size, or only at scale?
- How do mixed-precision calculations affect the RMS computation?
