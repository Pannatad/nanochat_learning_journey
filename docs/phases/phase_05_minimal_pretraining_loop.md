# Phase 05: Minimal Pretraining Loop

This phase turned the model from "can produce logits" into "can run a tiny
training loop." It introduced loss computation, AdamW, one training step,
validation, greedy generation, JSONL logging, and a CPU smoke script.

The training flow is:

$$
(x, y)
\rightarrow \text{model}(x)
\rightarrow \text{logits}
\rightarrow \text{cross-entropy loss}
\rightarrow \text{backward}
\rightarrow \text{optimizer step}
\rightarrow \text{log loss}
$$

## Loss Computation

The model returns logits shaped:

$$
(B, T, \text{vocab\_size})
$$

Targets are token IDs shaped:

$$
(B, T)
$$

Cross entropy expects one row per prediction, so batch and time are flattened:

$$
\text{logits}: (B, T, \text{vocab\_size})
\rightarrow
(B \cdot T, \text{vocab\_size})
$$

$$
\text{targets}: (B, T) \rightarrow (B \cdot T)
$$

Example:

$$
B = 2,\quad T = 3,\quad \text{vocab\_size} = 256
$$

The batch contains:

$$
2 \cdot 3 = 6
$$

next-token prediction problems.

## AdamW Optimizer

AdamW updates model parameters after gradients are computed.

The optimizer is built from:

```python
model.parameters()
```

The first exposed settings are:

```text
learning rate
weight decay
```

Adam keeps moving averages of gradients. The first moving average tracks the
recent direction of gradients:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

This behaves like momentum. If recent gradients point in a similar direction,
the update moves more smoothly.

The second moving average tracks squared gradients:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

This gives adaptive scaling. Parameters with large recent gradients get smaller
effective updates; parameters with smaller recent gradients can get relatively
larger updates.

At the beginning, `m_t` and `v_t` are biased toward zero, so Adam applies bias
correction:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

The rough Adam update is:

$$
\begin{aligned}
\theta_t &= \theta_{t-1}
- \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

`eps` is a tiny value that prevents division by zero.

AdamW separates weight decay from the adaptive gradient update:

$$
\begin{aligned}
\text{Adam update} &:\ \text{use gradients, momentum, and scaling} \\
\text{weight decay step} &:\ \text{separately shrink weights a little}
\end{aligned}
$$

This avoids distorting weight decay through Adam's adaptive scaling.

## Training Step

One training step does:

```python
model.train()
optimizer.zero_grad()
logits = model(x)
loss = compute_loss(logits, y)
loss.backward()
optimizer.step()
```

`optimizer.zero_grad()` is important because PyTorch accumulates gradients by
default. Without clearing old gradients, the next update would mix gradients
from previous batches with the current batch.

`loss.backward()` computes gradients. `optimizer.step()` updates the model
weights in place.

## Validation Step

Validation measures loss without training.

It uses:

```python
model.eval()
with torch.no_grad():
    ...
```

`model.eval()` switches the model into evaluation mode. `torch.no_grad()` tells
PyTorch not to track gradients. This saves memory and makes it clear that
validation is measurement only.

Validation returns a loss number, but it does not update parameters.

## Greedy Generation

Generation starts with prompt token IDs shaped:

$$
(B, T)
$$

Each loop:

$$
\text{tokens}
\rightarrow \text{model}
\rightarrow \text{last-position logits}
\rightarrow \arg\max
\rightarrow \text{append next token}
$$

The important line is:

```python
last_logits = logits[:, -1, :]
```

Shape:

$$
(B, \text{vocab\_size})
$$

Greedy decoding uses:

```python
next_token = torch.argmax(last_logits, dim=-1)
```

Then `unsqueeze(-1)` changes the shape from:

$$
(B) \rightarrow (B, 1)
$$

so the token can be appended to the sequence.

Because this model is barely trained, generated text can look strange or
repetitive. Phase 5 proves the mechanics, not model quality.

## JSONL Logging

The smoke script logs one JSON object per line:

```json
{"step": 0, "train_loss": 5.12}
{"step": 1, "train_loss": 4.98}
{"step": 10, "val_loss": 5.01}
```

JSONL is useful because each line is independent. Later, logs can be streamed,
read line by line, or plotted.

## Smoke Script

`scripts/train_tiny_smoke.py` runs a tiny CPU-only training smoke test. It uses
one fixed small batch for 10 steps, logs loss, runs validation, generates text,
and writes:

```text
outputs/phase_05/train_log.jsonl
outputs/phase_05/generated.txt
```

The fixed batch is not a real data pipeline. It is only a small proof that the
training loop, validation, logging, and generation all connect correctly.

## Small Example

For tokens:

$$
[10, 20, 30, 40, 50]
$$

with `block_size = 4`:

$$
x = [10, 20, 30, 40]
$$

$$
y = [20, 30, 40, 50]
$$

The model predicts all four target tokens in one forward pass. The loss is the
average over those token positions.
