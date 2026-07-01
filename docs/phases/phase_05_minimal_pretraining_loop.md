# Phase 05: Minimal Pretraining Loop

## What We Built

We built the first minimal pretraining loop. It can compute language-model
loss, create an AdamW optimizer, run one training step, run validation without
updating weights, generate token IDs greedily, and run a 10-step CPU smoke
training script.

The smoke script writes:

```text
outputs/phase_05/train_log.jsonl
outputs/phase_05/generated.txt
```

## What I Learned

- I learned that `x` and `y` are both token ID tensors with shape `(B, T)`.
- I learned that model logits have shape `(B, T, vocab_size)`.
- I learned that cross entropy needs logits flattened to `(B * T, vocab_size)`
  and targets flattened to `(B * T)`.
- I learned that `optimizer.zero_grad()` clears old gradients before
  `loss.backward()`.
- I learned that validation uses `model.eval()` and `torch.no_grad()` because
  it measures loss without training.
- I learned that generation repeatedly uses the last token's logits to choose
  the next token.

## High-Level Understanding

Pretraining teaches the model to predict the next token. For each input token
sequence, the target is the same sequence shifted one position to the right.

The training loop is:

```text
x, y batch
-> model(x)
-> logits
-> cross-entropy loss
-> backward
-> optimizer step
-> log loss
```

Validation uses the same forward and loss calculation, but skips backward and
optimizer updates.

## Intuition / Small Example

For tokens:

```text
[10, 20, 30, 40, 50]
```

with `block_size = 4`, one next-token example is:

```text
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

The model predicts a full vocabulary score vector at every position:

```text
logits shape: (B, T, vocab_size)
target shape: (B, T)
```

Then loss flattens batch and time:

```text
logits:  (B, T, vocab_size) -> (B * T, vocab_size)
targets: (B, T)             -> (B * T)
```

Generation starts from a prompt and appends one token at a time:

```text
start tokens
-> model
-> logits[:, -1, :]
-> argmax
-> append next token
-> repeat
```

## Detailed Explanation

### Loss

`compute_loss` receives logits and targets. The logits contain one vocabulary
prediction for every token position. The targets contain the correct next-token
IDs.

Cross entropy compares:

```text
one prediction vector -> one correct token ID
```

So `(B, T)` token positions become `B * T` separate prediction problems.

### Optimizer

`make_optimizer` creates AdamW over `model.parameters()`. AdamW is the optimizer
that decides how to change each parameter after `loss.backward()` computes
gradients.

For this phase, we only expose:

```text
learning rate
weight decay
```

AdamW's default `betas` and `eps` are good enough for this first loop:

```text
betas = (0.9, 0.999)
eps = 1e-8
```

The high-level Adam idea is:

```text
gradient -> momentum estimate -> scale estimate -> parameter update
```

The first moving average tracks the direction of recent gradients:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

This is like momentum. If several recent gradients point in a similar
direction, Adam keeps moving in that direction more smoothly.

The second moving average tracks the size of recent squared gradients:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

This is used for adaptive scaling. Parameters with consistently large
gradients get smaller effective updates, while parameters with smaller
gradients can get relatively larger updates.

Early in training, both `m_t` and `v_t` start near zero, so Adam applies bias
correction before using them:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Then the Adam-style update is roughly:

$$
\theta_t = \theta_{t-1}
- \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

`eps` is just a tiny number that prevents division by zero.

AdamW differs from older Adam-with-weight-decay by decoupling weight decay from
the adaptive gradient update. In plain terms:

```text
Adam update:        use gradients, momentum, and scaling
weight decay step:  separately shrink weights a little
```

This matters because coupling weight decay into the gradient can distort the
regularization through Adam's adaptive scaling. AdamW keeps the optimization
update and the weight-shrinking regularization as separate ideas.

### Training Step

One training step does:

```text
model.train()
optimizer.zero_grad()
logits = model(x)
loss = compute_loss(logits, y)
loss.backward()
optimizer.step()
```

The important detail is that gradients from old steps must be cleared before
the new backward pass.

### Validation Step

Validation does:

```text
model.eval()
with torch.no_grad():
    logits = model(x)
    loss = compute_loss(logits, y)
```

This keeps validation as measurement only. No gradients are stored, and no
parameters are updated.

### Generation

The first generation helper uses greedy decoding:

```text
next token = argmax(last-position logits)
```

Because the model is barely trained, generated text can still look strange.
That is expected. Phase 5 proves the mechanics of training and generation; it
does not try to produce good language yet.

### Smoke Script

`scripts/train_tiny_smoke.py` trains on one fixed tiny batch for 10 steps. This
is intentionally small. It proves the loop works, logs loss, runs validation,
and saves generated text.

A real dataset pipeline would continuously sample many shifted chunks from a
large token stream:

```text
x = tokens[start : start + block_size]
y = tokens[start + 1 : start + block_size + 1]
```

That belongs in a later data-pipeline phase.

## Experiments To Try

- Change `block_size` and make sure the text has at least `block_size + 1`
  tokens.
- Change the learning rate and compare the 10-step loss curve.
- Change the prompt in `scripts/train_tiny_smoke.py`.
- Replace greedy `argmax` generation with sampling later.
- Try shifting the batch start position each step instead of reusing one fixed
  batch.

## Tests / Checks

```bash
.venv/bin/python scripts/train_tiny_smoke.py
.venv/bin/pytest
.venv/bin/ruff check .
```

Expected result:

- the smoke script completes on CPU;
- `outputs/phase_05/train_log.jsonl` is written;
- `outputs/phase_05/generated.txt` is written;
- training-loop tests pass;
- all earlier phase tests still pass;
- Ruff reports all checks passed.
