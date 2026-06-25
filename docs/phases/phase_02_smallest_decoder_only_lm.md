# Phase 02: Tiny Decoder-Only Model

## What We Built

We built the smallest decoder-only language model skeleton. The model takes
token IDs, looks up token embeddings, projects those embeddings back to
vocabulary-sized logits, and lets the loss be computed outside the model.

## What I Learned

- I learned that `nn.Embedding(vocab_size, d_model)` is used for token
  embeddings.
- I learned that token IDs shaped `(B, T)` become embeddings shaped
  `(B, T, d_model)`.
- I learned that `nn.Linear(input, output)` projects vectors from one dimension
  to another.
- I learned that the output logits should have shape `(B, T, vocab_size)`.
- I learned that cross-entropy uses logits and targets, with logits flattened
  to `(B * T, C)` and targets flattened to `(B * T)`.

## High-Level Understanding

Phase 2 creates the first tiny language model. It does not have attention yet.
It only learns this basic path:

```text
token IDs -> token embedding -> linear output head -> logits -> cross entropy
```

The model returns logits. The training objective stays outside the model.

## Intuition / Small Example

If the input tokens are:

```python
x = [
    [1, 2, 3],
    [4, 5, 6],
]
```

then the input shape is:

```text
(B, T) = (2, 3)
```

After token embedding:

```text
(B, T, d_model)
```

After the output head:

```text
(B, T, vocab_size)
```

For cross-entropy, the logits and targets are flattened:

```text
logits:  (B, T, C) -> (B * T, C)
targets: (B, T)    -> (B * T)
```

This means each token position becomes one classification example.

## Detailed Explanation

`nn.Embedding(vocab_size, d_model)` creates a table with one learned vector per
token ID. For a byte tokenizer, `vocab_size` is `256`. If `d_model` is `16`,
each token becomes a vector of length `16`.

`nn.Linear(d_model, vocab_size)` converts each embedding vector into logits.
The logits are raw scores, one score for each possible next token.

`F.cross_entropy(logits, targets)` internally applies the softmax idea and
penalizes the model when the correct token has low probability. The logits are
flattened to `(B * T, C)` because cross-entropy expects one row per prediction.
The targets are flattened to `(B * T)` because each prediction has one correct
class ID.

## Experiments To Try

- Change `d_model` and check that the logits shape still ends with
  `vocab_size`.
- Change `vocab_size` and check that the output head changes the last
  dimension.
- Print the loss before any training and confirm it is finite.

## Tests / Checks

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

Expected result:

- model creation test passes;
- logits shape test passes;
- finite cross-entropy loss test passes;
- Ruff reports all checks passed.
