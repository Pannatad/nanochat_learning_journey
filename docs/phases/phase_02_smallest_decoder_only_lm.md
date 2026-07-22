# Phase 02: Tiny Decoder-Only Model

This phase created the first language model skeleton. It does not have
attention yet. It only learns the basic path from token IDs to logits.

The model flow is:

$$
\text{token IDs}
\rightarrow \text{token embeddings}
\rightarrow \text{output head}
\rightarrow \text{logits}
$$

## Token IDs

The input to the model is a tensor of token IDs shaped:

$$
(B, T)
$$

`B` is batch size. `T` is sequence length.

Example:

```python
x = [
    [1, 2, 3],
    [4, 5, 6],
]
```

Shape:

$$
(B, T) = (2, 3)
$$

Each number is a token ID from the tokenizer.

## Token Embeddings

`nn.Embedding(vocab_size, d_model)` creates a learned table with one vector per
token ID.

For the byte tokenizer:

Here, $V$ is the vocabulary size.

$$
V = 256
$$

If:

$$
d_{\text{model}} = 16
$$

then each token ID becomes a vector of length `16`.

Shape change:

$$
(B, T) \rightarrow (B, T, d_{\text{model}})
$$

Example:

$$
(2, 3) \rightarrow (2, 3, 16)
$$

## Output Head

`nn.Linear(d_model, vocab_size)` projects each token vector back to vocabulary
scores.

Shape change:

$$
(B, T, d_{\text{model}})
\rightarrow
(B, T, V)
$$

The output is called logits. Logits are raw scores, one score for every
possible next token.

For byte-level language modeling:

$$
\text{logits shape} = (B, T, 256)
$$

## Cross-Entropy Loss

The model returns logits. The loss is computed outside the model.

Cross entropy compares:

$$
\text{one vector of vocabulary scores}
\rightarrow
\text{one correct token ID}
$$

The logits start as:

$$
(B, T, V)
$$

Targets start as:

$$
(B, T)
$$

For cross entropy, batch and time are flattened together:

$$
\text{logits}: (B, T, C) \rightarrow (B \cdot T, C)
$$

$$
\text{targets}: (B, T) \rightarrow (B \cdot T)
$$

This turns every token position into one classification example.

## Small Example

If:

$$
B = 2,\quad T = 3,\quad V = 256
$$

then the model produces:

$$
\text{logits shape} = (2, 3, 256)
$$

There are:

$$
2 \cdot 3 = 6
$$

next-token predictions in that batch.

Each row after flattening asks:

```text
which of the 256 tokens should come next?
```
