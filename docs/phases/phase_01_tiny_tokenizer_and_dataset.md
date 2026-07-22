# Phase 01: Tiny Tokenizer And Dataset

This phase introduced the first data path: raw text becomes token IDs, and
token IDs become next-token prediction examples.

The two core pieces are:

```text
ByteTokenizer
make_next_token_example
```

## Byte Tokenization

A tokenizer converts text into numbers. Models do not read Python strings
directly; they read integer token IDs.

This phase used a byte tokenizer. A byte tokenizer represents text as UTF-8
bytes. Each byte is already an integer from `0` to `255`, so the vocabulary size
is `256`.

Example:

$$
\text{"hi"} \rightarrow [104, 105]
$$

$$
[104, 105] \rightarrow \text{"hi"}
$$

`ByteTokenizer.encode()` does:

$$
\text{string} \rightarrow \text{UTF-8 bytes} \rightarrow \text{list of integers}
$$

`ByteTokenizer.decode()` does:

$$
\text{list of integers} \rightarrow \text{bytes} \rightarrow \text{UTF-8 string}
$$

Byte tokenization is simple and educational. It avoids the complexity of BPE or
wordpiece tokenizers while still letting the model operate on real text.

## Next-Token Prediction Examples

Language models learn by predicting the next token.

Given tokens:

$$
\text{tokens} = [1, 2, 3, 4]
$$

and `block_size = 3`, the input and target are:

$$
x = [1, 2, 3]
$$

$$
y = [2, 3, 4]
$$

The target is shifted one position to the right. At each position, the model
sees the current and previous tokens and learns to predict the next token.

Position by position:

$$
1 \rightarrow 2
$$

$$
2 \rightarrow 3
$$

$$
3 \rightarrow 4
$$

The helper:

```python
make_next_token_example(token_ids, block_size)
```

returns exactly that shifted pair.

## Why Block Size Matters

`block_size` controls how many tokens are included in one training example.

Example:

$$
\text{tokens} = [10, 20, 30, 40, 50]
$$

With `block_size = 2`:

$$
x = [10, 20]
$$

$$
y = [20, 30]
$$

With `block_size = 4`:

$$
x = [10, 20, 30, 40]
$$

$$
y = [20, 30, 40, 50]
$$

Later, `block_size` becomes the context length: how many previous tokens the
model can use when predicting the next token.

## Small Example

Text:

```python
"hello"
```

Byte IDs:

$$
[104, 101, 108, 108, 111]
$$

With `block_size = 4`:

$$
x = [104, 101, 108, 108]
$$

$$
y = [101, 108, 108, 111]
$$

This tiny example contains four next-token prediction tasks.
