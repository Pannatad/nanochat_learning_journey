# Phase 01: Tiny Tokenizer and Dataset

## What We Built

We added a small byte tokenizer and a tiny next-token dataset helper. The
tokenizer can encode text into byte token IDs and decode those IDs back into
text. The dataset helper creates one shifted input/target example for
next-token prediction.

## What I Learned

- I learned how to write a test and run it to validate a function I wrote.
- I learned how a byte tokenizer works.
- I learned how to write `encode()` and `decode()` using UTF-8.
- I learned how to write a function that returns the next set of tokens based
  on `block_size` and the previous tokens.

## High-Level Understanding

Phase 1 turns raw text into token IDs, then turns those token IDs into a simple
training example. The tokenizer handles text conversion. The dataset helper
handles the next-token prediction shift.

## Intuition / Small Example

Byte tokenization converts text into numbers:

```python
"hi" -> [104, 105]
[104, 105] -> "hi"
```

Next-token prediction shifts the target by one:

```python
tokens = [1, 2, 3, 4]

x = [1, 2, 3]
y = [2, 3, 4]
```

The model reads `x` and learns to predict `y`. At position 0, it sees `1` and
should predict `2`. At position 1, it sees `2` and should predict `3`.

## Detailed Explanation

`ByteTokenizer.encode()` uses UTF-8 to turn a Python string into bytes, then
converts those bytes into a list of integers.

`ByteTokenizer.decode()` takes a list of byte IDs, converts it back into
`bytes`, and decodes it with UTF-8 to recover the original string.

`make_next_token_example()` takes token IDs and a `block_size`. It returns:

- `x`: the first `block_size` tokens;
- `y`: the next `block_size` tokens, shifted one position to the right.

This shift is the core idea behind next-token prediction.

## Experiments To Try

- Try encoding and decoding text with spaces, punctuation, or emojis.
- Change `block_size` and check how `x` and `y` change.
- Try passing too few tokens and decide what error behavior we should add
  later.

## Tests / Checks

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

Expected result:

- byte tokenizer round trip test passes;
- next-token dataset helper test passes;
- Ruff reports all checks passed.
