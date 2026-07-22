# Phase 07: Trainable BPE Tokenizer

This phase replaced the fixed byte tokenizer with a tokenizer that learns new
tokens from repeated byte pairs. It also added Unicode-aware pre-tokenization,
tokenizer persistence, merge inspection, and compression reporting.

The complete flow is:

$$
\text{text}
\rightarrow \text{pre-tokenized chunks}
\rightarrow \text{UTF-8 byte IDs}
\rightarrow \text{learned BPE merges}
\rightarrow \text{flat token IDs}
$$

## Why BPE Is Needed

A byte tokenizer begins with exactly 256 tokens:

$$
0, 1, 2, \ldots, 255
$$

It can represent any UTF-8 text, but common words require many tokens. For
example, the ASCII text `abab` starts as:

$$
[97, 98, 97, 98]
$$

Byte Pair Encoding, or BPE, finds frequent adjacent pairs and assigns them new
token IDs. If the pair $(97, 98)$ is assigned token ID $256$, then:

$$
[97, 98, 97, 98]
\rightarrow
[256, 256]
$$

The tokenizer still supports every UTF-8 byte, while frequent byte sequences
can be represented more compactly.

## Counting Adjacent Pairs

For a token sequence:

$$
[1, 2, 1, 2, 3]
$$

the adjacent pairs are:

$$
(1,2), (2,1), (1,2), (2,3)
$$

Therefore, `count_pairs` returns:

```python
{
    (1, 2): 2,
    (2, 1): 1,
    (2, 3): 1,
}
```

For a sequence of length $T$, the number of adjacent positions is:

$$
T - 1
$$

An empty sequence or a one-token sequence has no pairs.

## Choosing and Applying a Merge

`find_best_pair` selects the pair with the largest count. In the previous
example, it selects:

$$
(1,2)
$$

If the new token ID is $99$, `merge_pair` performs:

$$
[1,2,1,2,3]
\rightarrow
[99,99,3]
$$

Merging is left-to-right and non-overlapping. When a pair matches, the two
input tokens are replaced by one output token and the index advances by two.
Otherwise, the current token is copied and the index advances by one.

For example, merging $(1,1)$ in:

$$
[1,1,1]
$$

produces:

$$
[99,1]
$$

The final `1` cannot be reused by the first merge because BPE replacements do
not overlap.

## Training Multiple Merges

One BPE training step performs:

$$
\text{count pairs}
\rightarrow \text{choose best pair}
\rightarrow \text{replace pair}
$$

`train_merges` repeats this process. Starting with:

$$
[1,2,1,2,3]
$$

the first rule can be:

$$
(1,2) \rightarrow 99
$$

which gives:

$$
[99,99,3]
$$

The pair counts must then be recalculated because the sequence has changed.
The second rule can be:

$$
(99,99) \rightarrow 100
$$

which gives:

$$
[100,3]
$$

The learned rules are stored in order:

```python
[
    ((1, 2), 99),
    ((99, 99), 100),
]
```

Order matters because a later rule may use a token created by an earlier rule.
Training stops early when no adjacent pairs remain.

## Tokenizer State

`TrainableBPETokenizer` stores three related pieces of state:

```python
self.merges
self.vocab
self.vocab_size
```

`self.merges` stores operations:

```python
((left_token_id, right_token_id), new_token_id)
```

`self.vocab` stores token meanings:

```python
token_id -> bytes
```

The initial vocabulary is:

```python
{
    0: bytes([0]),
    1: bytes([1]),
    ...
    255: bytes([255]),
}
```

When the tokenizer learns:

$$
(97,98) \rightarrow 256
$$

it builds:

```python
self.vocab[256] = self.vocab[97] + self.vocab[98]
```

Therefore:

```python
self.vocab[256] == b"ab"
```

## Train, Encode, and Decode

The three main methods have different responsibilities:

```text
train  = learn merge rules from a corpus
encode = convert text into token IDs using learned rules
decode = convert token IDs back into text
```

`train` changes the tokenizer state. `encode` and `decode` use the learned
state without learning new rules.

Encoding starts from known UTF-8 bytes, so it needs `self.merges` to answer:

```text
Which pair should be replaced by which new token ID?
```

Decoding starts from token IDs, so it needs `self.vocab` to answer:

```text
Which bytes does this token ID represent?
```

For example:

$$
\text{encode: } [97,98,97,98] \rightarrow [256,256]
$$

$$
\text{decode: } [256,256] \rightarrow b\text{"ab"}+b\text{"ab"}
\rightarrow \text{"abab"}
$$

## Pre-Tokenization

Applying BPE to one continuous byte sequence can create undesirable merges
across unrelated boundaries. This phase first splits text into chunks such as
letters, numbers, whitespace, and punctuation.

For example:

```python
split_text("hello 123!")
```

returns:

```python
["hello", " ", "123", "!"]
```

The educational Unicode-aware pattern is:

```python
r"(?:\p{L}\p{M}*)+|\p{N}+|\s+|[^\s\p{L}\p{N}]+"
```

Its branches mean:

```text
Unicode letters with optional marks
Unicode numbers
whitespace
punctuation or symbols
```

The `regex` package is used because it supports Unicode properties such as
`\p{L}`, `\p{M}`, and `\p{N}`.

Unicode marks are important for scripts such as Thai. Some vowels and tone
marks belong to the Unicode Mark category rather than the Letter category.
Attaching `\p{M}*` to each letter keeps those marks with the surrounding word.

This is a small educational pattern inspired by GPT-style pre-tokenization. It
is not an exact copy of the production GPT-4 tokenizer pattern.

## Byte Chunks and Pair Boundaries

`text_to_byte_chunks` encodes every pre-tokenized chunk separately:

```python
text_to_byte_chunks("hi 42!")
```

returns:

```python
[
    [104, 105],
    [32],
    [52, 50],
    [33],
]
```

`count_pairs_in_chunks` counts pairs inside each chunk and adds the counts
together. It never counts a pair between the end of one chunk and the start of
the next chunk.

For:

```python
[
    [1, 2],
    [3, 4],
]
```

the pairs are $(1,2)$ and $(3,4)$. The pair $(2,3)$ does not exist because it
would cross a chunk boundary.

The accumulation operation is:

```python
total_counts[pair] = total_counts.get(pair, 0) + count
```

`dict.get(pair, 0)` returns the existing total or starts from zero when the
pair has not been seen before.

## Chunk-Aware Merge Training

`merge_pair_in_chunks` applies one merge independently to every chunk while
preserving the nested shape:

$$
[[1,2,1,2],[1,2,3]]
\rightarrow
[[99,99],[99,3]]
$$

`train_merges_in_chunks` repeats the complete BPE process:

$$
\text{count pairs in all chunks}
\rightarrow \text{choose one global best pair}
\rightarrow \text{merge inside every chunk}
\rightarrow \text{repeat}
$$

After encoding, the chunks are flattened because the language model expects
one token sequence:

```python
token_ids = [
    token_id
    for chunk in token_chunks
    for token_id in chunk
]
```

The internal shape changes from:

$$
\text{list[list[int]]} \rightarrow \text{list[int]}
$$

The boundaries control which merges are allowed, but the boundaries do not
need to remain in the final model input.

## Saving and Loading

Training a tokenizer can be expensive, so its learned rules must be reusable.
The tokenizer is saved as JSON containing its vocabulary size and merge rules.

One merge is stored in JSON as:

```json
[97, 98, 256]
```

This represents:

$$
(97,98) \rightarrow 256
$$

JSON arrays load as Python lists, so `load` reconstructs the internal tuple
format:

```python
((97, 98), 256)
```

The full vocabulary does not need to be saved. `load` starts with the 256 byte
tokens and rebuilds every merged token in learned order. This works because
later tokens can only depend on base tokens or earlier merged tokens.

The persistence flow is:

$$
\text{train}
\rightarrow \text{save merge rules}
\rightarrow \text{load merge rules}
\rightarrow \text{rebuild vocabulary}
$$

## Merge Inspection

Numeric merge rules are difficult to understand directly. `inspect_merges`
reports the pair, new token ID, represented bytes, and readable text.

For example:

```python
{
    "pair": [97, 98],
    "new_token_id": 256,
    "token_bytes": [97, 98],
    "token_text": "ab",
}
```

Some intermediate tokens contain only part of a multi-byte UTF-8 character.
Decoding such a token by itself displays the replacement character `�`. This
does not mean the corpus is corrupted. A later merge can combine the remaining
bytes and form a complete character such as `“` or `”`.

## Compression Statistics

The tokenizer compares the original UTF-8 byte count with the BPE token count:

$$
\text{compression ratio}
=
\frac{\text{UTF-8 byte count}}{\text{BPE token count}}
$$

For `abab`:

$$
\frac{4}{2}=2.0
$$

This means each BPE token represents two source bytes on average. A larger
ratio means the tokenizer produced a shorter sequence. Empty text reports a
ratio of $0.0$ to avoid division by zero.

Compression ratio measures sequence reduction, not language-model quality. A
larger vocabulary can shorten sequences but also increases the model's input
embedding and output projection sizes.

## Training on a Real Corpus

The Phase 7 script trains on the body of Project Gutenberg's *Alice's
Adventures in Wonderland*:

```text
data/processed/alice_in_wonderland_body.txt
```

The script is:

```text
scripts/train_bpe.py
```

It trains a target vocabulary of 512 tokens:

$$
512 - 256 = 256 \text{ learned merge tokens}
$$

The measured result was:

```text
Vocabulary size: 512
Learned merges: 256
UTF-8 byte count: 151096
BPE token count: 84335
Compression ratio: 1.79
```

The first learned pieces included:

```text
he, the, in, ou, an, it, er, at, on, ing, re, to, th, and
```

The saved tokenizer is:

```text
outputs/tokenizers/phase_07_tokenizer.json
```

Reloading it reproduced all 256 merges, the same token count, and a successful
text round trip:

$$
\text{decode}(\text{encode}(\text{text})) = \text{text}
$$

## Educational Limitations

This implementation favors clarity over training speed. For every merge, it
recounts pairs across the corpus and rebuilds affected chunk lists. Production
tokenizer trainers use more efficient data structures and update only counts
affected by a merge.

The split pattern is intentionally smaller than production GPT tokenizer
patterns. It demonstrates why pre-tokenization exists without copying all
special cases for contractions, spaces, line breaks, and special tokens.

Despite these limitations, this phase contains the complete tokenizer
lifecycle:

$$
\text{train}
\rightarrow \text{inspect}
\rightarrow \text{measure}
\rightarrow \text{save}
\rightarrow \text{load}
\rightarrow \text{encode/decode}
$$
