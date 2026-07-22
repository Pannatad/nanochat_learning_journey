# Phase 08: Tokenized Data Pipeline

This phase connected the trained BPE tokenizer to model-ready batches. It added
prepared token shards, deterministic loading, resumable loader state, dataset
identity metadata, and a reusable preparation script.

The complete flow is:

$$
\text{corpus}
\rightarrow \text{BPE token IDs}
\rightarrow \text{token shards}
\rightarrow \text{shifted batches } (x,y)
$$

## Splitting Tokens into Shards

A long token sequence is divided into smaller ordered pieces. For example,
with a shard size of two:

```text
[10, 20, 30, 40, 50]
        ↓
[
    [10, 20],
    [30, 40],
    [50],
]
```

The final shard may be smaller than the configured shard size. Splitting must
preserve every token's order without duplicating or losing tokens.

Each shard is stored as a one-dimensional `torch.long` tensor:

```text
shard_000.pt → tensor([10, 20])
shard_001.pt → tensor([30, 40])
shard_002.pt → tensor([50])
```

Zero-padded names keep lexical and numerical order aligned.

## Constructing Next-Token Batches

For batch size $B$ and block size $T$, the loader needs:

$$
B \times T + 1
$$

tokens. The extra token supplies the final prediction target. With $B=1$,
$T=3$, and this buffer:

```text
[10, 20, 30, 40]
```

the batch is:

```text
x = [10, 20, 30]
y = [20, 30, 40]
```

Both tensors are reshaped to `(B, T)`. The targets are shifted by one token,
while the loader itself advances by $B \times T$ tokens.

## Why Batches Overlap

After the first batch, the loader advances by three positions rather than
four:

```text
first buffer:  [10, 20, 30, 40]
second buffer:             [40, 50, 60, 70]
```

The shared token `40` was the final target in the first batch and becomes the
first input in the second batch. Advancing by $B \times T + 1$ would skip the
training pair `40 → 50`.

## Shard Boundaries

If the current shard cannot provide a complete $B \times T + 1$ buffer, the
loader advances to the next shard and resets its position to zero. The current
educational loader does not join an incomplete tail to the beginning of the
next shard.

This makes the loading and resume state simple, but a few tokens near a shard
boundary may not be used. Each loaded shard must also contain enough tokens for
at least one complete batch.

After the last shard, modulo arithmetic returns the loader to shard zero. A
fixed shard order and starting state therefore produce a deterministic cycle.

## Loader Resume State

The loader state contains:

```python
{
    "shard_index": 1,
    "position": 300,
}
```

Restoring the state sets both values and reloads the corresponding shard. It
does not call `next_batch`, so restoring state does not consume any tokens. The
next batch after restoration matches the next batch from uninterrupted
loading.

## Dataset Identity Metadata

The prepared dataset records SHA-256 fingerprints for both the source corpus
and saved tokenizer. A path alone is insufficient because a file's contents
can change without its name changing.

The metadata also records:

```text
format version
total token count
configured shard size
shard filenames
per-shard token counts
```

Building metadata and saving JSON are separate operations. This keeps dataset
description independent from persistence and makes both behaviors easy to
test.

## Reusable Preparation Pipeline

The preparation script performs the complete real-data workflow:

```text
load Alice corpus
→ load Phase 7 BPE tokenizer
→ encode text
→ save token shards
→ build metadata
→ save metadata.json
```

The Phase 8 Alice dataset contains:

```text
token count:       84,335
configured size:   10,000 tokens per shard
shard count:       9
final shard size:  4,335 tokens
```

The sum of the metadata shard counts and the actual saved tensor lengths both
equal 84,335.

## Reproducibility

Two loaders with the same shard paths, batch size, block size, and starting
state produce identical sequences of `(x, y)` batches. Tests cover advancement
inside one shard, movement across a shard boundary, and restoration from saved
state.

## Educational Limitations

The loader favors clarity over maximum data usage and throughput. It loads one
whole shard tensor at a time and leaves an incomplete tail behind rather than
stitching two shards together. Production pipelines may use memory mapping,
asynchronous prefetching, distributed shard assignment, and cross-shard
buffers.
