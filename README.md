# LLM From Scratch Lab

A small, readable PyTorch repository for rebuilding a decoder-only language
model one concept at a time.

## Choose what you want to do

| I want to... | Use this first |
|---|---|
| Learn the smallest model and training loop | `scripts/train_tiny_configured.py` |
| Run the current Transformer smoke test | `scripts/run_architecture_study.py` + smoke config |
| Run the full architecture comparison | `scripts/run_architecture_study.py` + study config |
| Load existing weights | `scripts/inspect_checkpoint.py` |
| Understand the model | [Model architecture](docs/MODEL_ARCHITECTURE.md) |
| Find files and outputs | [Model artifacts](docs/MODEL_ARTIFACTS.md) |

## Read in this order

1. [Repository guide](docs/REPO_GUIDE.md) — setup and copy-paste commands.
2. [Model architecture](docs/MODEL_ARCHITECTURE.md) — flow chart and tensor shapes.
3. [Model artifacts](docs/MODEL_ARTIFACTS.md) — tokenizer, datasets, reports, and weights.
4. [Phase index](docs/phases/README.md) — concepts and learning notes.

## Three useful commands

From the repository root, after installing the project:

```bash
# Smallest end-to-end training path
.venv/bin/python scripts/train_tiny_configured.py

# Current modern-model CPU smoke study
.venv/bin/python scripts/run_architecture_study.py \
  configs/phase_11_architecture_smoke.yaml

# Inspect the newest local modern checkpoint
CHECKPOINT="$(find outputs/runs -type f -name checkpoint.pt | sort | tail -n 1)"
.venv/bin/python scripts/inspect_checkpoint.py "$CHECKPOINT"
```

## Current model in one screen

```text
token IDs
→ token embeddings
→ learned positions or RoPE
→ pre-norm causal multi-head attention
→ ReLU feed-forward network
→ residual Transformer blocks
→ final normalization
→ tied vocabulary projection
→ next-token logits
```

`TinyLanguageModel` is the early educational model. `ModernLanguageModel` is
the current configurable model used by the Phase 9–11 work. QK norm, GQA,
optimized attention, KV-cache inference, chat, SFT, and RL are not implemented
yet.

There is no single checked-in pretrained model. Modern checkpoints are generated
locally under `outputs/runs/phase_11_architecture_<timestamp>/` and ignored by
Git. The [repository guide](docs/REPO_GUIDE.md) explains how to create or load
them.
