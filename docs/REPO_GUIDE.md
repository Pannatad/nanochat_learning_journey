# Repository Guide

Use this page when you want to run something. All commands assume the current
directory is the repository root:

```text
/Users/pannatad/Desktop/AI/LLM From scratch
```

## 1. Pick a path

There are two model paths. Start with the tiny path if you are learning the
fundamentals. Start with the modern path if you want a configurable model,
real tokenized data, experiments, or reloadable weights.

| Path | Model | Best for | Main entry point |
|---|---|---|---|
| Tiny | `TinyLanguageModel` | first forward pass and training loop | `scripts/train_tiny_configured.py` |
| Modern | `ModernLanguageModel` | architecture studies and checkpoints | `scripts/run_architecture_study.py` |

For the model’s internals, go to [Model architecture](MODEL_ARCHITECTURE.md).
For files produced by each path, go to [Model artifacts](MODEL_ARTIFACTS.md).

## 2. Install and verify

The project requires Python 3.11 or newer.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Verify the installation:

```bash
.venv/bin/python -c "import llm_lab; print(llm_lab.__version__)"
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

## 3. Learn with the tiny model

Run the configured ten-step training example:

```bash
.venv/bin/python scripts/train_tiny_configured.py
```

It uses a byte tokenizer, a fixed training/validation example, AdamW, and
greedy generation. The output is saved under:

```text
outputs/runs/phase_06_smoke_<timestamp>/
├── resolved_config.yaml
├── metadata.json
├── train_log.jsonl
└── generated.txt
```

This early script does not save model weights. To see the even smaller,
hardcoded version:

```bash
.venv/bin/python scripts/train_tiny_smoke.py
```

## 4. Prepare the modern model’s inputs

The modern study uses the processed Alice corpus, a 512-token BPE tokenizer,
and token shards. These artifacts already exist in this checkout. Rebuild them
only when you intentionally want to regenerate the inputs:

```bash
.venv/bin/python scripts/train_bpe.py
.venv/bin/python scripts/prepare_token_data.py
```

The important paths are:

```text
data/processed/alice_in_wonderland_body.txt
outputs/tokenizers/phase_07_tokenizer.json
outputs/datasets/phase_08_alice/metadata.json
outputs/datasets/phase_08_alice/shard_000.pt ... shard_008.pt
```

## 5. Train and compare the modern model

### First: CPU smoke study

This is the recommended first run. It tests all four combinations of
LayerNorm/RMSNorm and learned positions/RoPE with one seed and twenty steps:

```bash
.venv/bin/python scripts/run_architecture_study.py \
  configs/phase_11_architecture_smoke.yaml
```

The study creates four checkpoints under:

```text
outputs/runs/phase_11_architecture_smoke_<timestamp>/runs/<variant>/seed_0/
```

### Then: canonical five-seed study

This runs four variants, five seeds, 1,000 steps, and 512,000 training tokens
per run:

```bash
.venv/bin/python scripts/run_architecture_study.py \
  configs/phase_11_architecture_study.yaml
```

The canonical config requests Apple MPS. It fails clearly when MPS is not
available instead of silently changing the experimental conditions. The
portable CPU check is the smoke config above.

The full report is written to:

```text
outputs/reports/phase_11_architecture/
```

To compare two YAML recipes without training:

```bash
.venv/bin/python scripts/compare_configs.py \
  configs/phase_06_smoke.yaml \
  outputs/runs/phase_06_smoke_<timestamp>/resolved_config.yaml
```

To compare normalizations on a small fixed batch:

```bash
.venv/bin/python scripts/compare_normalizations.py
```

## 6. Find, inspect, and use weights

### Find a checkpoint

Modern weights follow this pattern:

```text
outputs/runs/phase_11_architecture_<timestamp>/runs/<variant>/seed_<seed>/checkpoint.pt
```

List them:

```bash
find outputs/runs/phase_11_architecture_* -type f -name checkpoint.pt -print | sort
```

### Inspect a checkpoint

Replace the placeholder with one path from the list:

```bash
.venv/bin/python scripts/inspect_checkpoint.py \
  outputs/runs/phase_11_architecture_<timestamp>/runs/<variant>/seed_<seed>/checkpoint.pt
```

This prints the variant, seed, model configuration, unique parameter counts,
and final metrics.

### Generate from a checkpoint

```bash
.venv/bin/python scripts/generate_from_checkpoint.py \
  outputs/runs/phase_11_architecture_<timestamp>/runs/<variant>/seed_<seed>/checkpoint.pt \
  --prompt 'Alice was beginning' \
  --max-new-tokens 64
```

The helper loads `outputs/tokenizers/phase_07_tokenizer.json` by default and
uses greedy next-token generation. The prompt plus new tokens must fit inside
the model’s `block_size` of 128.

For the newest local checkpoint by path order:

```bash
CHECKPOINT="$(find outputs/runs -type f -name checkpoint.pt | sort | tail -n 1)"
.venv/bin/python scripts/inspect_checkpoint.py "$CHECKPOINT"
.venv/bin/python scripts/generate_from_checkpoint.py "$CHECKPOINT"
```

## 7. Follow the code

```text
src/llm_lab/
├── model/        model definitions and architecture choices
├── tokenizer/    byte and trainable BPE tokenizers
├── data/         tiny examples and token-shard loading
├── training/     loss, optimizer, training, validation, generation
└── experiments/  YAML-driven architecture studies and reports

configs/          reproducible experiment recipes
scripts/          commands you run directly
docs/phases/      concept-by-concept learning notes
outputs/          generated data, logs, reports, and local checkpoints
tests/            numerical and behavior checks
```

The modern model’s source flow is mapped in
[Model architecture](MODEL_ARCHITECTURE.md). The exact checkpoint schema and
tracked/generated boundary are in [Model artifacts](MODEL_ARTIFACTS.md).

## 8. Common pitfalls

- Run from the repository root; the study configs use relative paths.
- A YAML config describes an experiment. `ModernModelConfig` describes the
  model constructor. They are related but not the same object.
- The tokenizer vocabulary and `model.vocab_size` must match. The current
  tokenizer has 512 tokens.
- The raw Phase 11 run folders are ignored by Git. A clean clone will not have
  checkpoints until you run a study.
- The current model is an educational float32 implementation. It does not yet
  have KV-cache inference, context-window cropping, or a chat interface.
