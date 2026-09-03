# Model Artifacts and Weights

This is the source of truth for where the current model inputs, reports, and
weights live. Paths are relative to the repository root.

## Quick answer

The reloadable model weights are here:

```text
outputs/runs/phase_11_architecture_<timestamp>/runs/<variant>/seed_<seed>/checkpoint.pt
```

There is no single checked-in pretrained model. The Phase 11 checkpoints are
generated locally and ignored by Git. If the folder does not exist, run the CPU
smoke study from [the repository guide](REPO_GUIDE.md).

## 1. Data versus weights

There are two different kinds of `.pt` files in this project:

| Path | What it is | Can it rebuild a model? |
|---|---|---|
| `outputs/datasets/phase_08_alice/shard_*.pt` | token ID tensors for training/validation | No; these are data |
| `outputs/runs/phase_11_architecture_<timestamp>/runs/<variant>/seed_<seed>/checkpoint.pt` | modern Transformer checkpoint | Yes |

The current checkout contains timestamped generated architecture runs under
`outputs/runs`; examples while this guide is being written are:

```text
outputs/runs/phase_11_architecture_smoke_20260902_205950/
outputs/runs/phase_11_architecture_smoke_20260902_211739/
outputs/runs/phase_11_architecture_study_20260902_210034/
```

Timestamps will differ after another run. Find the available weights with:

```bash
find outputs/runs/phase_11_architecture_* -type f -name checkpoint.pt -print | sort
```

The raw architecture-run directories are ignored by Git. A clean clone does
not include these generated weights; run the CPU smoke config to create a new
set.

## 2. The current model checkpoint

The checkpoint is produced by
`src/llm_lab/experiments/architecture_study.py` and contains:

```text
model_config       # enough information to reconstruct ModernLanguageModel
model_state_dict   # learned tensors
variant            # e.g. rms_norm_rope
seed               # initialization/training seed
final_metrics      # final validation and throughput measurements
```

The loader in `src/llm_lab/model/checkpoint.py` reconstructs the model from
`model_config`, loads `model_state_dict`, and switches it to evaluation mode.
The command-line helpers expose that loader:

```bash
.venv/bin/python scripts/inspect_checkpoint.py PATH_TO_CHECKPOINT
.venv/bin/python scripts/generate_from_checkpoint.py PATH_TO_CHECKPOINT \
  --prompt 'Alice was beginning' \
  --max-new-tokens 64
```

The second command uses the matching tokenizer by default:

```text
outputs/tokenizers/phase_07_tokenizer.json
```

Pass `--tokenizer PATH` when using another tokenizer, and make sure its
vocabulary size matches the checkpoint.

## 3. Model metadata used by the Phase 11 study

Every study variant uses this shared model shape:

| Field | Value |
|---|---:|
| vocabulary size | 512 |
| maximum context (`block_size`) | 128 |
| model width (`d_model`) | 64 |
| attention heads | 4 |
| head width | 16 |
| Transformer blocks | 2 |
| MLP width | 256 (`4 * d_model`) |
| activation | ReLU |
| attention | causal multi-head self-attention |
| optimizer | AdamW |
| training dtype | float32 |

The controlled variants are:

| Name | Normalization | Position method | Unique parameters |
|---|---|---|---:|
| `layer_norm_learned` | LayerNorm | learned position table | 122,624 |
| `rms_norm_learned` | RMSNorm | learned position table | 122,304 |
| `layer_norm_rope` | LayerNorm | RoPE on Q/K | 114,432 |
| `rms_norm_rope` | RMSNorm | RoPE on Q/K | 114,112 |

The output head is tied to `token_embeddings`, so the shared matrix is counted
once in the unique parameter numbers above. The serialized state dictionary
keeps both names (`token_embeddings.weight` and `output_head.weight`) because
both module paths refer to the same parameter.

## 4. Input artifacts

The current modern study consumes:

| Artifact | Path | Description |
|---|---|---|
| processed corpus | `data/processed/alice_in_wonderland_body.txt` | training text |
| BPE tokenizer | `outputs/tokenizers/phase_07_tokenizer.json` | 512-token vocabulary |
| training shards | `outputs/datasets/phase_08_alice/shard_000.pt` through `shard_007.pt` | 80,000 token IDs |
| validation shard | `outputs/datasets/phase_08_alice/shard_008.pt` | 4,335 token IDs |
| dataset metadata | `outputs/datasets/phase_08_alice/metadata.json` | hashes and shard counts |

The metadata reports 84,335 total token IDs and a configured shard size of
10,000. The architecture-study configs refer to these exact paths:

```text
configs/phase_11_architecture_smoke.yaml
configs/phase_11_architecture_study.yaml
```

## 5. Run artifact layout

Each architecture run follows this shape:

```text
outputs/runs/phase_11_architecture_<timestamp>/
├── study_config.yaml
└── runs/
    └── <variant>/
        └── seed_<seed>/
            ├── checkpoint.pt
            ├── final_metrics.json
            ├── generated.txt
            ├── metadata.json
            ├── metrics.jsonl
            └── resolved_config.yaml
```

The full five-seed study also writes its aggregate report to:

```text
outputs/reports/phase_11_architecture/
├── effects.csv
├── final_comparison.png
├── learning_curves.png
├── report_metadata.json
├── run_metrics.csv
├── samples.md
└── summary.csv
```

Read `resolved_config.yaml` before interpreting a result. It records the exact
variant, device, model values, data paths, and training settings used by that
run. Read `metadata.json` for the Python/PyTorch version, backend availability,
and seed.

## 6. Tiny-model artifacts

The earlier educational scripts use `TinyLanguageModel` instead:

```text
scripts/train_tiny_smoke.py
scripts/train_tiny_configured.py
scripts/generate_sample.py
```

Their outputs contain logs and generated text, for example:

```text
outputs/phase_05/
outputs/runs/phase_06_smoke_<timestamp>/
```

Those scripts currently do not save a reloadable model checkpoint. This is
intentional for the early learning phases; use the modern architecture study
when you need a saved model.

## 7. Tracked versus generated

The model source, configs, tokenizer, dataset metadata, and token shards are
repository artifacts. Study checkpoints and timestamped raw runs are generated
artifacts and are ignored so large binary files do not silently become part of
the source history. If a future checkpoint becomes a named release artifact,
give it an explicit versioned location and update this file rather than relying
on a timestamp alone.
