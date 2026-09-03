# Generated outputs

This directory contains local tokenizers, tokenized data, training logs,
reports, and model checkpoints.

Use [`docs/MODEL_ARTIFACTS.md`](../docs/MODEL_ARTIFACTS.md) for the artifact
contract, checkpoint schema, and the exact commands for inspecting or generating
from a model.

The most important paths are:

```text
outputs/tokenizers/phase_07_tokenizer.json
outputs/datasets/phase_08_alice/
outputs/runs/phase_06_smoke_<timestamp>/
outputs/runs/phase_11_architecture_<timestamp>/
outputs/reports/phase_11_architecture/
```

`outputs/datasets/phase_08_alice/shard_*.pt` files are token data, not model
weights. Reloadable modern-model weights are named `checkpoint.pt` inside a
Phase 11 architecture run.
