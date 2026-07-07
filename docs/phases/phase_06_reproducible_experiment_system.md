# Phase 06: Reproducible Experiment System

## What We Built

We built the first reproducible experiment system. The project can now load a
YAML config, create a unique run directory, save the exact config used for a
run, save environment metadata, set the random seed, run the tiny training loop
from config values, and compare config differences.

The main run flow is:

```text
config
-> run folder
-> resolved config + metadata + logs + generated text
```

## What I Learned

- I learned that a config file is the recipe for an experiment.
- I learned that a run directory keeps one experiment's outputs together.
- I learned that the resolved config is a saved copy of the config actually
  used by a run.
- I learned that metadata records environment details such as seed, Python
  version, PyTorch version, and platform.
- I learned that setting a seed makes random initialization and random tensors
  more repeatable.
- I learned that config diffs help compare two runs by showing which settings
  changed.

## High-Level Understanding

Phase 6 makes training runs easier to reproduce and compare. Instead of hiding
experiment settings inside Python code, the settings live in:

```text
configs/phase_06_smoke.yaml
```

The configured training script reads that file and writes each run into its own
folder under:

```text
outputs/runs/
```

Each run folder contains the files needed to understand what happened:

```text
resolved_config.yaml
metadata.json
train_log.jsonl
generated.txt
```

## Intuition / Small Example

Without Phase 6, the smoke script has values like this inside Python:

```python
steps = 10
block_size = 16
lr = 0.01
```

With Phase 6, those values live in YAML:

```yaml
training:
  steps: 10
  block_size: 16

optimizer:
  lr: 0.01
  weight_decay: 0.1
```

Python loads the YAML as a dictionary:

```python
config["training"]["steps"]
config["optimizer"]["lr"]
```

Then the run saves a copy of that config into the run folder. Later, if two
runs behave differently, the configs can be compared.

## Detailed Explanation

### Config Loading

`load_config` reads a YAML file and returns a Python dictionary:

```text
YAML file -> Python dict
```

For example:

```yaml
training:
  steps: 10
```

becomes:

```python
{"training": {"steps": 10}}
```

### Run Directory

`create_run_dir` creates a timestamped output directory:

```text
outputs/runs/phase_06_smoke_YYYYMMDD_HHMMSS/
```

This prevents new experiments from overwriting older logs and generated text.

### Resolved Config

`save_resolved_config` writes:

```text
resolved_config.yaml
```

inside the run folder. This file is the exact config used for that run.

The difference between baseline config and resolved config is:

```text
baseline config: the starting recipe
resolved config: the saved recipe for one actual run
```

In this phase they usually match. Later, command-line overrides or generated
defaults may make the resolved config different from the baseline.

### Seed

`set_seed` sets Python and PyTorch randomness:

```python
random.seed(seed)
torch.manual_seed(seed)
```

This helps make model initialization and random tensors repeatable. A seed does
not guarantee perfect reproducibility across every machine and backend, but it
is one of the first things an experiment system should record.

### Metadata

`collect_environment_metadata` collects run information:

```text
seed
python_version
platform
torch_version
cuda_available
mps_available
```

`save_metadata` writes that dictionary to:

```text
metadata.json
```

This makes the run folder explain not only what config was used, but also what
environment ran it.

### Config Differences

`diff_configs` compares two config dictionaries. It reports nested values with
dotted paths:

```text
optimizer.lr: 0.01 -> 0.001
training.steps: 10 -> 20
```

The helper walks through nested dictionaries recursively. If both values are
dictionaries, it goes deeper. If the values are different, it records the
change.

`scripts/compare_configs.py` exposes this from the command line.

### Configured Training Script

`scripts/train_tiny_configured.py` is the Phase 6 version of the smoke run. It
uses config values to create:

```text
tokenizer
model
optimizer
train batch
validation batch
```

Then it writes:

```text
train_log.jsonl
generated.txt
```

into the same run directory as the resolved config and metadata.

## Experiments To Try

- Change `optimizer.lr` in a copied config and compare it with the baseline.
- Change `training.steps` and confirm the JSONL log has the expected number of
  training rows plus one validation row.
- Run `scripts/train_tiny_configured.py` twice and compare the two resolved
  configs.
- Change `seed` and check how the generated text changes.
- Add a second config file for a slightly different smoke run.

## Tests / Checks

```bash
.venv/bin/python scripts/train_tiny_configured.py
.venv/bin/python scripts/compare_configs.py configs/phase_06_smoke.yaml outputs/runs/<run_dir>/resolved_config.yaml
.venv/bin/pytest
.venv/bin/ruff check .
```

Expected result:

- the configured training script creates a new run directory;
- the run directory contains `resolved_config.yaml`;
- the run directory contains `metadata.json`;
- the run directory contains `train_log.jsonl`;
- the run directory contains `generated.txt`;
- config comparison reports no differences for the baseline and matching
  resolved config;
- config, training, and earlier phase tests pass;
- Ruff reports all checks passed.
