# Phase 06: Reproducible Experiment System

This phase made the tiny training run reproducible and comparable. The main
change was moving hardcoded experiment settings out of Python and into a YAML
config file.

The experiment flow is:

$$
\text{config}
\rightarrow \text{run folder}
\rightarrow \text{resolved config + metadata + logs + generated text}
$$

## Config File

A config file is the recipe for an experiment.

The baseline config lives at:

```text
configs/phase_06_smoke.yaml
```

It stores values such as:

```yaml
seed: 1234

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

This makes experiments easier to change without editing training code.

## Loading Configs

`load_config` reads a YAML file and returns a Python dictionary.

Example YAML:

```yaml
training:
  steps: 10
```

Loaded Python object:

```python
{"training": {"steps": 10}}
```

The YAML parser is `PyYAML`, and the safe loading function is:

```python
yaml.safe_load(file)
```

## Run Directories

Each experiment run gets its own output folder:

```text
outputs/runs/phase_06_smoke_YYYYMMDD_HHMMSS/
```

This prevents runs from overwriting each other. It also makes each experiment
easy to inspect later.

A run folder contains:

```text
resolved_config.yaml
metadata.json
train_log.jsonl
generated.txt
```

## Resolved Config

The resolved config is the exact config saved inside one run folder:

```text
resolved_config.yaml
```

The baseline config is the starting recipe. The resolved config is the recipe
actually used by a specific run.

In this phase, the baseline and resolved config usually match. Later, command
line overrides or defaults could make the resolved config different.

## Seed

The seed controls randomness.

This phase sets:

```python
random.seed(seed)
torch.manual_seed(seed)
```

The seed affects things like model initialization and random tensors. Saving it
helps make repeated runs easier to compare.

A seed does not guarantee identical behavior across every machine or backend,
but it is one of the first pieces of information needed for reproducibility.

## Metadata

Metadata records the environment that produced a run.

The metadata file is:

```text
metadata.json
```

It stores:

```text
seed
python_version
platform
torch_version
cuda_available
mps_available
```

This helps answer questions like:

```text
Which Python version ran this?
Which PyTorch version ran this?
Was CUDA or MPS available?
What seed was used?
```

## Config Differences

Config comparison helps explain why two runs differ.

Example baseline:

```yaml
optimizer:
  lr: 0.01
```

Example candidate:

```yaml
optimizer:
  lr: 0.001
```

Diff output:

$$
\text{optimizer.lr}: 0.01 \rightarrow 0.001
$$

`diff_configs` walks through nested dictionaries recursively. If both values
are dictionaries, it goes deeper. If two values differ, it records the dotted
path.

Example nested path:

```text
training.steps
optimizer.lr
generation.prompt
```

`scripts/compare_configs.py` makes this available from the command line.

## Configured Training Script

`scripts/train_tiny_configured.py` is the Phase 6 version of the smoke run. It
reads the YAML config and uses it to create:

```text
tokenizer
model
optimizer
train batch
validation batch
```

Then it writes the run artifacts into the run directory:

```text
resolved_config.yaml
metadata.json
train_log.jsonl
generated.txt
```

## Small Example

Running:

```bash
.venv/bin/python scripts/train_tiny_configured.py
```

creates a folder like:

```text
outputs/runs/phase_06_smoke_20260706_150838/
```

That folder is one experiment record.

Comparing the baseline config with that run's resolved config:

```bash
.venv/bin/python scripts/compare_configs.py \
  configs/phase_06_smoke.yaml \
  outputs/runs/phase_06_smoke_20260706_150838/resolved_config.yaml
```

prints:

```text
No config differences.
```

If a copied config changes `optimizer.lr` from `0.01` to `0.001`, the diff
prints:

$$
\text{optimizer.lr}: 0.01 \rightarrow 0.001
$$
