# LLM From Scratch Lab: nanochat-Style Learning Blueprint

## 1. Purpose

This repository is a public learning lab for building a small ChatGPT-like
system step by step.

The destination is inspired by Andrej Karpathy's `nanochat`: a compact,
hackable, end-to-end LLM system that includes tokenizer training, pretraining,
supervised finetuning, reinforcement learning, evaluation, inference, and a chat
interface.

This repository should not copy `nanochat` directly. It should rebuild the
ideas in a slower, more educational way so each concept can be understood,
tested, modified, and documented.

The repo should make it possible to answer:

1. What did we build in this phase?
2. What new concept did we learn?
3. What are the tensor shapes and data contracts?
4. What can be swapped for an experiment?
5. What changed from the previous baseline?
6. What tests prove the behavior?
7. What questions remain open?

The code matters, but the learning journey matters too.

## 2. Starting Assumptions

Unless deliberately changed later, build with these choices:

- Language: Python 3.11+
- ML framework: PyTorch
- First model family: decoder-only Transformer
- First objective: causal next-token prediction
- First learning tokenizer: transparent byte tokenizer
- Main tokenizer destination: trainable GPT-4-style BPE tokenizer
- Main training stages: pretraining, SFT, minimal RL, inference/chat
- First execution mode: single process and one device
- Supported devices: CPU, CUDA, and Apple MPS
- Configuration: explicit, inspectable Python dataclasses loaded from YAML
- Testing: `pytest`
- Formatting and linting: Ruff
- Experiment output: local folders with config, metrics, logs, samples, and
  checkpoints
- Public learning record: `docs/phases/`

Do not start with distributed training, Lightning, Hydra, online experiment
tracking, or a large plugin framework. These can hide the execution path that
this project is meant to teach.

## 3. Core Design Principles

### 3.1 Start Small, Then Modernize

The first runnable model should be intentionally simple. Advanced nanochat-style
ideas are added one at a time after the baseline works.

Example progression:

```text
byte tokenizer
  -> tiny dataset
  -> manual attention
  -> simple GPT block
  -> pretraining loop
  -> reproducible experiments
  -> trainable BPE
  -> modern architecture options
  -> SFT
  -> RL
  -> chat inference
```

### 3.2 One Hard Topic Per Phase

Hard topics should not be compressed into one large phase. BPE, RoPE, RMSNorm,
QK norm, GQA, sliding-window attention, Muon, SFT, RL, and KV cache each get
their own phase.

Each phase must leave a working vertical slice.

### 3.3 Experimentation Is a First-Class Feature

The repo must make it easy to try:

- new tokenizers;
- new architecture blocks;
- new activation functions;
- different normalization methods;
- different attention implementations;
- different optimizer choices;
- different model depth, width, and head counts;
- different data mixtures;
- different SFT task mixtures;
- different RL reward functions;
- different precision and device policies.

Every experiment starts from a named baseline and changes the smallest practical
number of variables.

### 3.4 Public Learning Notes Are Required

Every phase must update `docs/phases/`.

A phase is not complete until:

1. the code works;
2. focused tests pass;
3. the smallest relevant command runs;
4. the phase learning note exists or is updated;
5. experiments to try are documented.

The notes should be useful to other learners. They should record mistakes,
confusing parts, tradeoffs, and open questions, not only polished final answers.

### 3.5 Explicit Contracts

Every public component should state:

- accepted inputs;
- returned outputs;
- tensor shapes;
- data types;
- important invariants;
- device behavior;
- checkpoint state, when relevant.

Keep model math out of training coordinators. Keep losses outside model forward
methods. Keep optimizer creation outside model classes unless a later phase
explicitly studies a nanochat-style integrated optimizer setup.

## 4. Required Explanation Format

Learning docs and code explanations should use:

1. **High-Level Understanding**
2. **Intuition / Small Example**
3. **Detailed Explanation**

Keep explanations concise unless more detail is requested.

## 5. Public Phase Documentation

Create a dedicated public learning folder:

```text
docs/phases/
├── README.md
├── phase_00_repository_foundation.md
├── phase_01_tiny_tokenizer_and_dataset.md
├── phase_02_smallest_decoder_lm.md
├── phase_03_manual_attention.md
├── phase_04_transformer_block_baseline.md
├── phase_05_minimal_pretraining_loop.md
└── ...
```

Each phase document should follow this template:

```markdown
# Phase NN: Title

## What We Built
Short summary of the code or system added.

## What I Learned
The main concepts learned in this phase.

## High-Level Understanding
The simple mental model.

## Intuition / Small Example
A tiny example, toy numbers, or diagram.

## Detailed Explanation
Only the necessary technical details.

## Experiments To Try
Small controlled variations.

## Tests / Checks
Commands and expected results.

## Open Questions
Things to revisit later.
```

`docs/phases/README.md` is the public table of contents. It should explain what
each phase teaches and link to the phase notes as they are created.

## 6. Proposed Repository Structure

Do not scaffold every file in advance. This structure is the destination map,
not the Phase 0 file list.

```text
llm-from-scratch-lab/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version
├── configs/
│   ├── base.yaml
│   ├── tokenizer/
│   ├── model/
│   ├── data/
│   ├── training/
│   ├── sft/
│   ├── rl/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── docs/
│   ├── phases/
│   ├── architecture_decisions.md
│   ├── tensor_shape_ledger.md
│   └── glossary.md
├── scripts/
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── pretrain.py
│   ├── sft.py
│   ├── rl.py
│   ├── evaluate.py
│   ├── chat_cli.py
│   └── chat_web.py
├── src/
│   └── llm_lab/
│       ├── __init__.py
│       ├── config.py
│       ├── tokenization/
│       ├── data/
│       ├── models/
│       ├── objectives/
│       ├── optimization/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── experiments/
│       └── utils/
└── tests/
```

## 7. Configuration and Experiment Contracts

Configuration should be explicit and inspectable.

Initial config groups:

- `experiment`: name, baseline, seed, notes
- `tokenizer`: kind, vocab size, special tokens, path
- `data`: dataset identity, split, sequence length, shard policy
- `model`: architecture, depth, width, heads, attention, MLP, normalization
- `training`: batch size, steps, precision, device, logging, checkpoints
- `optimizer`: optimizer kind, learning rates, weight decay, parameter groups
- `evaluation`: loss, bits-per-byte, samples, task evals
- `sft`: conversation rendering, assistant mask, task mixture
- `rl`: prompt source, reward function, sampling policy, objective

Every run must save:

- resolved config;
- config diff from baseline;
- Git state when available;
- Python, PyTorch, device, and dtype information;
- tokenizer identity and vocabulary size;
- dataset identity;
- model parameter count;
- training metrics;
- generated samples;
- checkpoints, when configured.

## 8. Tokenization Roadmap

The tokenizer path has two separate goals:

1. teach tokenization clearly;
2. move toward a nanochat-style trainable BPE tokenizer.

Phases should progress from:

```text
byte tokenizer
  -> tokenizer tests
  -> GPT-4-style split pattern concept
  -> trainable BPE
  -> special tokens
  -> conversation rendering
  -> SFT masks
  -> RL completion prompts
```

Special tokens should support chat-style data:

```text
<|bos|>
<|user_start|>
<|user_end|>
<|assistant_start|>
<|assistant_end|>
<|python_start|>
<|python_end|>
<|output_start|>
<|output_end|>
```

GPT-2 BPE may remain as a compatibility or reference tokenizer, but it is no
longer the main destination.

## 9. Model Architecture Roadmap

Start with a simple decoder-only Transformer. Then add nanochat-style ideas one
at a time.

Baseline components:

- token embeddings;
- positional information;
- causal self-attention;
- MLP;
- residual connections;
- normalization;
- logits head.

Advanced components:

- RMSNorm;
- RoPE;
- QK norm;
- ReLU-squared or SwiGLU-style activation experiments;
- untied embeddings;
- grouped-query attention;
- sliding-window attention;
- SDPA adapter;
- optional Flash Attention adapter;
- explicit dtype casting policy;
- KV-cache inference.

Each advanced component must be replaceable through config and must include a
focused test or numerical comparison.

## 10. Training Stage Roadmap

The repo should eventually support these stages:

```text
tokenizer training
  -> base model pretraining
  -> evaluation
  -> supervised finetuning
  -> minimal reinforcement learning
  -> inference/chat
```

Do not implement all stages at once. Each stage receives its own phase and
learning note.

### Pretraining

Pretraining teaches next-token prediction on plain text or token shards.

Required concepts:

- shifted targets;
- cross-entropy objective outside the model;
- AdamW baseline;
- learning-rate schedule;
- checkpointing;
- validation loss;
- generated samples;
- resume behavior.

### SFT

SFT teaches the model to answer in a chat format.

Required concepts:

- conversation schema;
- chat special tokens;
- assistant-only loss masks;
- loading from a pretrained checkpoint;
- task mixture;
- chat behavior evaluation.

### RL

RL should start minimal and educational.

Required concepts:

- prompt rendering;
- sampled completions;
- reward function contract;
- baseline comparison to SFT;
- reward logging;
- failure modes.

Avoid overcomplicating RL early. The goal is to understand the loop before
chasing performance.

## 11. Optimizer and Efficiency Roadmap

Start with AdamW because it is easier to understand and test.

Then add efficiency topics one at a time:

- parameter groups;
- fused optimizer availability checks;
- learning-rate schedule experiments;
- gradient accumulation;
- explicit dtype policy;
- Muon-style matrix-parameter optimization;
- throughput and memory metrics.

Muon should be its own learning phase. The phase should explain what problem it
tries to solve, which parameters it applies to, how it differs from AdamW, and
how to compare it fairly.

## 12. Evaluation Roadmap

Use multiple levels of evaluation:

- import and unit tests;
- tensor-shape tests;
- numerical tests for attention and losses;
- tiny overfit test;
- validation loss;
- bits-per-byte;
- generated samples;
- small task evaluations;
- SFT/chat evaluations;
- RL reward curves.

Evaluation should be lightweight first. Larger benchmark adapters can come after
the training stages are stable.

## 13. Bite-Size Phase Roadmap

The implementation proceeds phase by phase. Do not scaffold all future source
modules as empty placeholders.

### Phase 0: Repository Foundation

Create:

- Git repository;
- `README.md`;
- `AGENTS.md`;
- `pyproject.toml`;
- `src/` package layout;
- Ruff and pytest setup;
- basic CI;
- `.gitignore`;
- `.python-version`;
- initial `docs/phases/` table of contents.

Acceptance criteria:

- editable install succeeds;
- `python -c "import llm_lab"` succeeds;
- Ruff and an initial test pass;
- `docs/phases/phase_00_repository_foundation.md` exists.

### Phase 1: Tiny Educational Tokenizer and Dataset

Implement:

- byte tokenizer;
- tiny text dataset;
- encode/decode tests;
- shifted batch examples.

Acceptance criteria:

- tokenizer round trip works for simple strings;
- shifted input/target examples are documented;
- phase note explains why next-token prediction shifts by one.

### Phase 2: Smallest Decoder-Only LM

Implement:

- token embeddings;
- minimal forward pass;
- logits shape `(B, T, V)`;
- cross-entropy objective outside the model.

Acceptance criteria:

- import, shape, and finite-loss tests pass;
- phase note explains logits and targets.

### Phase 3: Manual Attention Learning Phase

Implement:

- single-head causal attention by hand;
- causal mask;
- attention score inspection;
- tiny numerical test.

Acceptance criteria:

- masked future tokens cannot be attended to;
- manual attention test passes;
- phase note includes a toy attention example.

### Phase 4: Transformer Block Baseline

Implement:

- multi-head attention;
- MLP;
- residual connections;
- LayerNorm baseline;
- stacked blocks.

Acceptance criteria:

- block preserves `(B, T, C)` shape;
- forward pass remains finite;
- tensor-shape ledger is updated.

### Phase 5: Minimal Pretraining Loop

Implement:

- AdamW builder;
- one training step;
- validation loss;
- basic generation;
- CSV or JSONL logging.

Acceptance criteria:

- 10-step CPU smoke run completes;
- loss is logged;
- generated text is saved;
- phase note explains the training loop.

### Phase 6: Reproducible Experiment System

Implement:

- YAML config loading;
- resolved config saving;
- run directories;
- config diff from baseline;
- seed and environment metadata.

Acceptance criteria:

- two runs can be compared;
- resolved config is saved;
- phase note explains baseline versus run.

### Phase 7: Trainable BPE Tokenizer

Implement:

- GPT-4-style split pattern concept;
- BPE training on a tiny corpus;
- vocab and merge inspection;
- tokenizer save/load.

Acceptance criteria:

- tokenizer can be trained and reloaded;
- compression statistics are reported;
- phase note explains merge intuition.

### Phase 8: Tokenized Data Pipeline

Implement:

- prepared token shards;
- deterministic shard loader;
- loader state for resume;
- dataset identity metadata.

Acceptance criteria:

- batches are reproducible;
- loader state can be restored;
- phase note explains shard boundaries.

### Phase 9: nanochat-Style Architecture Baseline

Implement:

- a modern baseline config separate from the simple learning model;
- clean component factories;
- parameter count report.

Acceptance criteria:

- simple baseline still works;
- modern baseline instantiates;
- phase note explains why architecture choices are isolated.

### Phase 10: RMSNorm

Implement:

- RMSNorm module;
- config switch against LayerNorm;
- numerical sanity tests.

Acceptance criteria:

- RMSNorm preserves shape;
- baseline comparison runs;
- phase note explains RMSNorm versus LayerNorm.

### Phase 11: RoPE

Implement:

- rotary position embeddings;
- config switch against learned position embeddings;
- small shape and rotation tests.

Acceptance criteria:

- RoPE attention forward pass works;
- phase note includes a small rotation intuition.

### Phase 12: QK Norm

Implement:

- query/key normalization option;
- attention comparison against baseline.

Acceptance criteria:

- QK norm path passes tests;
- phase note explains what is normalized and why.

### Phase 13: Activation Experiments

Implement:

- GELU baseline;
- ReLU-squared option;
- SiLU or SwiGLU-style option.

Acceptance criteria:

- activation can be selected by config;
- one controlled comparison is recorded;
- phase note explains activation effects.

### Phase 14: Untied Embeddings

Implement:

- tied versus untied token embedding and LM head;
- parameter count comparison.

Acceptance criteria:

- both modes run;
- parameter count difference is reported;
- phase note explains the tradeoff.

### Phase 15: GQA

Implement:

- grouped-query attention option;
- query head versus key/value head config;
- inference-focused explanation.

Acceptance criteria:

- GQA forward pass works;
- invalid head groupings fail clearly;
- phase note explains memory and inference motivation.

### Phase 16: Sliding-Window Attention

Implement:

- local attention window option;
- full/local pattern config;
- comparison with full attention.

Acceptance criteria:

- window mask is tested;
- phase note explains context tradeoffs.

### Phase 17: Optimized Attention Adapters

Implement:

- PyTorch SDPA adapter;
- optional Flash Attention adapter when available;
- manual attention reference comparison.

Acceptance criteria:

- manual and optimized outputs match within tolerance;
- unavailable kernels fail or skip clearly;
- phase note explains why optimized attention hides details.

### Phase 18: Explicit Dtype and Device Policy

Implement:

- CPU, MPS, and CUDA device selection;
- fp32, bf16, and fp16 policy;
- explicit runtime metadata.

Acceptance criteria:

- CPU path remains default-safe;
- unsupported precision combinations fail clearly;
- phase note explains dtype tradeoffs.

### Phase 19: Optimizer Efficiency and Muon

Implement:

- AdamW parameter groups;
- Muon-style optimizer path for matrix-like parameters;
- fair comparison protocol.

Acceptance criteria:

- AdamW baseline remains available;
- Muon experiment records config and metrics;
- phase note explains what Muon changes.

### Phase 20: Scaling Presets

Implement:

- optional depth-based presets inspired by nanochat;
- explicit override configs;
- parameter and token budget reports.

Acceptance criteria:

- depth preset can derive a model config;
- explicit overrides are still visible;
- phase note explains preset versus explicit config.

### Phase 21: Evaluation Suite

Implement:

- validation loss;
- bits-per-byte;
- generated sample inspection;
- small task-style evaluations.

Acceptance criteria:

- evaluation outputs structured metrics;
- phase note explains what each metric can and cannot prove.

### Phase 22: SFT Data Format

Implement:

- conversation schema;
- chat special tokens;
- assistant-only loss masks;
- SFT dataset examples.

Acceptance criteria:

- rendered conversations have matching token and mask lengths;
- user tokens are masked out of the loss;
- phase note explains supervised chat formatting.

### Phase 23: SFT Training Loop

Implement:

- load pretrained checkpoint;
- supervised finetuning loop;
- SFT evaluation samples.

Acceptance criteria:

- tiny SFT smoke run completes;
- assistant-only loss is finite;
- phase note explains how SFT differs from pretraining.

### Phase 24: RL Data and Reward Interface

Implement:

- prompt rendering for completions;
- reward function protocol;
- tiny reward examples.

Acceptance criteria:

- reward function can score completions;
- phase note explains prompt, completion, reward.

### Phase 25: Minimal RL Loop

Implement:

- sample completions;
- compute rewards;
- update with a small RL objective;
- compare against SFT baseline.

Acceptance criteria:

- tiny RL smoke run completes;
- reward metrics are logged;
- phase note explains failure modes.

### Phase 26: Inference Engine

Implement:

- generation cleanup;
- sampling controls;
- KV cache introduction.

Acceptance criteria:

- cached and uncached generation agree where expected;
- phase note explains KV cache intuition.

### Phase 27: Chat CLI

Implement:

- load model and tokenizer;
- render conversation history;
- stream or print assistant replies.

Acceptance criteria:

- local CLI chat works with a tiny checkpoint;
- phase note explains chat state.

### Phase 28: Simple Chat UI

Implement:

- minimal local web UI;
- API route or local server;
- basic conversation display.

Acceptance criteria:

- UI can send prompt and display reply;
- phase note explains UI/data flow.

### Phase 29: Experiment Matrix

Implement:

- controlled comparisons across tokenizer, architecture, optimizer, data mix,
  precision, SFT, RL, and scale;
- report generation.

Acceptance criteria:

- experiment matrix is documented;
- at least one comparison per category has a template;
- phase note explains how to avoid confounded comparisons.

### Phase 30: First Complete nanochat-Style Release

Complete:

- tokenizer training;
- pretraining;
- evaluation;
- SFT;
- minimal RL experiment;
- inference;
- chat CLI or UI;
- public learning notes.

Acceptance criteria:

- a new learner can follow the README;
- all phase docs are linked from `docs/phases/README.md`;
- tests and lint pass;
- the repo can produce and chat with a tiny model end to end.

## 14. Recommended First Experiment Matrix

Do not run all combinations. Start with controlled comparisons:

| Experiment | Baseline | Single primary change | Main question |
|---|---|---|---|
| `activation_relu_squared` | `baseline_tiny` | GELU -> ReLU-squared | Does this improve learning or speed? |
| `norm_rmsnorm` | `baseline_tiny` | LayerNorm -> RMSNorm | What changes when mean subtraction is removed? |
| `position_rope` | `baseline_tiny` | learned positions -> RoPE | How does position information move into attention? |
| `attention_sdpa` | `manual_attention` | manual -> SDPA | Same outputs, better speed? |
| `optimizer_muon` | `adamw_baseline` | AdamW -> Muon-style matrix optimizer | What changes in optimizer behavior? |
| `tokenizer_bpe_8k` | `byte_tokenizer` | byte -> BPE | How does compression affect training? |
| `sft_mask_assistant_only` | `pretrain_baseline` | full-token loss -> assistant-only loss | Why does chat SFT need masks? |
| `gqa_small` | `full_mha` | MHA -> GQA | What memory or speed changes? |
| `dtype_bfloat16` | `float32_cuda` | fp32 -> bf16 | What speed and stability tradeoffs appear? |

For every comparison, keep the dataset, seed, total tokens, evaluation process,
and runtime precision fixed unless that variable is the experiment.

## 15. Definition of Done for Each Phase

Each phase is complete only when:

- the intended vertical slice runs;
- focused tests pass;
- lint passes for touched files;
- commands are recorded in the phase note;
- the phase note includes `What I Learned`;
- at least one follow-up experiment is listed;
- open questions are documented honestly.

## 16. Definition of Done for First Public Release

Version `0.1.0` is complete when:

- the README explains how to run the project from scratch;
- Phase 0 through Phase 30 notes are linked;
- the tiny byte-tokenizer learning path works;
- the trainable BPE path works;
- the simple and nanochat-style model baselines work;
- pretraining, SFT, and minimal RL smoke runs complete;
- generated samples are saved;
- inference works through CLI or UI;
- controlled experiment configs are available;
- tests and Ruff pass in CI;
- the documentation uses the required three-part explanation format.

## 17. Agent Implementation Instructions

Give the following instruction together with this document to any building
agent:

```text
Build the repository described in LLM_FROM_SCRATCH_REPO_BLUEPRINT.md.

Work phase by phase. Complete and verify one vertical slice before expanding
the number of components. Do not scaffold all future modules as empty
placeholders.

For each phase:
1. state the files and contracts being added;
2. implement the main structure first;
3. add focused tests;
4. run tests and lint;
5. run the smallest relevant end-to-end command;
6. update docs/phases/ for the phase;
7. record what was learned, what was confusing, and what remains open;
8. report what is verified and what remains device-specific.

Keep source files focused. Keep scripts thin. Keep the execution path explicit.
Do not introduce Lightning, Hydra, distributed training, dynamic plugin
discovery, or required online experiment tracking unless separately requested.

When a design detail is not specified, choose the simplest explicit solution
that preserves component replaceability, experiment value, and learning value.
Record material decisions in docs/architecture_decisions.md.
```

