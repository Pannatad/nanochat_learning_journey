# Phase Learning Notes

This folder is the public learning record for the LLM From Scratch Lab.

Each phase documents what was built, what was learned, what experiments are
worth trying, and what questions remain open. The goal is that another learner
can follow the same path step by step.

## Phase Index

| Phase | Topic | Difficulty | Status |
|---|---|---|---|
| [Phase 0](phase_00_repository_foundation.md) | Repository foundation | Easy | Complete |
| [Phase 1](phase_01_tiny_tokenizer_and_dataset.md) | Tiny educational tokenizer and dataset | Medium | Complete |
| [Phase 2](phase_02_smallest_decoder_only_lm.md) | Smallest decoder-only LM | Medium | Complete |
| [Phase 3](phase_03_manual_attention_learning.md) | Manual attention learning phase | Hard | Complete |
| [Phase 4](phase_04_transformer_block_baseline.md) | Transformer block baseline | Hard | Complete |
| [Phase 5](phase_05_minimal_pretraining_loop.md) | Minimal pretraining loop | Hard | Complete |
| [Phase 6](phase_06_reproducible_experiment_system.md) | Reproducible experiment system | Medium | Complete |
| [Phase 7](phase_07_trainable_bpe_tokenizer.md) | Trainable BPE tokenizer | Hard | Complete |
| Phase 8 | Tokenized data pipeline | Medium | Planned |
| Phase 9 | nanochat-style architecture baseline | Very hard | Planned |
| Phase 10 | RMSNorm | Medium | Planned |
| Phase 11 | RoPE | Hard | Planned |
| Phase 12 | QK norm | Hard | Planned |
| Phase 13 | Activation experiments | Medium | Planned |
| Phase 14 | Untied embeddings | Medium | Planned |
| Phase 15 | GQA | Very hard | Planned |
| Phase 16 | Sliding-window attention | Very hard | Planned |
| Phase 17 | Optimized attention adapters | Very hard | Planned |
| Phase 18 | Explicit dtype and device policy | Hard | Planned |
| Phase 19 | Optimizer efficiency and Muon | Very hard | Planned |
| Phase 20 | Scaling presets | Hard | Planned |
| Phase 21 | Evaluation suite | Hard | Planned |
| Phase 22 | SFT data format | Medium | Planned |
| Phase 23 | SFT training loop | Hard | Planned |
| Phase 24 | RL data and reward interface | Very hard | Planned |
| Phase 25 | Minimal RL loop | Very hard | Planned |
| Phase 26 | Inference engine | Hard | Planned |
| Phase 27 | Chat CLI | Medium | Planned |
| Phase 28 | Simple chat UI | Medium | Planned |
| Phase 29 | Experiment matrix | Hard | Planned |
| Phase 30 | First complete nanochat-style release | Very hard | Planned |

## Phase Note Style

```markdown
# Phase NN: Title

Short intro explaining what this phase added and why it matters.

## Main Concept
Explain the central idea in plain language.

## Related Concept
Explain another important idea from the phase.

## Small Example
Use tiny numbers, shapes, or a short code snippet.

Continue with concept sections as needed. Phase documents should read like
learning notes, not like chat responses. Do not use the fixed sections
`High-Level Understanding`, `Intuition / Small Example`, or
`Detailed Explanation` in the phase files.
```
