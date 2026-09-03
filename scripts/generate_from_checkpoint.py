import argparse
from pathlib import Path

import torch

from llm_lab.model.checkpoint import load_checkpoint
from llm_lab.tokenizer.bpe_tokenizer import TrainableBPETokenizer
from llm_lab.training.loop import generate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "outputs/tokenizers/phase_07_tokenizer.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text from a saved ModernLanguageModel checkpoint."
    )
    parser.add_argument("checkpoint", help="Path to checkpoint.pt")
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_TOKENIZER_PATH),
        help="Path to the matching TrainableBPETokenizer JSON file",
    )
    parser.add_argument(
        "--prompt",
        default="Alice was beginning",
        help="Text prompt to continue",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of greedy tokens to append",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Inference device; auto selects MPS when available",
    )
    return parser


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(device_name)


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens < 0:
        raise SystemExit("--max-new-tokens must be non-negative")

    device = resolve_device(args.device)
    loaded = load_checkpoint(args.checkpoint, device=device)
    tokenizer = TrainableBPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != loaded.config.vocab_size:
        raise SystemExit(
            "Tokenizer vocabulary size does not match checkpoint: "
            f"{tokenizer.vocab_size} != {loaded.config.vocab_size}"
        )

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("--prompt must contain at least one token")
    if len(prompt_ids) + args.max_new_tokens > loaded.config.block_size:
        raise SystemExit(
            "Prompt plus --max-new-tokens exceeds checkpoint block_size "
            f"({loaded.config.block_size})"
        )

    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = generate(
        loaded.model,
        start,
        max_tokens=args.max_new_tokens,
    )[0].tolist()

    print(f"Checkpoint: {loaded.path}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Generated text: {tokenizer.decode(generated_ids)!r}")


if __name__ == "__main__":
    main()
