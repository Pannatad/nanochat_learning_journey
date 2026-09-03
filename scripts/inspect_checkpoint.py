import argparse
from typing import Any

import torch

from llm_lab.model.checkpoint import load_checkpoint
from llm_lab.model.inspection import build_parameter_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a saved ModernLanguageModel checkpoint."
    )
    parser.add_argument("checkpoint", help="Path to checkpoint.pt")
    return parser


def print_metrics(metrics: Any) -> None:
    if not isinstance(metrics, dict):
        return

    print("Final metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main() -> None:
    args = build_parser().parse_args()
    loaded = load_checkpoint(args.checkpoint)
    report = build_parameter_report(loaded.model)
    payload = loaded.payload
    state_dict = payload["model_state_dict"]
    serialized_values = sum(
        tensor.numel() for tensor in state_dict.values() if torch.is_tensor(tensor)
    )

    print(f"Checkpoint: {loaded.path}")
    print(f"Variant: {payload.get('variant', 'unknown')}")
    print(f"Seed: {payload.get('seed', 'unknown')}")
    print("Model config:")
    for key, value in loaded.config.__dict__.items():
        print(f"  {key}: {value}")

    print("Parameters:")
    print(f"  unique: {report.total:,}")
    print(f"  trainable: {report.trainable:,}")
    print(f"  frozen: {report.frozen:,}")
    for component, count in report.by_component.items():
        print(f"  {component}: {count:,}")

    print(f"State-dict entries: {len(state_dict)}")
    print(f"Serialized tensor values: {serialized_values:,}")
    print_metrics(payload.get("final_metrics"))


if __name__ == "__main__":
    main()
