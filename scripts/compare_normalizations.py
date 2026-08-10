import json

import torch

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.inspection import build_parameter_report
from llm_lab.model.modern_lm import ModernLanguageModel
from llm_lab.training.loop import (
    make_optimizer,
    train_step,
    validation_step,
)


def run_experiment(
    normalization: str,
    x: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, str | int | float]:
    # Reset the seed so shared model parameters begin identically.
    torch.manual_seed(0)
    # Construct the config and model.
    config = ModernModelConfig(
        vocab_size=32,
        block_size=8,
        d_model=16,
        n_head=4,
        n_layer=2,
        normalization=normalization,
    )
    model = ModernLanguageModel(config)
    # Record the parameter count.
    report = build_parameter_report(model)
    # Calculate loss before training.
    initial_loss = validation_step(model, x, targets)
    # Create the optimizer.
    optimizer = make_optimizer(model, lr=0.01, weight_decay=0.0)
    # Run the same small number of training steps.
    steps = 5
    for step in range(steps):
        train_step(model, optimizer, x, targets)
    # Calculate loss after training.
    final_loss = validation_step(model, x, targets)
    # Return one result dictionary.
    return {
        "normalization": normalization,
        "parameter_count": report.total,
        "initial_loss": initial_loss.item(),
        "steps": steps,
        "final_loss": final_loss.item(),
    }


def main() -> None:
    # Create one fixed x and target batch.
    x = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=torch.long,
    )

    results = [
        run_experiment("layer_norm", x, targets),
        run_experiment("rms_norm", x, targets),
    ]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
