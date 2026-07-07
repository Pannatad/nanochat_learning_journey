import json

import torch

from llm_lab.data.tiny_dataset import make_next_token_example
from llm_lab.experiments.config import (
    collect_environment_metadata,
    create_run_dir,
    load_config,
    save_metadata,
    save_resolved_config,
    set_seed,
)
from llm_lab.model.tiny_lm import TinyLanguageModel
from llm_lab.tokenizer.byte_tokenizer import ByteTokenizer
from llm_lab.training.loop import generate, make_optimizer, train_step, validation_step


def make_next_batch(
    text: str,
    tokenizer: ByteTokenizer,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_id = tokenizer.encode(text)
    x, y = make_next_token_example(tokens_id, block_size=block_size)
    x = torch.tensor(x).unsqueeze(0)
    y = torch.tensor(y).unsqueeze(0)
    return x, y


def decode_generated_bytes(token_ids: list[int]) -> str:
    return bytes(token_ids).decode("utf-8", errors="replace")


def main() -> None:
    config = load_config("configs/phase_06_smoke.yaml")
    set_seed(config["seed"])

    run_dir = create_run_dir(
        config["output"]["root_dir"],
        "phase_06_smoke",
    )
    save_resolved_config(config, run_dir)

    metadata = collect_environment_metadata(config["seed"])
    save_metadata(metadata, run_dir)

    block_size = config["training"]["block_size"]

    tokenizer = ByteTokenizer()
    model = TinyLanguageModel(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
    )
    optimizer = make_optimizer(
        model,
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    train_text = config["data"]["train_text"]
    val_text = config["data"]["val_text"]
    x_train, y_train = make_next_batch(train_text, tokenizer, block_size)
    x_val, y_val = make_next_batch(val_text, tokenizer, block_size)

    log_file = run_dir / "train_log.jsonl"
    with log_file.open("w", encoding="utf-8") as log:
        for step in range(config["training"]["steps"]):
            train_loss = train_step(model, optimizer, x_train, y_train)
            record = {
                "step": step,
                "train_loss": train_loss.item(),
            }

            log.write(json.dumps(record) + "\n")

    val_loss = validation_step(model, x_val, y_val)
    with log_file.open("a", encoding="utf-8") as log:
        record = {
            "step": config["training"]["steps"],
            "val_loss": val_loss.item(),
        }
        log.write(json.dumps(record) + "\n")

    prompt_token_ids = torch.tensor([tokenizer.encode(config["generation"]["prompt"])])
    generated_ids = generate(
        model,
        prompt_token_ids,
        max_tokens=config["generation"]["max_new_tokens"],
    )[0].tolist()

    generated_text = decode_generated_bytes(generated_ids)

    generated_file = run_dir / "generated.txt"
    generated_file.write_text(generated_text, encoding="utf-8")

    print(f"Wrote run outputs to {run_dir}")


if __name__ == "__main__":
    main()
