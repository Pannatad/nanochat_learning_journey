import json
from pathlib import Path

import torch

from llm_lab.data.tiny_dataset import make_next_token_example
from llm_lab.model.tiny_lm import TinyLanguageModel
from llm_lab.tokenizer.byte_tokenizer import ByteTokenizer
from llm_lab.training.loop import (
    generate,
    make_optimizer,
    train_step,
    validation_step,
)


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
    steps = 10
    block_size = 16
    output_dir = Path("outputs/phase_05")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteTokenizer()
    model = TinyLanguageModel(vocab_size=tokenizer.vocab_size, d_model=16)
    optimizer = make_optimizer(model, lr=0.01, weight_decay=0.1)

    train_text = "Hello, How are you?"
    val_text = "The Weather today is "
    x_train, y_train = make_next_batch(train_text, tokenizer, block_size)
    x_val, y_val = make_next_batch(val_text, tokenizer, block_size)

    log_file = output_dir / "train_log.jsonl"
    with log_file.open("w", encoding="utf-8") as log:
        for step in range(steps):
            train_loss = train_step(model, optimizer, x_train, y_train)
            record = {"step": step, "train_loss": train_loss.item()}
            log.write(json.dumps(record) + "\n")

    val_loss = validation_step(model, x_val, y_val)
    with log_file.open("a", encoding="utf-8") as log:
        record = {"step": steps, "val_loss": val_loss.item()}
        log.write(json.dumps(record) + "\n")

    prompt = "My name is"
    prompt_token_id = torch.tensor([tokenizer.encode(prompt)])
    generated_ids = generate(model, prompt_token_id, max_tokens=20)[0].tolist()
    generated_text = decode_generated_bytes(generated_ids)

    generated_file = output_dir / "generated.txt"
    generated_file.write_text(generated_text, encoding="utf-8")

    print(f"Wrote log to {log_file}")
    print(f"Wrote generated text to {generated_file}")


if __name__ == "__main__":
    main()

