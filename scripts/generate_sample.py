import torch

from llm_lab.model.tiny_lm import TinyLanguageModel
from llm_lab.tokenizer.byte_tokenizer import ByteTokenizer
from llm_lab.training.loop import generate


def decode_generated_bytes(token_ids: list[int]) -> str:
    return bytes(token_ids).decode("utf-8", errors="replace")


def main() -> None:
    tokenizer = ByteTokenizer()
    model = TinyLanguageModel(vocab_size=tokenizer.vocab_size, d_model=16)

    prompt = "hello"
    prompt_ids = tokenizer.encode(prompt)
    start_token_ids = torch.tensor([prompt_ids])

    generated = generate(model, start_token_ids, max_tokens=20)
    generated_ids = generated[0].tolist()

    print(f"Prompt: {prompt!r}")
    print(f"Generated token IDs: {generated_ids}")
    print(f"Generated text: {decode_generated_bytes(generated_ids)!r}")
    print("\nNote: this model is untrained, so the generated text is expected to look random.")


if __name__ == "__main__":
    main()
