from pathlib import Path

from llm_lab.data.token_shards import (
    build_dataset_metadata,
    save_dataset_metadata,
    save_token_shards,
)
from llm_lab.tokenizer.bpe_tokenizer import TrainableBPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "alice_in_wonderland_body.txt"
TOKENIZER_PATH = PROJECT_ROOT / "outputs" / "tokenizers" / "phase_07_tokenizer.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "datasets" / "phase_08_alice"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

SHARD_SIZE = 10_000


def main() -> None:
    text = CORPUS_PATH.read_text(encoding="utf-8")

    tokenizer = TrainableBPETokenizer.load(TOKENIZER_PATH)
    token_ids = tokenizer.encode(text)

    shard_paths = save_token_shards(
        token_ids,
        output_dir=OUTPUT_DIR,
        shard_size=SHARD_SIZE,
    )

    metadata = build_dataset_metadata(
        corpus_path=CORPUS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        shard_paths=shard_paths,
        token_count=len(token_ids),
        shard_size=SHARD_SIZE,
    )

    metadata_path = save_dataset_metadata(
        metadata,
        output_path=METADATA_PATH,
    )

    print(f"Token count: {len(token_ids)}")
    print(f"Shard count: {len(shard_paths)}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
