from pathlib import Path

from llm_lab.tokenizer.bpe_tokenizer import TrainableBPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data/processed/alice_in_wonderland_body.txt"
OUTPUT_PATH = PROJECT_ROOT / "outputs/tokenizers/phase_07_tokenizer.json"
VOCAB_SIZE = 512


def main() -> None:
    training_text = CORPUS_PATH.read_text(encoding="utf-8")

    tokenizer = TrainableBPETokenizer()
    tokenizer.train(training_text, vocab_size=VOCAB_SIZE)

    stats = tokenizer.compression_stats(training_text)
    merge_reports = tokenizer.inspect_merges()

    print(f"Corpus: {CORPUS_PATH}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Byte count: {stats['byte_count']}")
    print(f"Token count: {stats['token_count']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}")

    print("\nFirst 50 learned merges:")
    for report in merge_reports[:50]:
        print(report)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUTPUT_PATH)
    print(f"\nSaved tokenizer to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
