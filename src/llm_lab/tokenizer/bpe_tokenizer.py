import json
from pathlib import Path

import regex


SPLIT_PATTERN = r"(?:\p{L}\p{M}*)+|\p{N}+|\s+|[^\s\p{L}\p{N}]+"


def split_text(text: str) -> list[str]:
    return regex.findall(SPLIT_PATTERN, text)


def text_to_byte_chunks(text: str) -> list[list[int]]:
    chunks = split_text(text)
    return [list(chunk.encode("utf-8")) for chunk in chunks]


def count_pairs_in_chunks(
    token_chunks: list[list[int]],
) -> dict[tuple[int, int], int]:
    total_counts = {}

    for token_ids in token_chunks:
        chunk_counts = count_pairs(token_ids)

        for pair, count in chunk_counts.items():
            total_counts[pair] = total_counts.get(pair, 0) + count

    return total_counts


def merge_pair_in_chunks(
    token_chunks: list[list[int]],
    pair_to_merge: tuple[int, int],
    new_token_id: int,
) -> list[list[int]]:
    return [
        merge_pair(token_ids, pair_to_merge, new_token_id)
        for token_ids in token_chunks
    ]


def train_merges_in_chunks(
    token_chunks: list[list[int]],
    num_merges: int,
    first_new_token_id: int = 256,
) -> tuple[
    list[list[int]],
    list[tuple[tuple[int, int], int]],
]:
    current_chunks = token_chunks
    merges = []

    for merge_index in range(num_merges):
        pair_counts = count_pairs_in_chunks(current_chunks)
        best_pair = find_best_pair(pair_counts)

        if best_pair is None:
            break

        new_token_id = first_new_token_id + merge_index
        current_chunks = merge_pair_in_chunks(current_chunks, best_pair, new_token_id)
        merges.append((best_pair, new_token_id))

    return current_chunks, merges


def count_pairs(token_ids: list[int]) -> dict[tuple[int, int], int]:
    pair_counts = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        if pair in pair_counts:
            pair_counts[pair] += 1
        else:
            pair_counts[pair] = 1
    return pair_counts


def merge_pair(
    token_ids: list[int],
    pair_to_merge: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    output = []
    i = 0

    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair_to_merge:
            output.append(new_token_id)
            i += 2
        else:
            output.append(token_ids[i])
            i += 1

    return output


def find_best_pair(
    pair_counts: dict[tuple[int, int], int],
) -> tuple[int, int] | None:
    if not pair_counts:
        return None
    return max(pair_counts, key=pair_counts.get)


def train_one_merge(
    token_ids: list[int],
    new_token_id: int,
) -> tuple[list[int], tuple[int, int] | None]:
    pair_counts = count_pairs(token_ids)
    best_pair = find_best_pair(pair_counts)

    if best_pair is None:
        return token_ids, None

    merged_token_ids = merge_pair(token_ids, best_pair, new_token_id)
    return merged_token_ids, best_pair


def train_merges(
    token_ids: list[int],
    num_merges: int,
    first_new_token_id: int = 256,
) -> tuple[list[int], list[tuple[tuple[int, int], int]]]:
    current_token_ids = token_ids
    merges = []

    for merge_index in range(num_merges):
        new_token_id = first_new_token_id + merge_index
        merged_token_ids, best_pair = train_one_merge(current_token_ids, new_token_id)

        if best_pair is None:
            break

        merges.append((best_pair, new_token_id))
        current_token_ids = merged_token_ids

    return current_token_ids, merges


class TrainableBPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab_size = 256
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        token_chunks = text_to_byte_chunks(text)
        num_merges = vocab_size - self.vocab_size
        _, merges = train_merges_in_chunks(token_chunks, num_merges)
        self.merges = merges
        self.vocab_size = 256 + len(merges)

        for pair, new_token_id in self.merges:
            left_token_id, right_token_id = pair
            self.vocab[new_token_id] = (
                self.vocab[left_token_id] + self.vocab[right_token_id]
            )

    def encode(self, text: str) -> list[int]:
        token_chunks = text_to_byte_chunks(text)

        for pair, new_token_id in self.merges:
            token_chunks = merge_pair_in_chunks(token_chunks, pair, new_token_id)

        token_ids = [token_id for chunk in token_chunks for token_id in chunk]
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        pieces = []

        for token_id in token_ids:
            pieces.append(self.vocab[token_id])

        combined_bytes = b"".join(pieces)
        return combined_bytes.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        merge_data = []

        for pair, new_token_id in self.merges:
            left_token_id, right_token_id = pair

            merge_data.append([left_token_id, right_token_id, new_token_id])

        data = {
            "vocab_size": self.vocab_size,
            "merges": merge_data,
        }
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(data, file)

    @classmethod
    def load(cls, path: str | Path) -> "TrainableBPETokenizer":
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        tokenizer = cls()
        tokenizer.merges = []

        for left_token_id, right_token_id, new_token_id in data["merges"]:
            pair = (left_token_id, right_token_id)

            tokenizer.merges.append((pair, new_token_id))
            tokenizer.vocab[new_token_id] = (
                tokenizer.vocab[left_token_id] + tokenizer.vocab[right_token_id]
            )

        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer

    def compression_stats(self, text: str) -> dict[str, int | float]:
        byte_ids = list(text.encode("utf-8"))
        bpe_ids = self.encode(text)

        byte_count = len(byte_ids)
        token_count = len(bpe_ids)
        compression_ratio = byte_count / token_count if token_count > 0 else 0.0

        return {
            "byte_count": byte_count,
            "token_count": token_count,
            "compression_ratio": compression_ratio,
        }

    def inspect_merges(self) -> list[dict[str, object]]:
        reports = []

        for pair, new_token_id in self.merges:
            token_bytes = self.vocab[new_token_id]

            report = {
                "pair": list(pair),
                "new_token_id": new_token_id,
                "token_bytes": list(token_bytes),
                "token_text": token_bytes.decode("utf-8", errors="replace"),
            }

            reports.append(report)

        return reports
