import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: str | Path) -> str:
    file_bytes = Path(path).read_bytes()
    return hashlib.sha256(file_bytes).hexdigest()


def build_dataset_metadata(
    corpus_path: str | Path,
    tokenizer_path: str | Path,
    shard_paths: list[str | Path],
    token_count: int,
    shard_size: int,
) -> dict[str, object]:
    shard_reports = []

    for shard_path in shard_paths:
        path = Path(shard_path)
        tokens = torch.load(path, weights_only=True)

        shard_reports.append(
            {
                "filename": path.name,
                "token_count": len(tokens),
            }
        )

    return {
        "version": 1,
        "corpus_sha256": file_sha256(corpus_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "token_count": token_count,
        "shard_size": shard_size,
        "shards": shard_reports,
    }


def save_dataset_metadata(
    metadata: dict[str, object],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            sort_keys=True,
        )

    return path


def split_into_shards(
    token_ids: list[int],
    shard_size: int,
) -> list[list[int]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    return [token_ids[i : i + shard_size] for i in range(0, len(token_ids), shard_size)]


def save_token_shards(
    token_ids: list[int],
    output_dir: str | Path,
    shard_size: int,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = split_into_shards(token_ids, shard_size)
    shard_paths = []
    for shard_index, shard in enumerate(shards):
        shard = torch.tensor(shard, dtype=torch.long)
        shard_path = output_dir / f"shard_{shard_index:03d}.pt"
        torch.save(shard, shard_path)
        shard_paths.append(shard_path)
    return shard_paths


class TokenShardLoader:
    def __init__(
        self,
        shard_paths: list[str | Path],
        batch_size: int,
        block_size: int,
    ) -> None:
        self.shard_paths = [Path(path) for path in shard_paths]

        if not self.shard_paths:
            raise ValueError("shard_paths must contain at least one shard")

        self.batch_size = batch_size
        self.block_size = block_size

        self.shard_index = 0
        self.position = 0
        self.tokens = self._load_shard(0)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        tokens_per_batch = self.batch_size * self.block_size
        tokens_needed = tokens_per_batch + 1

        if self.position + tokens_needed > len(self.tokens):
            self.shard_index = (self.shard_index + 1) % len(self.shard_paths)
            self.position = 0
            self.tokens = self._load_shard(self.shard_index)

        buffer = self.tokens[self.position : self.position + tokens_needed]

        x = buffer[:-1]
        y = buffer[1:]

        x = x.view(self.batch_size, self.block_size)
        y = y.view(self.batch_size, self.block_size)

        self.position += tokens_per_batch

        return x, y

    def state_dict(self) -> dict[str, int]:
        return {
            "shard_index": self.shard_index,
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.shard_index = state["shard_index"]
        self.position = state["position"]

        self.tokens = self._load_shard(self.shard_index)

    def _load_shard(self, shard_index: int) -> torch.Tensor:
        shard_path = self.shard_paths[shard_index]
        tokens = torch.load(
            shard_path,
            weights_only=True,
        )

        tokens_needed = self.batch_size * self.block_size + 1

        if len(tokens) < tokens_needed:
            raise ValueError(f"shard must contain at least {tokens_needed} tokens")

        return tokens
