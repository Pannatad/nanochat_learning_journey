import json
from pathlib import Path

import pytest
import torch

from llm_lab.data.token_shards import (
    TokenShardLoader,
    build_dataset_metadata,
    file_sha256,
    save_dataset_metadata,
    save_token_shards,
    split_into_shards,
)


def test_split_into_shards_keeps_smaller_final_shard():
    token_ids = [10, 20, 30, 40, 50, 60, 70]

    shards = split_into_shards(token_ids, shard_size=3)

    assert shards == [[10, 20, 30], [40, 50, 60], [70]]


def test_split_into_shards_returns_empty_list_for_empty_input():
    shards = split_into_shards([], shard_size=3)

    assert shards == []


@pytest.mark.parametrize("shard_size", [0, -1])
def test_split_into_shards_rejects_non_positive_shard_size(shard_size):
    with pytest.raises(ValueError):
        split_into_shards([10, 20], shard_size=shard_size)


def test_save_token_shards_writes_numbered_tensor_files(tmp_path: Path):
    token_ids = [10, 20, 30, 40, 50]

    paths = save_token_shards(
        token_ids,
        output_dir=tmp_path,
        shard_size=2,
    )

    assert [path.name for path in paths] == [
        "shard_000.pt",
        "shard_001.pt",
        "shard_002.pt",
    ]

    first_shard = torch.load(paths[0], weights_only=True)

    assert first_shard.dtype == torch.long
    assert first_shard.tolist() == [10, 20]


def test_token_shard_loader_builds_shifted_batch(tmp_path: Path):
    paths = save_token_shards(
        [10, 20, 30, 40, 50, 60, 70],
        output_dir=tmp_path,
        shard_size=7,
    )

    loader = TokenShardLoader(
        shard_paths=paths,
        batch_size=2,
        block_size=3,
    )

    x, y = loader.next_batch()

    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    assert x.tolist() == [[10, 20, 30], [40, 50, 60]]
    assert y.tolist() == [[20, 30, 40], [50, 60, 70]]


def test_token_shard_loader_advances_to_next_batch(tmp_path: Path):
    paths = save_token_shards(
        [10, 20, 30, 40, 50, 60, 70],
        output_dir=tmp_path,
        shard_size=7,
    )
    loader = TokenShardLoader(paths, batch_size=1, block_size=3)

    loader.next_batch()
    second_x, second_y = loader.next_batch()

    assert second_x.tolist() == [[40, 50, 60]]
    assert second_y.tolist() == [[50, 60, 70]]


def test_token_shard_loader_moves_to_next_shard(tmp_path: Path):
    paths = save_token_shards(
        [10, 20, 30, 40, 50, 60, 70, 80],
        output_dir=tmp_path,
        shard_size=4,
    )
    loader = TokenShardLoader(paths, batch_size=1, block_size=3)

    loader.next_batch()
    second_x, second_y = loader.next_batch()

    assert second_x.tolist() == [[50, 60, 70]]
    assert second_y.tolist() == [[60, 70, 80]]
    assert loader.shard_index == 1
    assert loader.position == 3


def test_token_shard_loader_restores_next_batch(tmp_path: Path):
    paths = save_token_shards(
        [10, 20, 30, 40, 50, 60, 70, 80],
        output_dir=tmp_path,
        shard_size=4,
    )

    original = TokenShardLoader(
        paths,
        batch_size=1,
        block_size=3,
    )
    original.next_batch()
    saved_state = original.state_dict()

    restored = TokenShardLoader(
        paths,
        batch_size=1,
        block_size=3,
    )
    restored.load_state_dict(saved_state)

    expected_x, expected_y = original.next_batch()
    actual_x, actual_y = restored.next_batch()

    assert torch.equal(actual_x, expected_x)
    assert torch.equal(actual_y, expected_y)


def test_token_shard_loader_rejects_empty_shard_paths():
    with pytest.raises(ValueError, match="at least one shard"):
        TokenShardLoader(
            shard_paths=[],
            batch_size=1,
            block_size=3,
        )


def test_token_shard_loader_rejects_undersized_shard(tmp_path: Path):
    paths = save_token_shards(
        [10, 20, 30],
        output_dir=tmp_path,
        shard_size=3,
    )

    with pytest.raises(ValueError, match="shard must contain at least 4 tokens"):
        TokenShardLoader(
            shard_paths=paths,
            batch_size=1,
            block_size=3,
        )


def test_file_sha256_changes_when_file_contents_change(tmp_path: Path):
    path = tmp_path / "corpus.txt"

    path.write_text("first", encoding="utf-8")
    first_hash = file_sha256(path)

    path.write_text("second", encoding="utf-8")
    second_hash = file_sha256(path)

    assert len(first_hash) == 64
    assert first_hash != second_hash


def test_build_dataset_metadata_records_identity_and_shards(tmp_path: Path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"

    corpus_path.write_text("hello", encoding="utf-8")
    tokenizer_path.write_text("{}", encoding="utf-8")

    shard_paths = save_token_shards(
        [10, 20, 30, 40, 50],
        output_dir=tmp_path / "shards",
        shard_size=2,
    )

    metadata = build_dataset_metadata(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        shard_paths=shard_paths,
        token_count=5,
        shard_size=2,
    )

    assert metadata["version"] == 1
    assert metadata["token_count"] == 5
    assert metadata["shards"] == [
        {"filename": "shard_000.pt", "token_count": 2},
        {"filename": "shard_001.pt", "token_count": 2},
        {"filename": "shard_002.pt", "token_count": 1},
    ]


def test_save_dataset_metadata_writes_json_file(tmp_path: Path):
    metadata = {
        "version": 1,
        "token_count": 5,
        "shards": [],
    }
    output_path = tmp_path / "dataset" / "metadata.json"

    saved_path = save_dataset_metadata(
        metadata,
        output_path=output_path,
    )

    loaded = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path == output_path
    assert loaded == metadata


def test_token_shard_loader_reproduces_batches_from_same_start(tmp_path: Path):
    paths = save_token_shards(
        list(range(16)),
        output_dir=tmp_path,
        shard_size=8,
    )

    first_loader = TokenShardLoader(
        paths,
        batch_size=1,
        block_size=3,
    )
    second_loader = TokenShardLoader(
        paths,
        batch_size=1,
        block_size=3,
    )

    first_batches = [first_loader.next_batch() for _ in range(4)]
    second_batches = [second_loader.next_batch() for _ in range(4)]

    for (first_x, first_y), (second_x, second_y) in zip(
        first_batches,
        second_batches,
    ):
        assert torch.equal(first_x, second_x)
        assert torch.equal(first_y, second_y)
