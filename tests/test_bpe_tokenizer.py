from llm_lab.tokenizer.bpe_tokenizer import (
    TrainableBPETokenizer,
    count_pairs,
    count_pairs_in_chunks,
    find_best_pair,
    merge_pair,
    merge_pair_in_chunks,
    split_text,
    text_to_byte_chunks,
    train_merges,
    train_merges_in_chunks,
    train_one_merge,
)


def test_split_text_separates_words_numbers_whitespace_and_punctuation():
    assert split_text("hello 123!") == ["hello", " ", "123", "!"]


def test_split_text_supports_unicode_letters():
    assert split_text("สวัสดี 42!") == ["สวัสดี", " ", "42", "!"]


def test_text_to_byte_chunks_preserves_split_boundaries():
    assert text_to_byte_chunks("hi 42!") == [
        [104, 105],
        [32],
        [52, 50],
        [33],
    ]


def test_count_pairs_in_chunks_combines_counts_without_crossing_boundaries():
    token_chunks = [
        [1, 2, 1, 2],
        [1, 2, 3],
    ]

    assert count_pairs_in_chunks(token_chunks) == {
        (1, 2): 3,
        (2, 1): 1,
        (2, 3): 1,
    }


def test_merge_pair_in_chunks_preserves_chunk_boundaries():
    assert merge_pair_in_chunks(
        [[1, 2, 1, 2], [1, 2, 3]],
        pair_to_merge=(1, 2),
        new_token_id=99,
    ) == [
        [99, 99],
        [99, 3],
    ]


def test_train_merges_in_chunks_repeats_bpe_steps():
    final_chunks, merges = train_merges_in_chunks(
        [[1, 2, 1, 2], [1, 2, 3]],
        num_merges=2,
        first_new_token_id=99,
    )

    assert final_chunks == [[100], [99, 3]]
    assert merges == [
        ((1, 2), 99),
        ((99, 99), 100),
    ]


def test_train_merges_in_chunks_stops_when_no_pairs_remain():
    final_chunks, merges = train_merges_in_chunks(
        [[1], [2]],
        num_merges=2,
        first_new_token_id=99,
    )

    assert final_chunks == [[1], [2]]
    assert merges == []


def test_count_pairs_counts_adjacent_pairs():
    token_ids = [1, 2, 1, 2, 3]

    pair_counts = count_pairs(token_ids)

    assert pair_counts == {
        (1, 2): 2,
        (2, 1): 1,
        (2, 3): 1,
    }


def test_count_pairs_returns_empty_dict_for_too_few_tokens():
    assert count_pairs([]) == {}
    assert count_pairs([1]) == {}


def test_merge_pair_replaces_matching_adjacent_pairs():
    assert merge_pair([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3]


def test_merge_pair_keeps_non_matching_tokens():
    assert merge_pair([1, 3, 2], (1, 2), 99) == [1, 3, 2]


def test_find_best_pair_returns_most_common_pair():
    pair_counts = {
        (1, 2): 2,
        (2, 1): 1,
    }

    assert find_best_pair(pair_counts) == (1, 2)


def test_find_best_pair_returns_none_for_empty_counts():
    assert find_best_pair({}) is None


def test_train_one_merge_merges_best_pair():
    merged, pair = train_one_merge([1, 2, 1, 2, 3], 99)

    assert pair == (1, 2)
    assert merged == [99, 99, 3]


def test_train_one_merge_returns_original_tokens_when_no_pair_exists():
    merged, pair = train_one_merge([1], 99)

    assert pair is None
    assert merged == [1]


def test_train_merges_repeats_bpe_steps():
    final_tokens, merges = train_merges(
        [1, 2, 1, 2, 3],
        num_merges=2,
        first_new_token_id=99,
    )

    assert final_tokens == [100, 3]
    assert merges == [
        ((1, 2), 99),
        ((99, 99), 100),
    ]


def test_train_merges_stops_when_no_pairs_remain():
    final_tokens, merges = train_merges(
        [1],
        num_merges=2,
        first_new_token_id=99,
    )

    assert final_tokens == [1]
    assert merges == []


def test_trainable_bpe_tokenizer_stores_merges_after_training():
    tokenizer = TrainableBPETokenizer()

    tokenizer.train("abab", vocab_size=257)

    assert tokenizer.vocab_size == 257
    assert tokenizer.merges == [((97, 98), 256)]


def test_trainable_bpe_tokenizer_encode_applies_learned_merges():
    tokenizer = TrainableBPETokenizer()
    tokenizer.train("abab", vocab_size=257)

    assert tokenizer.encode("abab") == [256, 256]


def test_trainable_bpe_tokenizer_decodes_merged_tokens():
    tokenizer = TrainableBPETokenizer()
    tokenizer.train("abab", vocab_size=257)

    assert tokenizer.decode([256, 256]) == "abab"


def test_trainable_bpe_tokenizer_round_trip_preserves_unicode_text():
    tokenizer = TrainableBPETokenizer()
    tokenizer.train("hello สวัสดี", vocab_size=265)

    text = "hello สวัสดี"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_trainable_bpe_tokenizer_save_and_load(tmp_path):
    tokenizer = TrainableBPETokenizer()
    tokenizer.train("abab", vocab_size=257)

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = TrainableBPETokenizer.load(path)

    assert loaded.merges == tokenizer.merges
    assert loaded.encode("abab") == [256, 256]
    assert loaded.decode([256, 256]) == "abab"


def test_trainable_bpe_tokenizer_reports_compression_stats():
    tokenizer = TrainableBPETokenizer()
    tokenizer.train("abab", vocab_size=257)

    assert tokenizer.compression_stats("abab") == {
        "byte_count": 4,
        "token_count": 2,
        "compression_ratio": 2.0,
    }


def test_trainable_bpe_tokenizer_reports_zero_compression_for_empty_text():
    tokenizer = TrainableBPETokenizer()

    assert tokenizer.compression_stats("") == {
        "byte_count": 0,
        "token_count": 0,
        "compression_ratio": 0.0,
    }


def test_trainable_bpe_tokenizer_does_not_learn_across_split_boundaries():
    tokenizer = TrainableBPETokenizer()
    tokenizer.train("a a", vocab_size=257)

    assert tokenizer.merges == []
    assert tokenizer.encode("a a") == [97, 32, 97]


def test_trainable_bpe_tokenizer_inspects_learned_merges():
    tokenizer = TrainableBPETokenizer()
    tokenizer.train("abab", vocab_size=257)

    assert tokenizer.inspect_merges() == [
        {
            "pair": [97, 98],
            "new_token_id": 256,
            "token_bytes": [97, 98],
            "token_text": "ab",
        }
    ]
