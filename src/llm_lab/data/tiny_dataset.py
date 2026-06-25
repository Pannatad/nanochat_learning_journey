def make_next_token_example(token_ids: list[int], block_size: int) -> tuple[list[int], list[int]]:
    x = token_ids[:block_size]
    y = token_ids[1 : block_size + 1]
    return x, y
