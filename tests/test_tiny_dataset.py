from llm_lab.data.tiny_dataset import make_next_token_example


def test_make_next_token_example():
    tokens = [1, 2, 3, 4]
    x = [1, 2, 3]
    y = [2, 3, 4]
    assert (x, y) == make_next_token_example(tokens, 3)
