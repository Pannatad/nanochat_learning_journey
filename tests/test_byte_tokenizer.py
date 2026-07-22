from llm_lab.tokenizer.byte_tokenizer import ByteTokenizer


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "I am a dog"
    token_ids = tokenizer.encode(text)
    assert text == tokenizer.decode(token_ids)
