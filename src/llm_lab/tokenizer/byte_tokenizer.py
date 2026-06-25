class ByteTokenizer:
    vocab_size = 256
    
    def encode(self, text: str) -> list[int]:
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        return tokens
    
    def decode(self, token_ids: list[int]) -> str:
        text = bytes(token_ids)
        text = text.decode("utf-8")
        return text


