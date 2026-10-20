"""Shared GPT-2 token mapping used by corpus preparation and inference."""
import codecs

import tiktoken


VOCAB_SIZE = 50257
ENCODING = tiktoken.get_encoding("gpt2")
if ENCODING.n_vocab != VOCAB_SIZE:
    raise RuntimeError(f"GPT-2 tokenizer vocab mismatch: expected {VOCAB_SIZE}, got {ENCODING.n_vocab}")
EOT = ENCODING.eot_token


def encode_text(text):
    """Encode literal text without interpreting user text as a special token."""
    return ENCODING.encode_ordinary(text)


class StreamingDecoder:
    def __init__(self, encoding=ENCODING):
        self.encoding = encoding
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id):
        return self.decoder.decode(self.encoding.decode_single_token_bytes(int(token_id)))

    def finish(self):
        return self.decoder.decode(b"", final=True)
