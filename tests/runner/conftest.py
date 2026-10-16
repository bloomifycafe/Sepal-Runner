"""Shared pieces of the runner front-end tests: a tiny byte-level BPE tokenizer with the Qwen ChatML specials (no download, no torch)."""
from __future__ import annotations

import pytest

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CORPUS = [
    "Read the complete state and schema. Decide every field jointly.",
    "STATE: SCHEMA FIELDS: FIELD ID: TYPE: INSTRUCTION: ALLOWED OPTIONS: OPTION END FIELD JOINT SCHEMA DECISIONS:",
    "user=admin host=srv ip=10.0.0.1 action=login result=ok failed success true false yes no benign malicious",
    '{"description":"first","option_id":"a"} {"option_id":"b"} The proposition is true or the answer is yes.',
    "system user assistant event process command line parent child",
] * 4


@pytest.fixture(scope="session")
def tiny_tokenizer():
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=330, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer)
    return tok
