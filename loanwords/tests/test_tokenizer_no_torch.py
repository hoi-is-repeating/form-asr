"""Covariate n_bpe and prompt token counts need only the Whisper tokenizer (tokenizer-only path, no torch import)."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

CODE = """
import sys
sys.modules["torch"] = None           # any attempt to import torch on this path raises ImportError
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny")   # hub-cached; multilingual vocabulary shared by all sizes
ids = tok.encode("오늘 버스를 탔어요", add_special_tokens=False)
assert len(ids) > 3
prompt_ids = tok.get_prompt_ids("택시 커피")
assert prompt_ids[0] == tok.convert_tokens_to_ids("<|startofprev|>")
print("tokenizer_only_ok")
"""


def test_whisper_tokenizer_loads_and_counts_without_torch():
    """Fresh interpreter with torch import blocked: the tokenizer-only covariate path must not need torch."""
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert "tokenizer_only_ok" in r.stdout, r.stdout
