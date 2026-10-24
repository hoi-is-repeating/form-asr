"""g2pk2 access: in-process when importable, otherwise through a subprocess in a separate environment (`.venv-g2p` or
$LOANWORDS_G2P_PYTHON). The separate environment avoids MeCab package collisions with wordfreq[cjk]; the same g2pk2 0.0.3
is used either way and the backend is recorded in result headers."""

from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable

from loanword_eval.paths import ROOT

BRIDGE = ROOT / "loanword_eval" / "g2p_bridge.py"


def _candidate_pythons() -> list[str]:
    c = [os.environ.get("LOANWORDS_G2P_PYTHON"), str(ROOT / ".venv-g2p" / "bin" / "python")]
    return [x for x in c if x and Path(x).exists()]


class SubprocessG2p:
    backend = "subprocess:g2pk2"

    def __init__(self, python: str):
        self.python = python
        self._cache: dict[str, str] = {}

    def batch(self, words: list[str]) -> list[str]:
        todo = [w for w in dict.fromkeys(words) if w not in self._cache]
        if todo:
            r = subprocess.run(
                [self.python, str(BRIDGE)], input=json.dumps(todo, ensure_ascii=False), capture_output=True, text=True
            )
            if r.returncode != 0:
                raise RuntimeError(f"g2p bridge failed: {r.stderr[-800:]}")
            for w, out in zip(todo, json.loads(r.stdout)):
                self._cache[w] = out
        return [self._cache[w] for w in words]

    def __call__(self, word: str) -> str:
        return self.batch([word])[0]


@lru_cache(maxsize=1)
def load_g2p() -> Callable[[str], str] | None:
    try:
        from g2pk2 import G2p  # type: ignore

        g = G2p()
        g("테스트")
        f = lambda s: g(s)  # noqa: E731
        f.backend = "inprocess:g2pk2"  # type: ignore[attr-defined]
        return f
    except Exception:
        pass
    for py in _candidate_pythons():
        try:
            g = SubprocessG2p(py)
            g("테스트")
            return g
        except Exception:
            continue
    return None
