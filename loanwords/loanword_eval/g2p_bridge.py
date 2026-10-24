"""Runs inside the g2pk2 environment: reads a JSON list of strings on stdin, prints the JSON list of g2pk2 outputs."""

import contextlib
import json
import sys


def main():
    words = json.loads(sys.stdin.read() or "[]")
    with contextlib.redirect_stdout(sys.stderr):  # g2pk2 prints to stdout on import
        import nltk

        try:
            nltk.data.find("corpora/cmudict")
        except LookupError:
            nltk.download("cmudict", quiet=True)
        from g2pk2 import G2p

        g = G2p()
        out = [g(w) for w in words]
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
