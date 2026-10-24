# Synthetic fixture

Ten MathSpeech-style items (`clips.jsonl`: spoken form, LaTeX, a notation surface and a wrong hypothesis for the test decoder), three
in-discourse clips (`discourse.jsonl`) and a lookup table for the test converter (`converter_table.json`, with one deliberate error).
`python -m math_eval.fixture build DIR` writes matching sine-wave audio, one 60-s lecture per video, the benchmark item table and the
in-discourse clip labels.
