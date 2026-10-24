# FORM: Form-tolerant Outcomes of Recognition for Multi-form Items

Code and labels for the M.S. thesis *FORM: Form-tolerant Outcomes of Recognition for Multi-form Items* (Olivia S. Kim, Emory
University, 2026).

A spoken item can have more than one correct written form: a Korean loanword in Hangul or in Latin script, a mathematical expression
in words or in notation. A single-reference error rate counts a recognized item written in the other form as an error. FORM scores
each such item twice: whether it was recognized in any accepted form, and which form was produced. The repository applies the measure
to Korean loanwords on HiKE (`loanwords/`) and to spoken mathematics on MathSpeech and an in-discourse set built from its source
lectures (`math/`), and tests whether matched preceding-text prompts steer the written form.

## Layout

```
loanwords/      Korean loanwords on HiKE
  loanword_eval/  items, acceptance sets, scorer, recognizer wrappers, statistics
  scripts/        prepare_data, build_labels, count_tokens, decode, score, dominant_forms, analyze, report
  configs/        one decoding config per recognizer
  labels/         acceptance sets, comparison words, dominant forms, prompt texts
  envs/           package lists of the recognizer environments
math/           spoken mathematics on MathSpeech and the in-discourse set
  math_eval/      canonical-key grammar, scorer, converter, in-discourse conditions, statistics
  scripts/        prepare_data, build_labels, decode, score, dominant_forms, analyze, convert, attribute, report
  configs/, labels/
```

## Setup

Python 3.12 was used for all reported runs. Analysis runs on CPU; decoding needs a GPU (one 80 GB GPU runs any recognizer).

```bash
pip install -e "loanwords[dev]" -e "math[dev]"
python -m spacy download en_core_web_sm
```

LaTeX normalization in `math/` uses `process_formula`, which needs Node.js. Korean grapheme-to-phoneme conversion (g2pk2) runs in a
separate environment because its MeCab dictionary conflicts with wordfreq's:

```bash
python -m venv .venv-g2p && .venv-g2p/bin/pip install -r loanwords/envs/g2p_freeze.txt
export LOANWORDS_G2P_PYTHON=$PWD/.venv-g2p/bin/python
```

Whisper, Cohere Transcribe and Voxtral run in the main environment (`pip install -e "loanwords[gpu]"`). The other recognizers pin
conflicting framework versions and need their own environments: `loanwords/envs/qwen3asr_raon.txt` (Qwen3-ASR, Raon-Speech),
`owsm.txt` (OWSM v4) and `omnilingual.txt` (Omnilingual ASR). `analysis_freeze.txt` lists the full environment of the reported runs.
Gated checkpoints read `HF_TOKEN` from the environment.

## Data

| Data | Source | Used by |
|---|---|---|
| HiKE, 1,121 utterances | Hugging Face `thetaone-ai/HiKE` (Apache-2.0), downloaded by `prepare_data.py` | `loanwords/` |
| MathSpeech, 1,101 clips | Hugging Face `AAAI2025/MathSpeech`, downloaded by `prepare_data.py` | `math/` |
| MIT OpenCourseWare lectures | the 69 lectures marked in `math/labels/SOURCES.md` (CC BY-NC-SA 4.0) | `math/`, in-discourse set |

No audio is distributed here. For the in-discourse set, download the lecture videos from MIT OpenCourseWare, convert each to a 16 kHz
mono WAV named `<video_id>.wav`, and pass the folder as `--lectures-dir`. Intermediate files go to `work/` and outputs to
`outputs/` inside each project (`LOANWORDS_WORK`, `MATH_WORK` and the matching `*_DATA` variables move them).

Released datasets on Hugging Face:
[`hoi-is-repeating/form-hike-labels`](https://huggingface.co/datasets/hoi-is-repeating/form-hike-labels),
[`hoi-is-repeating/form-indiscourse-math`](https://huggingface.co/datasets/hoi-is-repeating/form-indiscourse-math) and
[`hoi-is-repeating/form-tts-loanword-pairs`](https://huggingface.co/datasets/hoi-is-repeating/form-tts-loanword-pairs)
(the synthetic loanword / native pairs with audio).

## Running

Each project runs as a pipeline of scripts; `make pipeline` runs all of them for one recognizer.

```bash
cd loanwords
make data labels                      # HiKE items, acceptance sets, comparison words, covariates
make decode score RECOGNIZER=whisper  # transcribe under the no-prompt and prompted conditions, then score
make analyze report                   # base rate (H1a, H1b) and prompt steering (H2a, H2b)
```

```bash
cd math
make data labels LECTURES=/path/to/lectures
make decode score analyze RECOGNIZER=whisper
make convert attribute report         # conversion to LaTeX and the attribution of its errors
```

Recognizers: `whisper`, `owsm`, `qwen3asr`, `raon`, `cohere`, `voxtral`, `omni` (see `configs/`). Statistics use a cluster bootstrap
over utterances (5,000 resamples) and sign-flip permutation tests (10,000 permutations) with a fixed seed, so a rerun on the same
transcripts reproduces every interval and p value.

The default speech-to-LaTeX converter is a pair of LoRA adapters on Qwen2.5-Math-1.5B-Instruct that are not included; point
`CH4_CKPT` at them, or run `convert.py` and `attribute.py` with `--converter s2l_equations`, the released Speech-to-LaTeX checkpoint.

`math/labels/accept_sets.v1.jsonl` ships without the fields derived from MathSpeech's transcripts. `build_labels.py` recomputes
them from MathSpeech; `python math/scripts/rebuild_labels.py` restores the full file and checks it against the recorded sha256.

## Tests

```bash
make -C loanwords test && make -C math test
```

The tests run every script end to end on a synthetic fixture with a test decoder; they need no GPU or corpus, only the small
`openai/whisper-tiny` tokenizer from the Hugging Face Hub.

## License

Code: MIT (`LICENSE`). Labels carry the license of the material they are derived from; see `DATA_LICENSES.md`.

## Citation

```bibtex
@mastersthesis{kim2026form,
  author = {Kim, Olivia S.},
  title  = {{FORM}: Form-tolerant Outcomes of Recognition for Multi-form Items},
  school = {Emory University},
  year   = {2026}
}
```
