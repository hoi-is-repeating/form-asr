# Data licenses

The code in this repository is released under the MIT License (`LICENSE`). The label files carry the license of the material they
are derived from. No audio is included.

| Path | License | Derived from |
|---|---|---|
| `loanwords/labels/` | Apache License 2.0 (`loanwords/labels/LICENSE`) | HiKE (Paik et al., 2026; `thetaone-ai/HiKE`, Apache-2.0) |
| `math/labels/` | CC BY-NC-SA 4.0 | MIT OpenCourseWare lectures and their English captions (CC BY-NC-SA 4.0); MathSpeech (Hyeon et al., 2025) by item identifier |
| `loanwords/configs/`, `math/configs/` | MIT | written for this work |

## loanwords/labels

Changes from HiKE: no HiKE file is included in its original form. The files hold labels derived from HiKE's text: character offsets
into its `text` field, the item words and comparison words at those offsets, accepted spellings, covariates, the written form each
recognizer produces more often, and the prompt texts written for this work. Please cite HiKE as well.

Accepted spellings marked `g2p` were generated with g2pk2 0.0.3 (Apache-2.0). Zipf frequencies are from wordfreq 3.1.1 (Speer; data
CC BY-SA 4.0).

## math/labels

Lectures: MIT OpenCourseWare course videos by MIT faculty, https://ocw.mit.edu, licensed under Creative Commons BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/). `math/labels/SOURCES.md` lists every lecture referenced, with its title and
link. Changes: the caption text in `discourse_clips.v1.jsonl` and the comparison words in `controls.v1.jsonl` were excerpted around
each expression and aligned to it in time. These files are shared under the same license, for non-commercial use. MIT
OpenCourseWare does not endorse this work.

MathSpeech's release states no license, so its transcripts and LaTeX are not redistributed, nor anything computed from them beyond
categories and counts. `accept_sets.v1.jsonl` ships without the transcript and the canonical keys; `build_labels.py` recomputes them
from MathSpeech's Hugging Face release, and `math/scripts/rebuild_labels.py` restores the full file and checks it against
`accept_sets.v1.jsonl.original.sha256`. The caption text can contain the lecturers' wording of other benchmark expressions, as
captioned by MIT OpenCourseWare, and the tests and grammar examples use a few short standard expressions that also occur among
MathSpeech's items (for example "two over the square root of pi").

Comparison-word frequencies are from wordfreq 3.1.1 (Speer; data CC BY-SA 4.0).

## Third-party code

None is included. Recognizers, tokenizers and the Speech-to-LaTeX comparator are loaded from their own releases at the revisions
pinned in `configs/`, under their own licenses.
