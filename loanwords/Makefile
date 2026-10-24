PY ?= python
RECOGNIZER ?= whisper
CONDITIONS ?= none N0 H1 H2 L1 L2 R1 R2

.PHONY: data labels decode score analyze report pipeline test

data:
	$(PY) scripts/prepare_data.py

labels:
	$(PY) scripts/build_labels.py
	$(PY) scripts/count_tokens.py --recognizer $(RECOGNIZER)

decode:
	$(PY) scripts/decode.py --recognizer $(RECOGNIZER) --condition $(CONDITIONS)

score:
	$(PY) scripts/score.py --recognizer $(RECOGNIZER) --condition $(CONDITIONS)

analyze:
	$(PY) scripts/analyze.py --recognizer $(RECOGNIZER)

report:
	$(PY) scripts/report.py

pipeline: data labels decode score analyze report

test:
	$(PY) -m pytest -q tests -m "not slow and not network and not gpu"
