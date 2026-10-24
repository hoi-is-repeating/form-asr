PY ?= python
RECOGNIZER ?= whisper
LECTURES ?=
BENCHMARK ?= none N0 W1 S1 W2 S2
DISCOURSE ?= D0 D1 D2

.PHONY: data labels decode score analyze convert attribute report pipeline test

data:
	$(PY) scripts/prepare_data.py $(if $(LECTURES),--lectures-dir $(LECTURES))

labels:
	$(PY) scripts/build_labels.py

decode:
	$(PY) scripts/decode.py --recognizer $(RECOGNIZER) --condition $(BENCHMARK)
	$(PY) scripts/decode.py --recognizer $(RECOGNIZER) --condition $(DISCOURSE)

score:
	$(PY) scripts/score.py --recognizer $(RECOGNIZER) --condition $(BENCHMARK) $(DISCOURSE)

analyze:
	$(PY) scripts/analyze.py --recognizer $(RECOGNIZER)

convert:
	$(PY) scripts/convert.py --reference
	$(PY) scripts/convert.py --recognizer $(RECOGNIZER) --condition D0
	$(PY) scripts/convert.py --recognizer $(RECOGNIZER) --condition D2

attribute:
	$(PY) scripts/attribute.py --recognizer $(RECOGNIZER)

report:
	$(PY) scripts/report.py

pipeline: data labels decode score analyze convert attribute report

test:
	$(PY) -m pytest -q tests -m "not slow and not network and not gpu"
