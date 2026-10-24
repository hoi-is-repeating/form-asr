"""Spoken-form-to-LaTeX converters (configs/converter.yaml).

`load_converter(name)` returns an object with `convert(texts) -> list[str]` (raw LaTeX; normalization happens in math_eval.latex_norm).
Kinds:
  hf_lora_two_stage  base causal LM + two LoRA adapters: a corrector (transcript -> corrected spoken form), then a translator (-> LaTeX);
                     adapter paths may name environment variables (${NAME})
  hf_causal          one instruction-tuned causal LM (the Speech-to-LaTeX checkpoints), its prompt and beam settings
  fake               lookup table on the canonical key (tests only)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from math_eval.paths import ROOT, load_yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def expand(s: str) -> str:
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), s)


def _first_line(s: str) -> str:
    return (s or "").strip().split("\n")[0].strip()


def _s2l_latex_line(s: str) -> str:
    s = (s or "").split("<|im_end|>")[0].strip()
    m = re.search(r"LaTe[Xx]\s*:\s*(.*)", s, flags=re.DOTALL)
    out = m.group(1) if m else s
    return out.strip().split("\n")[0].strip()


def is_bare_latex(raw: str) -> bool:
    """True when the raw output is one line of LaTeX: no code fence, no prose line, no math delimiters around it."""
    r = (raw or "").strip()
    return (
        bool(r)
        and "\n" not in r
        and "```" not in r
        and not r.startswith(("$", "\\[", "\\("))
        and not re.search(r"\b(the|is|this|here|answer)\b", r, flags=re.I)
    )


POST = {"first_line": _first_line, "s2l_latex_line": _s2l_latex_line, "none": lambda s: (s or "").strip()}


class FakeConverter:
    def __init__(self, spec: dict):
        from math_eval.mathforms import canonical

        self._canon = canonical
        self.table = json.loads((ROOT / spec["table"]).read_text())

    def convert(self, texts: list[str]) -> list[str]:
        return [self.table.get(self._canon(t), "") for t in texts]


def build_messages(prompts: dict, text: str) -> list[dict]:
    """Chat messages for one input: system, optional few-shot `examples` ({spoken, latex}) as prior turns, then `user_prefix + text`."""
    msgs = [{"role": "system", "content": prompts["system"]}]
    for ex in prompts.get("examples") or []:
        msgs.append({"role": "user", "content": prompts["user_prefix"] + ex["spoken"]})
        msgs.append({"role": "assistant", "content": ex["latex"]})
    msgs.append({"role": "user", "content": prompts["user_prefix"] + text})
    return msgs


def _dtype(spec: dict, device: str):
    import torch

    if device == "cpu":
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        spec.get("dtype", "bfloat16")
    ]


class HFCausalConverter:
    """One instruction-tuned causal LM with its chat template; greedy or beam search as configured."""

    def __init__(self, name: str, spec: dict, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.spec = name, spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(spec["model"], revision=spec.get("revision"))
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                spec["model"], revision=spec.get("revision"), torch_dtype=_dtype(spec, self.device)
            )
            .to(self.device)
            .eval()
        )
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.post = POST[spec.get("postprocess", "none")]

    def _generate(self, model, texts, prompts, gen, batch_size=16):
        import torch

        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = [
                self.tok.apply_chat_template(build_messages(prompts, t), tokenize=False, add_generation_prompt=True)
                for t in texts[i : i + batch_size]
            ]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                ids = model.generate(**enc, pad_token_id=self.tok.pad_token_id, **gen)
            outs.extend(self.tok.batch_decode(ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True))
        return outs

    def convert(self, texts: list[str]) -> list[str]:
        raw = self._generate(self.model, texts, self.spec["prompts"], dict(self.spec["generation"]))
        self.last_raw = raw
        return [self.post(r) for r in raw]


class LoraTwoStageConverter(HFCausalConverter):
    """Base model + corrector adapter (transcript -> corrected spoken form) + translator adapter (spoken form -> LaTeX)."""

    def __init__(self, name: str, spec: dict, device: str | None = None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.spec = name, spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        paths = {k: expand(v) for k, v in spec["adapters"].items()}
        for k, p in paths.items():
            if not Path(p).exists():
                raise FileNotFoundError(f"converter adapter {k!r} not found at {p}")
        self.tok = AutoTokenizer.from_pretrained(spec["base_model"], revision=spec.get("base_revision"))
        base = (
            AutoModelForCausalLM.from_pretrained(
                spec["base_model"], revision=spec.get("base_revision"), torch_dtype=_dtype(spec, self.device)
            )
            .to(self.device)
            .eval()
        )
        self.model = PeftModel.from_pretrained(base, paths["corrector"], adapter_name="corrector")
        self.model.load_adapter(paths["translator"], adapter_name="translator")
        self.model.eval()
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.post = POST[spec.get("postprocess", "first_line")]

    def convert(self, texts: list[str]) -> list[str]:
        gen = dict(self.spec["generation"])
        self.model.set_adapter("corrector")
        corrected = [self.post(r) for r in self._generate(self.model, texts, self.spec["prompts"]["corrector"], gen)]
        self.model.set_adapter("translator")
        return [self.post(r) for r in self._generate(self.model, corrected, self.spec["prompts"]["translator"], gen)]


def load_converter(name: str | None = None, cfg: dict | None = None, **kw):
    cfg = cfg or load_yaml("converter.yaml")
    name = name or cfg["default"]
    spec = cfg["converters"][name]
    kinds = {
        "fake": lambda: FakeConverter(spec),
        "hf_causal": lambda: HFCausalConverter(name, spec, **kw),
        "hf_lora_two_stage": lambda: LoraTwoStageConverter(name, spec, **kw),
    }
    return kinds[spec["kind"]]()
