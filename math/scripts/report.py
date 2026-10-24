"""Summarize outputs/<recognizer>.json and outputs/attribution_<recognizer>.json into markdown tables."""

from __future__ import annotations

import json

from math_eval.cli import common_parser, resolve_dirs


def ci(d, k="estimate"):
    if not d or d.get(k) is None:
        return "—"
    return f"{d[k]:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def num(x):
    return "—" if x is None else f"{x:.3f}"


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    a = ap.parse_args(argv)
    _, out = resolve_dirs(a)
    runs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(out.glob("*.json"))
        if not p.name.startswith("attribution_")
    ]
    attrib = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("attribution_*.json"))]
    if not runs:
        raise SystemExit(f"no analysis outputs in {out}; run analyze.py first")
    lines = [
        "## Base rate (no prompt)",
        "",
        "| recognizer | set | items | strict error | form-tolerant error | form share |",
        "|---|---|---|---|---|---|",
    ]
    names = {
        "benchmark_none": "benchmark",
        "discourse_D0": "in-discourse, expression",
        "discourse_D1_in_context": "in-discourse, 30-s window",
    }
    for r in runs:
        for key, label in names.items():
            h = r["H1a"].get(key)
            if h and h.get("n"):
                lines.append(
                    f"| {r['recognizer_id']} | {label} | {h['n']} | {ci(h['E_S'])} | {ci(h['E_A'])} | {ci(h['share'])} |"
                )
    fitted = [r for r in runs if r.get("H1b")]
    if fitted:
        lines += [
            "",
            "## Items against comparison words (30-s window)",
            "",
            "| recognizer | syllables | item error | word error | odds ratio [95% CI] | p |",
            "|---|---|---|---|---|---|",
        ]
        for r in fitted:
            for blk in (r["H1b"], r.get("H1b_full_set_descriptive") or {}):
                o = blk.get("OR_is_item") or {}
                rng = blk.get("n_syll_range")
                orr = f"{o['OR']:.2f} [{o['lo']:.2f}, {o['hi']:.2f}]" if o.get("OR") else "—"
                p = f"{o['p']:.3g}" if o.get("p") is not None else "—"
                syl = f"{rng[0]}–{rng[1]}" if rng else "all"
                lines.append(
                    f"| {r['recognizer_id']} | {syl} | {num(blk.get('error_items'))} | {num(blk.get('error_controls'))} | {orr} | {p} |"
                )
    steering = [r for r in runs if r.get("H2a")]
    if steering:
        lines += [
            "",
            "## Prompt steering (notation minus words prompt, pooled over pairs)",
            "",
            "| recognizer | notation share change | recall change, notation-dominant cell | recall change, words-dominant cell |",
            "|---|---|---|---|",
        ]
        for r in steering:
            h2a, h2b = r["H2a"].get("pooled", {}), r["H2b"].get("pooled", {})
            cells = h2b.get("cell_differences", {})
            lines.append(
                f"| {r['recognizer_id']} | {ci(h2a.get('difference_S_minus_W'))} | {ci(cells.get('notation'))} | {ci(cells.get('words'))} |"
            )
    if attrib:
        lines += [
            "",
            "## Recognition and conversion (in-discourse, fixed converter)",
            "",
            "| recognizer | converter | items | LaTeX error, transcript | LaTeX error, reference | recognition excess | P(R=0) change, caption prompt |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in attrib:
            h3a, h3b = r["H3a"], r.get("H3b") or {}
            lines.append(
                f"| {r['recognizer_id']} | {r['converter']} | {r['n_items']} | {ci(h3a.get('P_C0_asr'))} | {ci(h3a.get('P_C0_reference'))} | "
                f"{ci(h3a.get('delta_rec'))} | {ci(h3b.get('difference_D2_minus_D0'))} |"
            )
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
