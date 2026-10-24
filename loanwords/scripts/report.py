"""Summarize outputs/<recognizer>.json into markdown tables (base rate, then prompt steering)."""

from __future__ import annotations

import json

from loanword_eval.cli import common_parser, resolve_dirs


def ci(d, k="estimate"):
    if not d or d.get(k) is None:
        return "—"
    return f"{d[k]:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def main(argv=None) -> int:
    ap = common_parser(__doc__)
    a = ap.parse_args(argv)
    _, out = resolve_dirs(a)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.json"))]
    if not runs:
        raise SystemExit(f"no analysis outputs in {out}; run analyze.py first")
    lines = [
        "## Base rate (no prompt)",
        "",
        "| recognizer | items | strict error | form-tolerant error | form share | odds ratio, item vs word [95% CI] | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        h1a, h1b = r["H1a"], r["H1b"]
        o = h1b.get("OR_is_item") or {}
        orr = f"{o['OR']:.2f} [{o['lo']:.2f}, {o['hi']:.2f}]" if o.get("OR") else "—"
        p = f"{o['p']:.3g}" if o.get("p") is not None else "—"
        lines.append(
            f"| {r['recognizer_id']} | {h1a['n']} | {ci(h1a['E_S'])} | {ci(h1a['E_A'])} | {ci(h1a['share'])} | {orr} | {p} |"
        )
    steering = [r for r in runs if r.get("H2a")]
    if steering:
        lines += [
            "",
            "## Prompt steering (Latin minus Hangul prompt, pooled over pairs)",
            "",
            "| recognizer | Latin share change | recall change, Latin-dominant cell | recall change, Hangul-dominant cell |",
            "|---|---|---|---|",
        ]
        for r in steering:
            h2a, h2b = r["H2a"].get("pooled", {}), r["H2b"].get("pooled", {})
            cells = h2b.get("cell_differences", {})
            lines.append(
                f"| {r['recognizer_id']} | {ci(h2a.get('difference_L_minus_H'))} | {ci(cells.get('latin'))} | {ci(cells.get('hangul'))} |"
            )
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
