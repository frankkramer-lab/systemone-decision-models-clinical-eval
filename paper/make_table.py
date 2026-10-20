"""Table 1 data from runs/results.csv, runs/latency.jsonl and the network references (no hand-copied numbers).

Cells are zero-shot "F1/ECE" without the leading zero (e.g. .612/.233); the one-shot results are in runs/results.csv.
Rows are grouped by model type. Latency: median per decision, mean over Medical Abstracts and HoC; for hosted or shared
endpoints the median round trip of a rejected request (no model computation) is subtracted, marked "a".

    python paper/make_table.py   -> paper/table1.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RUNS = HERE.parent / "runs"
TASKS = [("medabs", "Medical Abstracts"), ("hoc", "Hallmarks of Cancer"), ("cxr", "MIMIC-CXR")]
GROUPS = [
    ("System One models", [("jeff-0.8b", "Jeff 0.8B"), ("jeff-2b", "Jeff 2B"), ("openjev-fp8", "OpenJev 27B"),
                           ("jev", "Jev (API)")]),
    ("Non-fine-tuned base models of Jeff (same prompt and scoring)", [("qwen3.5-0.8b", "Qwen3.5 0.8B"),
                                                                 ("qwen3.5-2b", "Qwen3.5 2B")]),
    ("LLMs, probabilities of the answer options", [("chat_Qwen_Qwen3.8-27B-FP8", "Qwen3.8 27B"),
                                                  ("chat_GLM-5.3-Flash", "GLM 5.3 Flash"),
                                                  ("or_deepseek_deepseek-v4-flash-0731", "DeepSeek V4 Flash (API)")]),
    ("LLM with reasoning", [("reason_Qwen_Qwen3.8-27B-FP8", "Qwen3.8 27B + reasoning")]),
]
HOSTED = {"jev": "jev", "or_deepseek_deepseek-v4-flash-0731": "openrouter"}  # GLM: institutional server, network time ~5 ms, not subtracted
NOT_PERMITTED = {("cxr", "jev"), ("cxr", "or_deepseek_deepseek-v4-flash-0731")}


def f(x):
    """.612 style: three decimals, no leading zero."""
    return f"{x:.3f}".replace("0.", ".", 1)


def network_refs():
    """Median round trip (ms) of a rejected request per endpoint; several measurements are averaged."""
    refs = {"jev": [json.loads((RUNS / "jev_network.json").read_text())["warm_rejected_request_ms"]["median"]]}
    p = RUNS / "network_refs.jsonl"
    for r in (map(json.loads, open(p)) if p.exists() else []):
        refs.setdefault(r["name"], []).append(r["rejected_request_ms"]["median"])
    return {k: float(np.mean(v)) for k, v in refs.items()}


def latency():
    lat = pd.DataFrame([json.loads(l) for l in open(RUNS / "latency.jsonl")]).drop_duplicates(["backend", "task"], keep="last")
    lat["model"] = (lat.backend.str.replace("/", "_", regex=False).str.replace(":", "_", regex=False))
    return lat


def build():
    """-> [("group", title) | ("row", name, cells, ms_text)], network refs."""
    df = pd.read_csv(RUNS / "results.csv")
    lat, net = latency(), network_refs()
    out = []
    for title, models in GROUPS:
        out.append(("group", title))
        for m, name in models:
            cells = []
            for t, _ in TASKS:
                na = "n/a" if (t, m) in NOT_PERMITTED else "-"
                z = df[(df.model == m) & (df.task == t) & (df.k == 0)]
                cells.append(f"{f(z.f1.iloc[0])}/{f(z.ece.iloc[0])}" if len(z) else na)
            ms = lat[(lat.model == m) & lat.task.isin(["medabs", "hoc"])].median_ms
            if len(ms) != 2:
                ms_text = "-"
            elif m in HOSTED:
                ms_text = f"{ms.mean() - net[HOSTED[m]]:.0f}a" if HOSTED[m] in net else "-"
            else:
                ms_text = f"{ms.mean():.0f}"
            out.append(("row", name, cells, ms_text))
    out.append(("group", "Rule-based"))
    out.append(("row", "CheXpert labeler [7]", ["-", "-", f"{f(0.799)}/-"], "-"))
    return out, net


def main():
    rows, net = build()
    head = "| Model | " + " | ".join(t for _, t in TASKS) + " | ms |"
    lines = [head, "|" + "---|" * 5]
    for r in rows:
        if r[0] == "group":
            lines.append(f"| *{r[1]}* |" + " |" * 4)
        else:
            lines.append(f"| {r[1]} | " + " | ".join(r[2]) + f" | {r[3].replace('a', '^a^')} |")
    (HERE / "table1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("network refs (ms):", {k: round(v, 1) for k, v in net.items()})


if __name__ == "__main__":
    main()
