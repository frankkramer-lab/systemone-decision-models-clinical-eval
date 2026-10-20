"""Collect every finished run (orders=2, full test set) into one table: runs/results.csv and runs/results.md.

k=0: point estimate with bootstrap 95% CI. k=1: mean and SD over the few-shot seeds. Field "listed" gives the
single-order ablation (first pass only, options as listed).

    python -m jt.report [--field probs|listed] [--n_boot 1000]
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from jt import config as C
from jt import metrics

NAME = re.compile(r"^k(?P<k>\d)_s(?P<seed>\d)_o2\.jsonl$")
ORDER = ["qwen3.5-0.8b", "jeff-0.8b", "qwen3.5-2b", "jeff-2b", "chat_Qwen_Qwen3.8-27B-FP8", "openjev-fp8", "jev",
         "chat_GLM-5.3-Flash", "or_deepseek_deepseek-v4-flash-0731", "reason_Qwen_Qwen3.8-27B-FP8"]


def rows(field, n_boot):
    out = []
    for path in sorted(C.RUNS.glob("*/*/k*_o2.jsonl")):
        m = NAME.match(path.name)
        if not m:
            continue
        task, backend = path.parent.parent.name, path.parent.name
        expected = {"medabs": 2888, "hoc": 371 * 10, "cxr": 685 * 4}.get(task)
        n = sum(1 for _ in open(path))
        if expected and n != expected:
            print(f"skip incomplete {path.relative_to(C.ROOT)} ({n}/{expected})")
            continue
        if field != "probs":
            with open(path) as fp:
                first = json.loads(fp.readline())
            if field not in first:
                continue
        r = metrics.evaluate(path, n_boot=n_boot if m["k"] == "0" else 0, field=field)
        out.append({"task": task, "model": backend, "k": int(m["k"]), "seed": int(m["seed"]), **{
            key: r[key] for key in ("f1", "macro_f1", "auroc", "ece", "brier", "accuracy", "seconds_per_question", "outside_mass")},
            "f1_lo": r["ci95"].get("f1", [np.nan])[0], "f1_hi": r["ci95"].get("f1", [np.nan, np.nan])[1]})
    return pd.DataFrame(out)


def table(df):
    lines = []
    for task in ("medabs", "hoc", "cxr"):
        sub = df[df.task == task]
        if sub.empty:
            continue
        lines += [f"\n### {task}\n", "| model | k=0 F1 [95% CI] | k=0 AUROC | k=0 ECE | k=0 Brier | k=1 macro-F1 (mean ± SD, n seeds) | k=1 ECE |",
                  "|---|---|---|---|---|---|---|"]
        for model in sorted(sub.model.unique(), key=lambda m: ORDER.index(m) if m in ORDER else 99):
            z = sub[(sub.model == model) & (sub.k == 0)]
            f = sub[(sub.model == model) & (sub.k == 1)]
            zero = (f"{z.f1.iloc[0]:.3f} [{z.f1_lo.iloc[0]:.3f}, {z.f1_hi.iloc[0]:.3f}] | {z.auroc.iloc[0]:.3f} | "
                    f"{z.ece.iloc[0]:.3f} | {z.brier.iloc[0]:.3f}") if len(z) else "– | – | – | –"
            few = (f"{f.f1.mean():.3f} ± {f.f1.std(ddof=1) if len(f) > 1 else 0:.3f} (n={len(f)}) | {f.ece.mean():.3f}"
                   if len(f) else "– | –")
            lines.append(f"| {model} | {zero} | {few} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="probs", choices=["probs", "listed"])
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    df = rows(a.field, a.n_boot)
    suffix = "" if a.field == "probs" else f"_{a.field}"
    df.to_csv(C.RUNS / f"results{suffix}.csv", index=False)
    md = table(df)
    (C.RUNS / f"results{suffix}.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
