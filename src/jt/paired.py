"""Paired bootstrap differences between two models on the same task and condition (items resampled jointly for both).

Primary contrasts (paper/OUTLINE.md): decision-trained model minus its untrained same base. Exploratory: Jev versus
the 27B models. k=1: in every resample the difference is averaged over the few-shot seeds. Differences are A - B; for
ECE and Brier negative means A is better. No multiplicity correction; CIs are reported as estimates.

    python -m jt.paired [--n_boot 1000]   -> runs/paired.csv, runs/paired.md
"""
import argparse

import numpy as np
import pandas as pd

from jt import config as C
from jt import metrics

PRIMARY = [("jeff-0.8b", "qwen3.5-0.8b"), ("jeff-2b", "qwen3.5-2b"), ("openjev-fp8", "chat_Qwen_Qwen3.8-27B-FP8")]
EXPLORATORY = [("jev", "openjev-fp8"), ("jev", "chat_Qwen_Qwen3.8-27B-FP8"),
               # amendment 2: the decision model and the first-token LLM readout against the same LLM with reasoning
               ("openjev-fp8", "reason_Qwen_Qwen3.8-27B-FP8"), ("chat_Qwen_Qwen3.8-27B-FP8", "reason_Qwen_Qwen3.8-27B-FP8"),
               ("openjev-fp8", "chat_GLM-5.3-Flash"), ("openjev-fp8", "or_deepseek_deepseek-v4-flash-0731")]
METRICS = ("f1", "macro_f1", "auroc", "ece", "brier")


def files(task, model, k):
    seeds = [0] if k == 0 else [0, 1, 2]
    paths = [C.RUNS / task / model / f"k{k}_s{s}_o2.jsonl" for s in seeds]
    return paths if all(p.exists() for p in paths) else None


def contrast(task, a, b, k, n_boot, seed=0):
    fa, fb = files(task, a, k), files(task, b, k)
    if not fa or not fb:
        return None
    A, B = [metrics.load(p) for p in fa], [metrics.load(p) for p in fb]
    ids = sorted(set.intersection(*(set(r) for by_q in A + B for r in by_q.values())))

    pos = metrics.MICRO_POSITIVE.get(task)

    def diff(sample):
        sa = [metrics.scores(by_q, sample, pos) for by_q in A]
        sb = [metrics.scores(by_q, sample, pos) for by_q in B]
        return {m: np.mean([x[m] for x in sa]) - np.mean([x[m] for x in sb]) for m in METRICS}

    point = diff(ids)
    rng = np.random.default_rng(seed)
    boots = [diff([ids[j] for j in rng.integers(0, len(ids), len(ids))]) for _ in range(n_boot)]
    out = {"task": task, "a": a, "b": b, "k": k, "n_items": len(ids)}
    for m in METRICS:
        v = np.array([x[m] for x in boots])
        out |= {m: point[m], f"{m}_lo": np.percentile(v, 2.5), f"{m}_hi": np.percentile(v, 97.5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    rows = []
    for kind, pairs in (("primary", PRIMARY), ("exploratory", EXPLORATORY)):
        for task in ("medabs", "hoc", "cxr"):
            for x, y in pairs:
                for k in (0, 1):
                    r = contrast(task, x, y, k, a.n_boot)
                    if r:
                        rows.append({"kind": kind, **r})
                        print(f"{kind} {task} k{k} {x} - {y}: dF1 {r['f1']:+.3f} [{r['f1_lo']:+.3f}, "
                              f"{r['f1_hi']:+.3f}]  dECE {r['ece']:+.3f} [{r['ece_lo']:+.3f}, {r['ece_hi']:+.3f}]",
                              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.RUNS / "paired.csv", index=False)
    cell = lambda r, m: f"{r[m]:+.3f} [{r[m + '_lo']:+.3f}, {r[m + '_hi']:+.3f}]"
    md = ["| kind | task | k | A − B | ΔF1 | ΔmacroF1 | ΔAUROC | ΔECE | ΔBrier |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r["kind"]} | {r["task"]} | {r["k"]} | {r["a"]} − {r["b"]} | " + " | ".join(cell(r, m) for m in METRICS) + " |"
           for _, r in df.iterrows()]
    (C.RUNS / "paired.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
