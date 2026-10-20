"""Scores for one prediction file, with item-level bootstrap 95% CIs (items resampled jointly across questions).

Per question: macro-F1 over its options, macro one-vs-rest AUROC. Pooled over all (item, question) decisions:
top-label ECE (15 equal-width bins) and multiclass Brier score. Headline "f1": the mean of the per-question macro-F1
(Medical Abstracts, MIMIC-CXR), except for the multi-label HoC, where it is the micro-F1 over the
positive labels (all yes decisions pooled over abstracts and hallmarks). "macro_f1" keeps the per-question mean for
every task (for HoC the earlier binary macro-F1).

    python -m jt.metrics runs/cxr/jeff-0.8b/k0_s0_o1.jsonl [...]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def load(path, field="probs"):
    """field="listed": the first pass only (options as listed), for the single-order ablation."""
    by_q = defaultdict(dict)
    with open(path) as fp:
        for r in map(json.loads, fp):
            by_q[r["q"]][r["id"]] = {**r, "probs": r[field]}
    return by_q


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def question_arrays(records, ids):
    keys = records[ids[0]]["options"]
    y = np.array([keys.index(records[i]["gold"]) for i in ids])
    P = np.array([[records[i]["probs"][k] for k in keys] for i in ids])
    return y, P


MICRO_POSITIVE = {"hoc": "true"}  # multi-label task -> its positive option; headline f1 = micro-F1 over that label


def scores(by_q, ids, micro_positive=None):
    per_f1, per_auc, conf, correct, brier = [], [], [], [], []
    tp = fp = fn = 0
    for q, records in by_q.items():
        y, P = question_arrays(records, ids)
        n = P.shape[1]
        per_f1.append(f1_score(y, P.argmax(1), labels=range(n), average="macro", zero_division=0))
        if micro_positive is not None:
            pos = records[ids[0]]["options"].index(micro_positive)
            pred, gold = P.argmax(1) == pos, y == pos
            tp += int((pred & gold).sum()); fp += int((pred & ~gold).sum()); fn += int((~pred & gold).sum())
        present = [c for c in range(n) if 0 < (y == c).sum() < len(y)]
        per_auc.append(np.mean([roc_auc_score(y == c, P[:, c]) for c in present]) if present else np.nan)
        conf.append(P.max(1))
        correct.append(P.argmax(1) == y)
        brier.append(((P - np.eye(n)[y]) ** 2).sum(1))
    conf, correct, brier = map(np.concatenate, (conf, correct, brier))
    macro = float(np.mean(per_f1))
    f1 = 2 * tp / (2 * tp + fp + fn) if micro_positive is not None and (tp + fp + fn) else macro
    return {"f1": float(f1), "macro_f1": macro, "auroc": float(np.nanmean(per_auc)), "ece": ece(conf, correct.astype(float)),
            "brier": float(brier.mean()), "accuracy": float(correct.mean())}


def evaluate(path, n_boot=1000, seed=0, field="probs"):
    by_q = load(path, field)
    ids = sorted(set.intersection(*(set(r) for r in by_q.values())))
    pos = MICRO_POSITIVE.get(Path(path).parent.parent.name)  # runs/<task>/<model>/<file>
    point = scores(by_q, ids, pos)
    rng = np.random.default_rng(seed)
    boots = [scores(by_q, [ids[j] for j in rng.integers(0, len(ids), len(ids))], pos) for _ in range(n_boot)]
    ci = {m: [float(np.percentile([b[m] for b in boots], 2.5)), float(np.percentile([b[m] for b in boots], 97.5))]
          for m in point} if boots else {}
    flat = [r for records in by_q.values() for r in records.values()]
    return {"file": str(path), "n_items": len(ids), "n_questions": len(by_q), **point, "ci95": ci,
            "seconds_per_question": float(np.median([r["seconds"] for r in flat])),
            "outside_mass": float(np.mean([r["outside"] for r in flat]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    for f in a.files:
        r = evaluate(f, a.n_boot)
        ci = r["ci95"] or {"macro_f1": [float("nan")] * 2}
        print(f"{f}\n  n={r['n_items']}x{r['n_questions']}  macroF1 {r['macro_f1']:.3f} [{ci['macro_f1'][0]:.3f},{ci['macro_f1'][1]:.3f}]"
              f"  AUROC {r['auroc']:.3f}  ECE {r['ece']:.3f}  Brier {r['brier']:.3f}  acc {r['accuracy']:.3f}"
              f"  {r['seconds_per_question'] * 1000:.0f} ms/q  outside {r['outside_mass']:.3f}")


if __name__ == "__main__":
    main()
