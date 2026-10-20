"""Run one backend on one task and write a prediction per (item, question).

    python -m jt.run --task cxr --backend jeff-0.8b --k 0
    python -m jt.run --task medabs --backend chat:Qwen/Qwen3.8-27B-FP8 --k 1 --seed 2 --orders 2

--orders 2 also asks with the options reversed and averages the two distributions (position-bias control, as
jeff-serve's "orders": 2). Output: runs/<task>/<backend>/k<k>_s<seed>_o<orders>.jsonl, resumable.
"""
import argparse
import json
import time

from jt import backends, tasks
from jt import config as C


def reversed_question(question):
    if question["type"] == "noul":
        return {**question, "true_first": True}
    return {**question, "criteria": dict(reversed(list(question["criteria"].items())))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(tasks.TASKS))
    ap.add_argument("--backend", required=True)
    ap.add_argument("--k", type=int, default=0, choices=[0, 1], help="0 = zero-shot, 1 = one example per option")
    ap.add_argument("--seed", type=int, default=0, help="few-shot example draw")
    ap.add_argument("--orders", type=int, default=1, choices=[1, 2])
    ap.add_argument("--limit", type=int, help="first N test items (smoke tests)")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--pilot", action="store_true", help="held-out non-test items for checks before freezing")
    a = ap.parse_args()

    task = tasks.TASKS[a.task]
    out = C.RUNS / a.task / a.backend.replace("/", "_").replace(":", "_") / f"{'pilot_' if a.pilot else ''}k{a.k}_s{a.seed}_o{a.orders}{'_lim' + str(a.limit) if a.limit else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as fp:
            done = {(r["id"], r["q"]) for r in map(json.loads, fp)}
    todo = [r for r in tasks.requests(task, a.k, a.seed, a.limit, a.pilot) if (r[0], r[1]) not in done]
    print(f"{a.task} {a.backend} k={a.k} seed={a.seed} orders={a.orders}: {len(done)} done, {len(todo)} to go -> {out}")
    if not todo:
        return
    backend = backends.make(a.backend)
    backend.check_tier(task.tier)  # before any text leaves the process

    with open(out, "a") as fp:
        for start in range(0, len(todo), a.chunk):
            chunk = todo[start:start + a.chunk]
            rows = [r[3] for r in chunk]
            t = time.time()
            results = backend.predict(rows)
            if a.orders == 2:
                back = backend.predict([{**row, "question": reversed_question(row["question"])} for row in rows])
                for res, rev in zip(results, back):
                    res["listed"], res["reversed"] = res["probs"], rev["probs"][::-1]  # kept for the single-order ablation
                    res["probs"] = [(p + q) / 2 for p, q in zip(res["listed"], res["reversed"])]
                    res["seconds"] += rev["seconds"]
                    res["outside"] = (res["outside"] + rev["outside"]) / 2
                    if "extra" in res:
                        res["extra"] = {"listed": res["extra"], "reversed": rev["extra"]}
            for (item_id, qid, gold, row), res in zip(chunk, results):
                keys = tasks.option_keys(row["question"])
                fp.write(json.dumps({"id": item_id, "q": qid, "gold": gold, "options": keys,
                                     "probs": dict(zip(keys, res["probs"])), "pred": keys[max(range(len(keys)), key=res["probs"].__getitem__)],
                                     "seconds": res["seconds"], "outside": res["outside"],
                                     **({"listed": dict(zip(keys, res["listed"])), "reversed": dict(zip(keys, res["reversed"]))}
                                        if "listed" in res else {}),
                                     **({"extra": res["extra"]} if "extra" in res else {})}) + "\n")
            fp.flush()
            print(f"  {start + len(chunk)}/{len(todo)}  ({(time.time() - t) / len(chunk):.3f} s/question)", flush=True)


if __name__ == "__main__":
    main()
