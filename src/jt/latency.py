"""Per-decision latency, warm, one request at a time (batch 1), zero-shot, single order.

200 fixed test questions per task (seeded sample) after 20 warm-up questions; median and p95 wall time per decision
as seen by the caller. Run each backend alone on an otherwise idle GPU; remote endpoints include network time and
are reported separately.

    python -m jt.latency --backend jeff-0.8b [--n 200]
"""
import argparse
import json
import random
import time

import numpy as np

from jt import backends, tasks
from jt import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tasks", nargs="+", default=["medabs", "hoc", "cxr"])
    a = ap.parse_args()
    backend = backends.make(a.backend)
    if hasattr(backend, "concurrency"):
        backend.concurrency = 1
    out = C.RUNS / "latency.jsonl"
    for name in a.tasks:
        task = tasks.TASKS[name]
        backend.check_tier(task.tier)
        pool = list(tasks.requests(task, 0))
        sample = random.Random(f"latency-{name}").sample(pool, a.n + 20)
        times = []
        for i, (_, _, _, row) in enumerate(sample):
            t = time.perf_counter()
            backend.predict([row])
            if i >= 20:
                times.append(time.perf_counter() - t)
        r = {"backend": a.backend, "task": name, "n": len(times), "median_ms": 1000 * float(np.median(times)),
             "p95_ms": 1000 * float(np.percentile(times, 95)), "host": backend.host}
        print(json.dumps(r))
        with open(out, "a") as fp:
            fp.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
