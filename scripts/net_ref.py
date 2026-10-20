"""Network reference for a hosted endpoint: median round trip of a request the server rejects without running a model
(no credentials, body {}), over one warm connection, plus the TCP connect time. Nothing is billed or computed.

    python scripts/net_ref.py <name> <url> [--n 60]   -> appends to runs/network_refs.jsonl
"""
import argparse, datetime, json, socket, ssl, time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("url"); ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()
host = urlparse(a.url).hostname
conn = []
for _ in range(20):
    t = time.perf_counter(); s = socket.create_connection((host, 443), timeout=10); conn.append(time.perf_counter() - t); s.close()
warm, codes = [], set()
with httpx.Client(http2=False, timeout=30) as c:
    c.post(a.url, json={})  # open the connection
    for _ in range(a.n):
        t = time.perf_counter(); r = c.post(a.url, json={}); warm.append(time.perf_counter() - t); codes.add(r.status_code)
rec = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "name": a.name, "url": a.url,
       "status_codes": sorted(codes), "tcp_connect_ms_median": 1000 * float(np.median(conn)),
       "rejected_request_ms": {"n": len(warm), "median": 1000 * float(np.median(warm)), "p5": 1000 * float(np.percentile(warm, 5)),
                               "p95": 1000 * float(np.percentile(warm, 95))}}
print(json.dumps(rec))
with open(Path(__file__).resolve().parents[1] / "runs" / "network_refs.jsonl", "a") as fp:
    fp.write(json.dumps(rec) + "\n")
