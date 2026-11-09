"""
monitoring/load_test.py
-----------------------
SMALL, CONTROLLED load test (do NOT run unlimited traffic against Cloud Run).
Uses only the Python standard library + the dataset, so no extra installs.

Usage examples
    # local API
    python monitoring/load_test.py --url http://localhost:8080 --mode ecoroute --n 300 --workers 5
    # Cloud Run API (use YOUR url)
    python monitoring/load_test.py --url https://ecoroute-xxxx.a.run.app --mode dt --n 300 --workers 10

    --mode     ecoroute | dt | rf        (which strategy the API should use)
    --n        total number of requests (keep it small, e.g. 200-500)
    --workers  number of parallel clients (= how much concurrency you generate)
    --drift    send artificially shifted data (to trigger the /drift warning)

It replays REAL rows from the TEST period so the data looks realistic, measures
the end-to-end latency seen by the client (includes network), and prints:
average / p95 latency, throughput, errors and the share of requests per model.
It appends a line to results/load_test_log.csv so you can compare experiments.
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # allow "import src"
from src import config as C
from src.preprocessing import get_splits


def build_payloads(n, drift=False):
    """Take n real rows from the test period and turn them into API request bodies.
    If drift=True the features are artificially shifted (same shift as
    monitoring/drift_experiment.py) to demonstrate drift detection on the live API."""
    _, _, _, _, X_te, _ = get_splits()
    rows = X_te.sample(n=n, replace=True, random_state=1).copy()
    if drift:
        rows["RH_out"] = (rows["RH_out"] + 25).clip(upper=100)
        rows["Windspeed"] = (rows["Windspeed"] * 3).clip(upper=60)
        for c in ("RH_1", "RH_2", "RH_3"):
            rows[c] = (rows[c] + 15).clip(upper=100)
    return [{k: (int(v) if k in ("hour", "dayofweek") else float(v)) for k, v in r.items()}
            for r in rows.to_dict("records")]


def one_request(url, payload):
    """Send one POST /predict and return (latency_seconds, ok, model_used)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        return time.perf_counter() - t0, True, body.get("model_used")
    except Exception:
        return time.perf_counter() - t0, False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--mode", default="ecoroute", choices=["ecoroute", "dt", "rf"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--label", default="", help="free text, e.g. 'maxinst=1,conc=10'")
    ap.add_argument("--drift", action="store_true", help="send artificially drifted data")
    a = ap.parse_args()
    if a.n > 2000:
        sys.exit("Refusing n > 2000: keep tests small and cheap.")

    endpoint = a.url.rstrip("/") + f"/predict?mode={a.mode}"
    payloads = build_payloads(a.n, a.drift)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda p: one_request(endpoint, p), payloads))
    wall = time.perf_counter() - t_start

    lat = np.array([r[0] for r in results]) * 1000
    ok = np.array([r[1] for r in results])
    models = [r[2] for r in results if r[1]]
    rf_share = 100 * models.count("random_forest") / max(len(models), 1)

    summary = {
        "label": a.label, "mode": a.mode, "requests": a.n, "workers": a.workers,
        "errors": int((~ok).sum()),
        "avg_latency_ms": round(float(lat.mean()), 1),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 1),
        "throughput_req_s": round(a.n / wall, 1),
        "rf_share_percent": round(rf_share, 1),
    }
    print(json.dumps(summary, indent=2))

    log = C.RESULTS_DIR / "load_test_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary.keys())
        if new:
            w.writeheader()
        w.writerow(summary)


if __name__ == "__main__":
    main()
