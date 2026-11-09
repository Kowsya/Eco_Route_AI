"""
evaluate.py
-----------
PHASE 3: The MOST IMPORTANT experiment of the project.

Compares three approaches on the SAME held-out TEST period:
        A) Always Decision Tree
        B) Always Random Forest
        C) EcoRoute (router decides per request)

Run:    python -m src.evaluate

Outputs
    results/final_comparison.csv   (table for the report)
    results/comparison_charts.png  (charts for the report / PPT)

Latency is measured one request at a time (like a real API call), so numbers
include the router's own cost for approach C. Absolute numbers depend on YOUR
computer; compare approaches with each other, not with other machines.
"""
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # draw charts to a file (no window needed)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import config as C
from src.preprocessing import get_splits

# ------------------------------------------------------------------
# ASSUMED Cloud Run prices - ILLUSTRATIVE ONLY.
# BEFORE you put a cost number in the report, open the official Cloud Run
# pricing page for your region and update these 3 values.
# ------------------------------------------------------------------
PRICE_PER_VCPU_SECOND = 0.000024     # USD per vCPU-second  (CHECK current price)
PRICE_PER_GIB_SECOND = 0.0000025     # USD per GiB-second   (CHECK current price)
VCPU = 1.0                           # vCPU allocated to the container
MEMORY_GIB = 0.5                     # memory allocated (GiB)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def run_strategy(name, X, decide):
    """
    Run one request at a time through `decide(row_df) -> (prediction, used_rf)`.
    Returns predictions, per-request latency (seconds) and RF usage flags.
    """
    preds, lat, used = [], [], []
    for i in range(len(X)):
        row = X.iloc[[i]]                          # one-row DataFrame = one API request
        t0 = time.perf_counter()
        p, rf_flag = decide(row)
        lat.append(time.perf_counter() - t0)
        preds.append(p)
        used.append(rf_flag)
    return np.array(preds), np.array(lat), np.array(used)


def main():
    _, _, _, _, X_te, y_te = get_splits()
    dt, rf, router = (joblib.load(p) for p in (C.DT_PATH, C.RF_PATH, C.ROUTER_PATH))
    y = y_te.to_numpy()

    # warm-up so the first call's one-time overhead does not distort averages
    for m in (dt, rf, router):
        m.predict(X_te.iloc[[0]])

    # Three decision functions -------------------------------------------------
    def always_dt(row):
        return float(dt.predict(row)[0]), 0

    def always_rf(row):
        return float(rf.predict(row)[0]), 1

    def ecoroute(row):
        go_rf = int(router.predict(row)[0])       # router: 0 -> DT, 1 -> RF
        model = rf if go_rf else dt
        return float(model.predict(row)[0]), go_rf

    results = []
    for name, fn in [("DT only", always_dt), ("RF only", always_rf), ("EcoRoute", ecoroute)]:
        print(f"Running {name} over {len(X_te)} test requests ...")
        pred, lat, used = run_strategy(name, X_te, fn)
        total_s = lat.sum()
        # Compute-cost estimate = busy time x price of the allocated vCPU+memory
        est_cost_per_1k = (lat.mean() * 1000) * (VCPU * PRICE_PER_VCPU_SECOND
                                                 + MEMORY_GIB * PRICE_PER_GIB_SECOND)
        results.append({
            "approach": name,
            "MAE_Wh": round(mean_absolute_error(y, pred), 3),
            "RMSE_Wh": round(rmse(y, pred), 3),
            "avg_latency_ms": round(lat.mean() * 1000, 3),
            "p95_latency_ms": round(np.percentile(lat, 95) * 1000, 3),
            "throughput_req_per_s": round(len(lat) / total_s, 1),   # single-thread, compute only
            "DT_share_%": round((1 - used.mean()) * 100, 1),
            "RF_share_%": round(used.mean() * 100, 1),
            "est_compute_cost_per_1000_req_USD": round(est_cost_per_1k, 8),
        })

    table = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print("FINAL COMPARISON ON HELD-OUT TEST PERIOD")
    print("=" * 100)
    print(table.to_string(index=False))
    table.to_csv(C.RESULTS_DIR / "final_comparison.csv", index=False)

    # Charts -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, col, title in zip(ax, ["MAE_Wh", "RMSE_Wh", "avg_latency_ms", "RF_share_%"],
                             ["MAE (Wh, lower=better)", "RMSE (Wh, lower=better)",
                              "Avg latency (ms)", "Requests sent to RF (%)"]):
        a.bar(table["approach"], table[col], color=["#4C78A8", "#F58518", "#54A24B"])
        a.set_title(title, fontsize=10)
        for i, v in enumerate(table[col]):
            a.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(C.RESULTS_DIR / "comparison_charts.png", dpi=150)
    print("\nSaved results/final_comparison.csv and results/comparison_charts.png")


if __name__ == "__main__":
    main()
