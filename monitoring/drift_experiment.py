"""
monitoring/drift_experiment.py
------------------------------
PHASE 7: Data-drift demonstration (works offline, no cloud needed).

    1. IN-DISTRIBUTION = random rows from the training period (should be "Normal")
    2. REAL TEST PERIOD = May 2016 rows. The dataset runs winter -> spring, so
                      real SEASONAL drift is expected here (a genuine finding!)
    3. SIMULATED DRIFT = test rows with features artificially shifted:
                      outside humidity +25 %, wind speed x3, indoor humidity +15 %
    All are compared against the training baseline using PSI (src/drift.py)

Run:   python -m monitoring.drift_experiment
Expected: train sample -> Normal ; real test period -> Warning (seasonal change);
          simulated drift -> very high PSI.
It also shows what drift does to the models (MAE) because drift is only a
WARNING; the real effect must be checked with labelled data.
Saves results/drift_experiment.csv.
"""
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src import config as C
from src import drift
from src.preprocessing import get_splits


def main():
    X_tr, y_tr, _, _, X_te, y_te = get_splits()
    baseline = drift.load_baseline()
    dt, rf = joblib.load(C.DT_PATH), joblib.load(C.RF_PATH)

    in_dist = X_tr.sample(n=len(X_te), random_state=C.RANDOM_STATE)   # same size as test
    y_in = y_tr.loc[in_dist.index]

    drifted = X_te.copy()                                   # artificial drift
    drifted["RH_out"] = (drifted["RH_out"] + 25).clip(upper=100)
    drifted["Windspeed"] = drifted["Windspeed"] * 3
    for c in ("RH_1", "RH_2", "RH_3"):
        drifted[c] = (drifted[c] + 15).clip(upper=100)

    rows = []
    for name, data, y_true in [("In-distribution (train sample)", in_dist, y_in),
                               ("Real test period (seasonal)", X_te, y_te),
                               ("Simulated drift", drifted, y_te)]:
        rep = drift.drift_report(baseline, data)
        worst = max(rep["features"].items(), key=lambda kv: kv[1]["psi"])
        rows.append({
            "dataset": name,
            "overall_status": rep["overall_status"],
            "max_PSI": rep["max_psi"],
            "most_drifted_feature": worst[0],
            "DT_MAE": round(mean_absolute_error(y_true, dt.predict(data)), 2),
            "RF_MAE": round(mean_absolute_error(y_true, rf.predict(data)), 2),
        })
        print(f"\n--- {name}: {rep['overall_status']} (max PSI {rep['max_psi']}) ---")
        for f, v in rep["features"].items():
            print(f"  {f:11s} PSI={v['psi']:<8} {v['status']}")
    out = pd.DataFrame(rows)
    print("\n", out.to_string(index=False))
    out.to_csv(C.RESULTS_DIR / "drift_experiment.csv", index=False)


if __name__ == "__main__":
    main()
