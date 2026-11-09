"""
feature_selection_cv.py
-----------------------
Justifies the feature set in config.py WITHOUT touching the test set.

We use forward-in-time cross-validation (TimeSeriesSplit) on train+validation:
    fold 1: train on the oldest 1/6, validate on the next 1/6
    fold 2: train on the oldest 2/6, validate on the next 1/6 ... etc.
This mimics real deployment (always predicting the FUTURE) and exposes
features that only work because of seasonal patterns.

Run:  python -m src.feature_selection_cv      (takes ~1 minute)
Saves results/feature_set_cv.csv  -> screenshot / put the table in the report.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from src import config as C
from src.preprocessing import load_raw, add_time_features, chronological_split

# Candidate feature sets to compare
CANDIDATES = {
    "A: all temperature + humidity + weather + time": [
        "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T_out", "RH_out",
        "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint", "hour", "dayofweek"],
    "B: no absolute temperature (our choice)": C.FEATURES,
    "C: lights + indoor humidity + time only": [
        "lights", "RH_1", "RH_2", "RH_3", "hour", "dayofweek"],
}


def main():
    df = add_time_features(load_raw())
    train, val, _test = chronological_split(df)       # test part is IGNORED here
    dev = pd.concat([train, val])                       # development data only
    rows = []
    for name, feats in CANDIDATES.items():
        maes = {"DT": [], "RF": []}
        for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(dev):
            a, b = dev.iloc[tr_idx], dev.iloc[va_idx]
            models = {
                "DT": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20,
                                            random_state=C.RANDOM_STATE),
                "RF": RandomForestRegressor(n_estimators=60, max_depth=8,
                                            min_samples_leaf=10, n_jobs=-1,
                                            random_state=C.RANDOM_STATE),
            }
            for k, m in models.items():
                m.fit(a[feats], a[C.TARGET])
                maes[k].append(mean_absolute_error(b[C.TARGET], m.predict(b[feats])))
        rows.append({"feature_set": name, "n_features": len(feats),
                     "DT_cv_MAE": round(np.mean(maes["DT"]), 2),
                     "RF_cv_MAE": round(np.mean(maes["RF"]), 2)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(C.RESULTS_DIR / "feature_set_cv.csv", index=False)


if __name__ == "__main__":
    main()
