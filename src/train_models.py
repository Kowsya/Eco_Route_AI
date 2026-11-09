"""
train_models.py
---------------
PHASE 1: Train Model 1 (Decision Tree) and Model 2 (Random Forest).

Run from the project root:
        python -m src.train_models

What it does
    1. Loads data + chronological split (train / validation / test)
    2. Tries a SMALL grid of hyper-parameters (2 weeks only -> don't overdo it)
       and picks the best by MAE on the VALIDATION set (never the test set)
    3. Saves models/decision_tree.pkl and models/random_forest.pkl
    4. Saves the drift baseline (training feature distribution)
    5. Writes results/baseline_results.csv  (MAE, RMSE, latency on TEST set)
"""
import itertools
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from src import config as C
from src.preprocessing import get_splits
from src import drift


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error: penalises big mistakes more than MAE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def single_request_latency_ms(model, X: pd.DataFrame, n: int = 300) -> float:
    """
    Average time (milliseconds) to predict ONE row, like a real API call.
    We time single-row predictions because that is how the API uses the model.
    """
    rows = [X.iloc[[i]] for i in range(min(n, len(X)))]
    model.predict(rows[0])                       # warm-up (first call is slower)
    t0 = time.perf_counter()
    for r in rows:
        model.predict(r)
    return (time.perf_counter() - t0) / len(rows) * 1000


def tune_decision_tree(X_tr, y_tr, X_val, y_val):
    """Grid-search a few Decision Tree settings using VALIDATION MAE."""
    grid = {"max_depth": [4, 6, 8, 12],
            "min_samples_split": [2, 10, 30],
            "min_samples_leaf": [1, 5, 20]}
    best, best_mae, rows = None, 1e18, []
    for d, s, l in itertools.product(grid["max_depth"], grid["min_samples_split"],
                                     grid["min_samples_leaf"]):
        m = DecisionTreeRegressor(max_depth=d, min_samples_split=s, min_samples_leaf=l,
                                  random_state=C.RANDOM_STATE).fit(X_tr, y_tr)
        mae = mean_absolute_error(y_val, m.predict(X_val))
        rows.append((d, s, l, mae))
        if mae < best_mae:
            best, best_mae = dict(max_depth=d, min_samples_split=s, min_samples_leaf=l), mae
    print(f"[DT] tried {len(rows)} settings -> best {best}  (val MAE={best_mae:.3f})")
    return best


def tune_random_forest(X_tr, y_tr, X_val, y_val):
    """Grid-search a few Random Forest settings using VALIDATION MAE."""
    grid = {"max_depth": [8, 12, None], "min_samples_leaf": [1, 5, 10]}
    best, best_mae = None, 1e18
    for d, l in itertools.product(grid["max_depth"], grid["min_samples_leaf"]):
        m = RandomForestRegressor(n_estimators=100, max_depth=d, min_samples_leaf=l,
                                  n_jobs=-1, random_state=C.RANDOM_STATE).fit(X_tr, y_tr)
        mae = mean_absolute_error(y_val, m.predict(X_val))
        if mae < best_mae:
            best, best_mae = dict(max_depth=d, min_samples_leaf=l), mae
    print(f"[RF] tried {len(grid['max_depth'])*len(grid['min_samples_leaf'])} settings "
          f"-> best {best}  (val MAE={best_mae:.3f})")
    return best


def main():
    print("\n>>> STEP 1: load + inspect + split data")
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_splits(verbose=True)

    print("\n>>> STEP 2: tune + train Decision Tree (Model 1 - lightweight)")
    dt_params = tune_decision_tree(X_tr, y_tr, X_val, y_val)
    dt = DecisionTreeRegressor(**dt_params, random_state=C.RANDOM_STATE).fit(X_tr, y_tr)
    joblib.dump(dt, C.DT_PATH)

    print("\n>>> STEP 3: tune + train Random Forest (Model 2 - expensive)")
    rf_params = tune_random_forest(X_tr, y_tr, X_val, y_val)
    # n_jobs=1 inside the saved model: on Cloud Run each request should use one
    # CPU thread; spawning threads per tiny request only adds overhead.
    rf = RandomForestRegressor(n_estimators=100, **rf_params, n_jobs=1,
                               random_state=C.RANDOM_STATE).fit(X_tr, y_tr)
    joblib.dump(rf, C.RF_PATH)

    print("\n>>> STEP 4: save drift baseline (training feature distribution)")
    drift.save_baseline(drift.build_baseline(X_tr))
    print("Saved", C.BASELINE_PATH)

    print("\n>>> STEP 5: evaluate both models on the held-out TEST period")
    rows = []
    for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
        pred = model.predict(X_te)
        rows.append({
            "model": name,
            "MAE_Wh": round(mean_absolute_error(y_te, pred), 3),
            "RMSE_Wh": round(rmse(y_te, pred), 3),
            "latency_ms_per_request": round(single_request_latency_ms(model, X_te), 4),
            "model_file_size_KB": round((C.DT_PATH if name == "Decision Tree"
                                         else C.RF_PATH).stat().st_size / 1024, 1),
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(C.RESULTS_DIR / "baseline_results.csv", index=False)

    # Keep an "accountability record" of how the models were produced
    meta = {"decision_tree_params": dt_params, "random_forest_params": rf_params,
            "n_estimators": 100, "features": C.FEATURES,
            "split": {"train": len(X_tr), "val": len(X_val), "test": len(X_te)}}
    C.META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    print("\nSaved models + results/baseline_results.csv")


if __name__ == "__main__":
    main()
