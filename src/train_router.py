"""
train_router.py
---------------
PHASE 2: Train Model 3 = the ROUTER (a very small Decision Tree Classifier).

Run (after train_models.py):
        python -m src.train_router

CONCEPT
    For every VALIDATION row we know three things:
          DT prediction, RF prediction, real value.
    So we can compute how much better (or worse) the Random Forest was:

          gain = |actual - DT_pred| - |actual - RF_pred|      (in Wh)
                 gain > 0  -> RF was closer to the truth

    The router label is:
          1 ("use RF")  if gain > MARGIN     (RF's benefit is worth its cost)
          0 ("use DT")  otherwise

    The router is trained on the VALIDATION set because DT/RF predictions on the
    TRAINING set are over-optimistic (the models have already seen those rows).

HOW THE MARGIN IS CHOSEN (cost-aware part)
    Each candidate margin is scored with forward-in-time cross-validation inside
    the validation set (router trained on older rows, tested on newer rows):

          score = routed_MAE + COST_WEIGHT * (fraction of requests sent to RF)

    COST_WEIGHT is "how many Wh of MAE is one RF call worth paying for".
    The test set is NOT used here.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from src import config as C
from src.preprocessing import get_splits


def make_router() -> DecisionTreeClassifier:
    """
    Tiny on purpose (depth 3): the routing decision must be MUCH cheaper than
    running the Random Forest, otherwise routing would defeat its own purpose.
    """
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=40,
                                  class_weight="balanced", random_state=C.RANDOM_STATE)


def main():
    _, _, X_val, y_val, _, _ = get_splits()
    dt = joblib.load(C.DT_PATH)
    rf = joblib.load(C.RF_PATH)

    # ---- Step 1: predictions of both models on validation data ----
    dt_pred, rf_pred = dt.predict(X_val), rf.predict(X_val)
    dt_err = np.abs(y_val.to_numpy() - dt_pred)
    rf_err = np.abs(y_val.to_numpy() - rf_pred)
    gain = dt_err - rf_err                       # positive -> RF better on that row
    print(f"Validation MAE  DT={dt_err.mean():.2f}  RF={rf_err.mean():.2f}")
    print(f"RF strictly better on {np.mean(gain > 0)*100:.1f}% of validation rows")

    # ---- Step 2: choose the margin by forward-in-time CV on validation ----
    X = X_val.to_numpy()
    rows = []
    for margin in C.MARGIN_GRID:
        label = (gain > margin).astype(int)
        if label.sum() < 50 or label.sum() == len(label):    # not enough of both classes
            continue
        routed_err, chosen, truth = [], [], []
        for tr, te in TimeSeriesSplit(n_splits=4).split(X):
            if len(np.unique(label[tr])) < 2:
                continue
            clf = make_router().fit(X[tr], label[tr])
            go_rf = clf.predict(X[te]).astype(bool)
            routed_err.append(np.where(go_rf, rf_err[te], dt_err[te]))
            chosen.append(go_rf)
            truth.append(dt_err[te].mean())
        routed_err, chosen = np.concatenate(routed_err), np.concatenate(chosen)
        eval_idx = np.concatenate([te for tr, te in TimeSeriesSplit(n_splits=4).split(X)])
        rows.append({
            "margin_Wh": margin,
            "RF_share_%": round(chosen.mean() * 100, 1),
            "routed_MAE": round(routed_err.mean(), 3),
            "DT_only_MAE_same_rows": round(dt_err[eval_idx].mean(), 3),
            "RF_only_MAE_same_rows": round(rf_err[eval_idx].mean(), 3),
            "score": round(routed_err.mean() + C.COST_WEIGHT * chosen.mean(), 3),
        })
    table = pd.DataFrame(rows)
    print("\nMargin search (lower score = better quality/cost balance):")
    print(table.to_string(index=False))
    table.to_csv(C.RESULTS_DIR / "router_margin_search.csv", index=False)

    best_margin = int(table.loc[table["score"].idxmin(), "margin_Wh"])
    print(f"\n>>> Selected margin = {best_margin} Wh")

    # ---- Step 3: fit the FINAL router on all validation rows ----
    label = (gain > best_margin).astype(int)
    router = make_router().fit(X_val, label)
    joblib.dump(router, C.ROUTER_PATH)
    print("Router training accuracy on its labels:",
          round(accuracy_score(label, router.predict(X_val)), 3))
    print("Saved", C.ROUTER_PATH)

    # ---- Step 4: record decisions for accountability ----
    meta = json.loads(C.META_PATH.read_text())
    meta["router"] = {"margin_Wh": best_margin, "cost_weight": C.COST_WEIGHT,
                      "max_depth": 3, "trained_on": "validation split"}
    C.META_PATH.write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
