"""
config.py
---------
Single place that stores every constant used by training, evaluation,
monitoring and the API.  Keeping them here guarantees that the training
code and the API code use EXACTLY the same feature list (a very common
source of bugs in deployed ML systems: "training/serving skew").
"""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent        # EcoRouteAI/ folder
DATA_PATH = ROOT / "data" / "energydata_complete.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
for _d in (MODELS_DIR, RESULTS_DIR):
    try:
        _d.mkdir(exist_ok=True)
    except OSError:
        pass

DT_PATH = MODELS_DIR / "decision_tree.pkl"
RF_PATH = MODELS_DIR / "random_forest.pkl"
ROUTER_PATH = MODELS_DIR / "router.pkl"
BASELINE_PATH = MODELS_DIR / "drift_baseline.json"   # training feature statistics (for drift)
META_PATH = MODELS_DIR / "model_meta.json"           # versions, thresholds, accountability record

# ---------- Target ----------
TARGET = "Appliances"        # appliance energy use in Wh (per 10 minutes)

# ---------- Features ----------
# We deliberately use a compact feature set (data minimisation + simple API).
#   lights          -> lighting energy use (Wh)
#   RH_1, RH_2, RH_3 -> indoor humidity (%) in 3 rooms
#   RH_out          -> outside humidity (%)
#   Windspeed, Visibility -> weather station readings
#   hour, dayofweek -> time features extracted from the 'date' column
#
# WHY NO ABSOLUTE TEMPERATURE COLUMNS (T1..T9, T_out, Tdewpoint)?
#   The dataset spans winter -> late spring, so temperature acts as a "calendar
#   proxy": a chronologically-trained model memorises "cold = winter pattern" and
#   then fails when spring temperatures never seen in training appear.
#   src/feature_selection_cv.py proves this with forward-in-time cross-validation
#   on TRAIN+VALIDATION only (the test set is NOT used for this decision).
#   'rv1' and 'rv2' are random-noise columns in the UCI data, so we drop them.
FEATURES = [
    "lights", "RH_1", "RH_2", "RH_3", "RH_out",
    "Windspeed", "Visibility", "hour", "dayofweek",
]

# Numeric features that we monitor for drift (time features excluded)
DRIFT_FEATURES = [f for f in FEATURES if f not in ("hour", "dayofweek")]

# ---------- Chronological split (no shuffling!) ----------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15          # remaining 0.15 is the final test set

# ---------- Reproducibility ----------
RANDOM_STATE = 42

# ---------- Routing ----------
# 'COST_WEIGHT' = the "price" (in Wh of MAE) we are willing to pay for using
# the Random Forest on one request.  Larger value -> router uses RF less often.
COST_WEIGHT = 1.0
# candidate "RF must be better than DT by at least this many Wh" thresholds
MARGIN_GRID = [0, 5, 10, 15, 20, 30, 40]
