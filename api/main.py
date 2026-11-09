"""
api/main.py
-----------
EcoRoute AI - FastAPI prediction service (runs locally AND on Cloud Run).

Endpoints
    GET  /          project information
    GET  /health    health check (are all 3 model files loaded?)
    POST /predict   validate -> route -> predict -> return prediction + model used
    GET  /metrics   request count, latency average/p95, DT/RF usage, error count
    GET  /drift     data-drift status of recent requests vs the training baseline

Run locally (from the project root):
        uvicorn api.main:app --reload --port 8080
Then open  http://localhost:8080/docs  (interactive Swagger page).
"""
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src import config as C
from src import drift

from pathlib import Path

# ----------------------------------------------------------------------
# 1. LOGGING  (privacy-friendly)
#    Cloud Run collects anything printed to stdout and sends it to Cloud
#    Logging. We print ONE JSON line per request containing only: request id,
#    model used, latency, status. We NEVER log the input values.
# ----------------------------------------------------------------------
logger = logging.getLogger("ecoroute")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()          # stdout/stderr -> Cloud Logging
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def log_event(**fields):
    """Write a structured (JSON) log line. Cloud Logging can then filter on fields."""
    logger.info(json.dumps(fields))


# ----------------------------------------------------------------------
# 2. IN-MEMORY STATE (models + monitoring counters)
#    Simple and free; resets when the Cloud Run instance restarts.
#    (A production system would export these to Cloud Monitoring / Prometheus.)
# ----------------------------------------------------------------------
STATE = {
    "models": {},                          # filled at startup
    "baseline": None,                      # drift baseline (training stats)
    "latencies_ms": deque(maxlen=2000),    # last 2000 request latencies
    "window": deque(maxlen=500),           # last 500 FEATURE rows (numbers only, no identity)
    "counts": {"requests": 0, "dt": 0, "rf": 0, "errors": 0, "validation_errors": 0},
    "started": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the container starts: load the 3 models + drift baseline."""
    try:
        STATE["models"] = {
            "dt": joblib.load(C.DT_PATH),
            "rf": joblib.load(C.RF_PATH),
            "router": joblib.load(C.ROUTER_PATH),
        }
        STATE["baseline"] = drift.load_baseline()
        log_event(event="startup", status="models_loaded")
    except Exception as exc:                      # file missing/corrupt -> /health will say unhealthy
        log_event(event="startup", status="model_load_failed", error=type(exc).__name__)
    yield


app = FastAPI(title="EcoRoute AI",
              description="Cost-aware model routing for smart-home energy prediction",
              version="1.0.0", lifespan=lifespan)

# Frontend files are copied into the Docker image by the Dockerfile.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)


# ----------------------------------------------------------------------
# 3. INPUT VALIDATION (security)
#    Pydantic checks types and ranges BEFORE our code runs. Wrong type,
#    missing field, out-of-range value or unknown extra field -> HTTP 422.
# ----------------------------------------------------------------------
class EnergyRequest(BaseModel):
    lights: float = Field(..., ge=0, le=200, description="Lighting energy use (Wh)")
    RH_1: float = Field(..., ge=0, le=100, description="Kitchen humidity (%)")
    RH_2: float = Field(..., ge=0, le=100, description="Living room humidity (%)")
    RH_3: float = Field(..., ge=0, le=100, description="Laundry room humidity (%)")
    RH_out: float = Field(..., ge=0, le=100, description="Outside humidity (%)")
    Windspeed: float = Field(..., ge=0, le=60, description="Wind speed (m/s)")
    Visibility: float = Field(..., ge=0, le=100, description="Visibility (km)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of week (Monday=0)")

    model_config = ConfigDict(
        extra="forbid",                              # reject unexpected fields (security)
        json_schema_extra={"example": {
            "lights": 10, "RH_1": 45.5, "RH_2": 40.2, "RH_3": 41.0, "RH_out": 80,
            "Windspeed": 4.0, "Visibility": 40, "hour": 18, "dayofweek": 2}},
    )


class PredictionResponse(BaseModel):
    request_id: str
    prediction_wh: float
    model_used: str
    routing_reason: str
    latency_ms: float


# ----------------------------------------------------------------------
# 4. ERROR HANDLING - never expose Python tracebacks to the client
# ----------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    STATE["counts"]["errors"] += 1
    rid = str(uuid.uuid4())
    log_event(event="error", request_id=rid, error_type=type(exc).__name__)   # type only, no details
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error", "request_id": rid})


@app.middleware("http")
async def count_validation_errors(request: Request, call_next):
    """Count 422 responses (invalid input) so they appear in /metrics."""
    response = await call_next(request)
    if response.status_code == 422:
        STATE["counts"]["validation_errors"] += 1
        log_event(event="validation_error", path=request.url.path)
    return response


# ----------------------------------------------------------------------
# 5. ENDPOINTS
# ----------------------------------------------------------------------
@app.get("/")
def root():
    """Serve the EcoRoute AI web interface."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    """Healthy only if all three model files were loaded at startup."""
    ok = all(k in STATE["models"] for k in ("dt", "rf", "router"))
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "healthy" if ok else "unhealthy",
                                 "uptime_s": round(time.time() - STATE["started"], 1)})


@app.post("/predict", response_model=PredictionResponse)
def predict(
    req: EnergyRequest,
    mode: Literal["ecoroute", "dt", "rf"] = Query(
        "ecoroute", description="ecoroute = router decides (default). "
                                "'dt' / 'rf' force one model (used for experiments).")):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Build a one-row table in the SAME column order used during training
    row = pd.DataFrame([[getattr(req, f) for f in C.FEATURES]], columns=C.FEATURES)

    # --- Routing decision ---
    if mode == "dt":
        use_rf, reason = False, "forced: decision tree (experiment mode)"
    elif mode == "rf":
        use_rf, reason = True, "forced: random forest (experiment mode)"
    else:
        use_rf = bool(STATE["models"]["router"].predict(row)[0])      # 0 -> DT, 1 -> RF
        reason = ("router expects the random forest to be meaningfully more accurate"
                  if use_rf else "router expects the lightweight decision tree to be sufficient")

    model_name = "random_forest" if use_rf else "decision_tree"
    prediction = float(STATE["models"]["rf" if use_rf else "dt"].predict(row)[0])

    latency_ms = (time.perf_counter() - t0) * 1000

    # --- Update monitoring state (numbers only) ---
    STATE["counts"]["requests"] += 1
    STATE["counts"]["rf" if use_rf else "dt"] += 1
    STATE["latencies_ms"].append(latency_ms)
    STATE["window"].append({f: getattr(req, f) for f in C.FEATURES})

    # --- Privacy-friendly log line: NO input values ---
    log_event(event="predict", request_id=request_id, model_used=model_name,
              latency_ms=round(latency_ms, 3), status="success", mode=mode)

    return PredictionResponse(request_id=request_id, prediction_wh=round(prediction, 2),
                              model_used=model_name, routing_reason=reason,
                              latency_ms=round(latency_ms, 3))


@app.get("/metrics")
def metrics():
    """Application monitoring snapshot (what you screenshot for the report)."""
    lat = np.array(STATE["latencies_ms"]) if STATE["latencies_ms"] else np.array([0.0])
    c = STATE["counts"]
    served = max(c["dt"] + c["rf"], 1)
    return {
        "requests_total": c["requests"],
        "decision_tree_requests": c["dt"],
        "random_forest_requests": c["rf"],
        "rf_share_percent": round(c["rf"] / served * 100, 1),
        "server_errors": c["errors"],
        "validation_errors_422": c["validation_errors"],
        "avg_latency_ms": round(float(lat.mean()), 3),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 3),
        "uptime_s": round(time.time() - STATE["started"], 1),
    }


@app.get("/drift")
def drift_status(min_requests: int = Query(50, ge=10, le=500)):
    """
    Compare the last (up to 500) requests with the training baseline using PSI.
    Needs at least `min_requests` requests, otherwise the comparison is too noisy.
    """
    n = len(STATE["window"])
    if STATE["baseline"] is None or n < min_requests:
        return {"overall_status": "Insufficient data", "requests_in_window": n,
                "needed": min_requests}
    window = pd.DataFrame(list(STATE["window"]))
    report = drift.drift_report(STATE["baseline"], window)
    report["note"] = "Drift is a warning signal for investigation, not proof of model failure."
    log_event(event="drift_check", overall_status=report["overall_status"],
              max_psi=report["max_psi"])
    return report


if __name__ == "__main__":                  # allows:  python -m api.main
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))