"""
monitoring/smoke_test.py
------------------------
Security + functionality demo against a running API (local or Cloud Run).
Prints a PASS/FAIL table - take ONE screenshot of the whole output as evidence.

Usage:  python monitoring/smoke_test.py --url http://localhost:8080
        python monitoring/smoke_test.py --url https://YOUR-CLOUD-RUN-URL
"""
import argparse
import json
import urllib.error
import urllib.request

VALID = {"lights": 10, "RH_1": 45.5, "RH_2": 40.2, "RH_3": 41.0, "RH_out": 80,
         "Windspeed": 4.0, "Visibility": 40, "hour": 18, "dayofweek": 2}


def post(url, body):
    """POST JSON, return (status_code, parsed_json). HTTP errors are returned, not raised."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    base = ap.parse_args().url.rstrip("/")

    with urllib.request.urlopen(base + "/health", timeout=30) as r:
        health = (r.status, json.loads(r.read()))

    missing = {k: v for k, v in VALID.items() if k != "RH_2"}
    tests = [
        ("Health check",                      health[0], 200),
        ("Valid input -> prediction",         post(base + "/predict", VALID)[0], 200),
        ("Invalid type (RH_1='hello')",       post(base + "/predict", dict(VALID, RH_1="hello"))[0], 422),
        ("Missing field (RH_2 removed)",      post(base + "/predict", missing)[0], 422),
        ("Out of range (RH_1=500)",           post(base + "/predict", dict(VALID, RH_1=500))[0], 422),
        ("Unexpected field (home_address)",   post(base + "/predict", dict(VALID, home_address="x"))[0], 422),
    ]
    print(f"{'TEST':40s} {'GOT':>5s} {'EXPECTED':>9s}  RESULT")
    for name, got, exp in tests:
        print(f"{name:40s} {got:>5d} {exp:>9d}  {'PASS' if got == exp else 'FAIL'}")
    print("\nSample valid response:", json.dumps(post(base + "/predict", VALID)[1], indent=2))


if __name__ == "__main__":
    main()
