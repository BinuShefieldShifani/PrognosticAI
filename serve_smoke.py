"""
serve_smoke.py — Check a running PrognosticAI API end to end
============================================================
Sends real FD001 test engines to the API and checks the answers. Used by CI against
the Docker container, and handy after any deployment.

    python serve_smoke.py --url http://localhost:8000
    python serve_smoke.py --url http://localhost:8000 --compare-local serving/champion

--compare-local loads the same package in this Python process and checks that the
API returns the same RUL values, so the container is proven to serve the same model.
"""

import argparse
import sys
import time

import numpy as np
import requests

from data_processor import load_raw


def wait_until_healthy(url: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
            last = r.text
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(2)
    raise TimeoutError(f"API not healthy after {timeout:.0f}s: {last}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Smoke-test a running PrognosticAI API")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--engines", type=int, default=5, help="how many FD001 test engines to send")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--compare-local", help="path of the same packaged model, to compare predictions")
    args = p.parse_args(argv)
    url = args.url.rstrip("/")
    failures = []

    health = wait_until_healthy(url, args.timeout)
    print(f"health: {health}")
    print(f"model:  {requests.get(f'{url}/model', timeout=10).json()}")

    raw = load_raw("test")
    units = sorted(raw["unit"].unique())[: args.engines]
    sample = raw[raw["unit"].isin(units)]
    payload = {"cycles": sample.to_dict(orient="records")}

    started = time.perf_counter()
    r = requests.post(f"{url}/predict", json=payload, timeout=60)
    ms = (time.perf_counter() - started) * 1000
    if r.status_code != 200:
        print(f"FAIL /predict returned {r.status_code}: {r.text[:300]}")
        return 1
    preds = r.json()["predictions"]
    print(f"\n/predict: {len(sample)} cycles, {len(units)} engines, {ms:.0f} ms, "
          f"model {r.headers.get('X-Model-Type')} v{r.headers.get('X-Model-Version')}")
    print(f"{'unit':>5} {'last_cycle':>10} {'rul':>8} {'band':>17} short_history")
    for row in preds:
        print(f"{row['unit']:>5} {row['last_cycle']:>10} {row['rul']:>8.1f} "
              f"{row['rul_lower']:>7.1f} – {row['rul_upper']:<7.1f} {row['short_history']}")

    if [row["unit"] for row in preds] != [int(u) for u in units]:
        failures.append("one prediction per engine, in unit order")
    if not all(row["rul"] >= 0 for row in preds):
        failures.append("RUL must be non-negative")

    if args.compare_local:
        from rul_predictor import load_predictor
        local = load_predictor(args.compare_local, fallback=False).predict(sample)
        diff = np.abs(local["rul"].values - np.array([row["rul"] for row in preds])).max()
        print(f"\nmax |API - local| RUL difference: {diff:.6f} cycles")
        if diff > 1e-3:
            failures.append(f"API and local package disagree by {diff:.4f} cycles")

    bad = sample.drop(columns=["s4"])
    r = requests.post(f"{url}/predict", json={"cycles": bad.to_dict(orient="records")}, timeout=60)
    print(f"\nmissing sensor s4 -> {r.status_code}")
    if r.status_code != 422 or "s4" not in r.text:
        failures.append(f"a request without s4 should be rejected with 422 naming s4, got {r.status_code}")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nAll smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
