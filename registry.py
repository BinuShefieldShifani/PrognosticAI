"""
registry.py — Model registry and champion/challenger promotion
===============================================================
One registered model, `prognosticai-rul`, holds every architecture.

  register_run(...)   new version from a training run, tagged with its metrics,
                      alias  challenger-<model_type>  moved to it
  promote(...)        compares the challengers with the current champion and
                      moves the  champion  alias if one is clearly better

Promotion rules (in this order):
  1. A version is only eligible if its test RMSE is under its quality gate.
     The test set is used here as a regression floor, not to rank models.
  2. Candidates are ranked by VALIDATION RMSE (engines held out from training,
     same split for every architecture). Choosing on test RMSE would leak the
     test set into model selection.
  3. With no champion yet, the best eligible candidate becomes champion.
  4. Otherwise a challenger must beat the champion by PROMOTION_MIN_GAIN cycles.
     Ties go to the incumbent, so run-to-run noise does not flip the champion.
"""

from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from config import (
    CHALLENGER_ALIAS_PREFIX, CHAMPION_ALIAS, MLFLOW_REGISTERED_MODEL,
    MLFLOW_TRACKING_URI, PROMOTION_MIN_GAIN,
)


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def register_run(run_id: str, model_type: str, metrics: dict, rmse_gate: float,
                 name: str = MLFLOW_REGISTERED_MODEL) -> str:
    """Register runs:/<run_id>/model as a new version and make it the challenger for its type."""
    client = _client()
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    tags = {
        "model_type": model_type,
        "val_rmse": f"{metrics['val_rmse']:.4f}",
        "test_rmse": f"{metrics['test_rmse']:.4f}",
        "nasa_score": f"{metrics['nasa_score']:.2f}",
        "rmse_gate": f"{rmse_gate:.2f}",
        "passed_gate": str(metrics["test_rmse"] <= rmse_gate).lower(),
    }
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, v)
    client.set_registered_model_alias(name, f"{CHALLENGER_ALIAS_PREFIX}{model_type}", mv.version)
    print(f"Registered {name} v{mv.version} ({model_type}) as {CHALLENGER_ALIAS_PREFIX}{model_type}")
    return mv.version


def _describe(client: MlflowClient, name: str, version: str, role: str) -> dict:
    mv = client.get_model_version(name, version)
    t = mv.tags
    return {
        "version": str(mv.version),
        "role": role,
        "model_type": t.get("model_type", "unknown"),
        "val_rmse": float(t["val_rmse"]),
        "test_rmse": float(t["test_rmse"]),
        "nasa_score": float(t["nasa_score"]),
        "passed_gate": t.get("passed_gate") == "true",
        "run_id": mv.run_id,
    }


def candidates(name: str = MLFLOW_REGISTERED_MODEL) -> tuple[dict | None, list[dict]]:
    """Current champion (or None) and the challenger versions, one per model type."""
    client = _client()
    try:
        aliases = client.get_registered_model(name).aliases
    except MlflowException:
        return None, []
    champion = None
    if CHAMPION_ALIAS in aliases:
        champion = _describe(client, name, aliases[CHAMPION_ALIAS], "champion")
    challengers = [
        _describe(client, name, v, alias)
        for alias, v in sorted(aliases.items())
        if alias.startswith(CHALLENGER_ALIAS_PREFIX) and (champion is None or str(v) != champion["version"])
    ]
    return champion, challengers


def promote(min_gain: float = PROMOTION_MIN_GAIN, dry_run: bool = False,
            name: str = MLFLOW_REGISTERED_MODEL) -> dict:
    """Decide whether a challenger replaces the champion. Returns the decision as a dict."""
    champion, challengers = candidates(name)
    eligible = sorted((c for c in challengers if c["passed_gate"]), key=lambda c: c["val_rmse"])
    best = eligible[0] if eligible else None

    if best is None:
        promoted, reason = None, "no eligible challenger (none registered, or all failed their RMSE gate)"
    elif champion is None:
        promoted, reason = best, "no champion yet; best eligible challenger by validation RMSE"
    elif best["val_rmse"] <= champion["val_rmse"] - min_gain:
        gain = champion["val_rmse"] - best["val_rmse"]
        promoted, reason = best, f"beats champion by {gain:.2f} val RMSE cycles (needs {min_gain:.2f})"
    else:
        gain = champion["val_rmse"] - best["val_rmse"]
        who = f"best challenger ({best['model_type']} v{best['version']})"
        how = (f"is {-gain:.2f} cycles worse on validation RMSE" if gain < 0 else
               f"is only {gain:.2f} cycles better, below the {min_gain:.2f} margin")
        promoted, reason = None, f"{who} {how}; champion kept"

    decision = {
        "promoted": promoted is not None,
        "champion_before": champion,
        "champion_after": promoted or champion,
        "challengers": challengers,
        "min_gain": min_gain,
        "reason": reason,
        "dry_run": dry_run,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if promoted and not dry_run:
        client = _client()
        client.set_registered_model_alias(name, CHAMPION_ALIAS, promoted["version"])
        client.set_model_version_tag(name, promoted["version"], "promoted_at", decision["decided_at"])
        client.set_model_version_tag(name, promoted["version"], "promotion_reason", reason)
    return decision


def format_decision(decision: dict) -> str:
    rows = [r for r in [decision["champion_before"]] + decision["challengers"] if r]
    lines = [
        "| version | role | model | val RMSE | test RMSE | NASA score | passed gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| v{r['version']} | {r['role']} | {r['model_type']} | {r['val_rmse']:.2f} | "
                     f"{r['test_rmse']:.2f} | {r['nasa_score']:.1f} | {'yes' if r['passed_gate'] else 'no'} |")
    after = decision["champion_after"]
    verdict = ("PROMOTED" if decision["promoted"] else "NO CHANGE") + (" (dry run)" if decision["dry_run"] else "")
    lines += ["", f"**Decision: {verdict}.** {decision['reason']}"]
    if after:
        lines.append(f"Champion: {after['model_type']} v{after['version']} "
                     f"(val RMSE {after['val_rmse']:.2f}, test RMSE {after['test_rmse']:.2f})")
    return "\n".join(lines)
