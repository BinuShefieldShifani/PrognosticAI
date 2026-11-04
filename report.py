"""
report.py — Model comparison report from the MLflow registry
=============================================================
  python report.py                     # writes reports/model_comparison.md (+ PNG if matplotlib is installed)

Lists every registered version with its metrics and aliases, the latest
promotion decision, and plots the validation curves of the newest run of
each architecture.
"""

import argparse
import json
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from config import MLFLOW_REGISTERED_MODEL, MLFLOW_TRACKING_URI


def registry_table(name: str = MLFLOW_REGISTERED_MODEL) -> list[dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        aliases = client.get_registered_model(name).aliases
    except MlflowException:
        return []
    by_version = {}
    for alias, v in aliases.items():
        by_version.setdefault(str(v), []).append(alias)
    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        t = mv.tags
        rows.append({
            "version": int(mv.version),
            "model_type": t.get("model_type", "?"),
            "val_rmse": float(t.get("val_rmse", "nan")),
            "test_rmse": float(t.get("test_rmse", "nan")),
            "nasa_score": float(t.get("nasa_score", "nan")),
            "passed_gate": t.get("passed_gate") == "true",
            "aliases": ", ".join(sorted(by_version.get(str(mv.version), []))) or "-",
            "run_id": mv.run_id,
        })
    return sorted(rows, key=lambda r: r["version"])


def plot_curves(rows: list[dict], path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    latest = {}
    for r in rows:
        latest[r["model_type"]] = r          # rows are sorted, so the newest wins
    if not latest:
        return None
    client = MlflowClient()
    fig, axes = plt.subplots(1, len(latest), figsize=(6 * len(latest), 4), squeeze=False)
    for ax, (kind, r) in zip(axes[0], sorted(latest.items())):
        hist = client.get_metric_history(r["run_id"], "val_rmse_curve")
        ax.plot([m.step for m in hist], [m.value for m in hist])
        ax.set_title(f"{kind} v{r['version']} (best val RMSE {r['val_rmse']:.2f})")
        ax.set_xlabel("boosting round" if kind == "xgboost" else "epoch")
        ax.set_ylabel("validation RMSE (cycles)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_report(out_dir: str = "reports", decision: dict | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = registry_table()
    lines = ["# PrognosticAI model comparison", "",
             "Promotion is decided on validation RMSE (held-out training engines). "
             "Test RMSE is reported and used only as a quality gate.", "",
             "| version | model | val RMSE | test RMSE | NASA score | passed gate | aliases |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| v{r['version']} | {r['model_type']} | {r['val_rmse']:.2f} | {r['test_rmse']:.2f} | "
                     f"{r['nasa_score']:.1f} | {'yes' if r['passed_gate'] else 'no'} | {r['aliases']} |")
    if not rows:
        lines.append("| - | no registered versions yet | | | | | |")
    if decision:
        from registry import format_decision
        lines += ["", "## Latest promotion decision", "", format_decision(decision)]
    png = plot_curves(rows, os.path.join(out_dir, "validation_curves.png"))
    if png:
        lines += ["", "![Validation curves](validation_curves.png)"]
    path = os.path.join(out_dir, "model_comparison.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "registry.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Write the model comparison report")
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args(argv)
    path = write_report(args.out_dir)
    print(open(path).read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
