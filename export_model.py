"""
export_model.py — Copy a packaged model to a folder the API container can mount
===============================================================================
The registry (mlflow.db + mlruns/) stores absolute paths from the machine that
trained the model, so it does not travel well into a container. This script copies
one packaged model out of it into a plain folder, plus a small serving_info.json
saying which version it is and how it scored.

    python export_model.py                                    # registry champion
    python export_model.py --uri models/rul_predictor         # a local package
    python export_model.py --out serving/champion

Then:
    docker run -p 8000:8000 -v "$PWD/serving/champion:/models/champion:ro" prognosticai-api
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import mlflow

from config import MLFLOW_TRACKING_URI, PREDICTOR_URI

DEFAULT_OUT = os.path.join("serving", "champion")


def _registry_meta(uri: str) -> dict:
    if not uri.startswith("models:/") or "@" not in uri:
        return {}
    name, alias = uri[len("models:/"):].split("@", 1)
    mv = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    meta = {"version": str(mv.version), "registered_model": name, "alias": alias}
    for k in ("model_type", "val_rmse", "test_rmse", "nasa_score"):
        if k in mv.tags:
            meta[k] = mv.tags[k] if k == "model_type" else float(mv.tags[k])
    return meta


def export(uri: str = PREDICTOR_URI, out: str = DEFAULT_OUT) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with tempfile.TemporaryDirectory() as tmp:
        if os.path.isdir(uri):
            src = os.path.join(tmp, "model")
            shutil.copytree(uri, src)
        else:
            src = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp)
        if not os.path.exists(os.path.join(src, "MLmodel")):
            raise FileNotFoundError(f"'{uri}' is not a packaged MLflow model (no MLmodel file)")
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        shutil.copytree(src, out)

    meta = {"source": uri, "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_registry_meta(uri)}
    with open(os.path.join(out, "serving_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export a packaged model for the API container")
    p.add_argument("--uri", default=PREDICTOR_URI,
                   help="models:/<name>@<alias>, runs:/<id>/model or a local folder (default: champion)")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        meta = export(args.uri, args.out)
    except Exception as exc:
        print(f"Export failed: {type(exc).__name__}: {exc}")
        if args.uri.startswith("models:/"):
            print("No champion yet? Run  python run_pipeline.py  first, "
                  "or export a local package with  --uri models/rul_predictor")
        return 1
    print(f"Exported {args.uri} -> {args.out}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
