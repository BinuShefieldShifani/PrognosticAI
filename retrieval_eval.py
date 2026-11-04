"""
retrieval_eval.py — Measure how well each retriever finds the right procedure
=============================================================================
Runs the labelled queries in eval/procedure_queries.jsonl through:

  keyword   the original retriever in knowledge_base.py (baseline, no Elasticsearch needed)
  bm25      Elasticsearch lexical search
  dense     Elasticsearch kNN over MiniLM embeddings
  hybrid    BM25 + kNN fused with RRF

Each query has one correct procedure (grade 2) and sometimes an acceptable second one
(grade 1). Metrics:

  Hit@1     correct procedure ranked first
  Hit@2     correct procedure in the top 2 (the agent attaches the top 2 to a work order)
  MRR       mean of 1 / rank of the correct procedure
  nDCG@3    ranking quality using both grades

Usage:
    python retrieval_eval.py                                   # keyword only
    python retrieval_eval.py --es-url http://localhost:9200    # all four
    python retrieval_eval.py --es-url ... --gate --out retrieval_metrics.md --json-out r.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from knowledge_base import MAINTENANCE_PROCEDURES, retrieve_procedures

QUERIES_PATH = Path(__file__).parent / "eval" / "procedure_queries.jsonl"
RANK_DEPTH = 5            # every rankable procedure, so the rank of the correct one is always known
CATEGORIES = ["agent_output", "paraphrase", "symptom", "procedure", "reference"]


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Metrics ───────────────────────────────────────────────────────────────

def primary_id(relevant: dict[str, int]) -> str:
    return max(relevant, key=relevant.get)


def rank_of(ranked: list[str], doc_id: str) -> int | None:
    return ranked.index(doc_id) + 1 if doc_id in ranked else None


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 3) -> float:
    dcg = sum((2 ** relevant.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def score_query(ranked: list[str], relevant: dict[str, int]) -> dict:
    r = rank_of(ranked, primary_id(relevant))
    return {
        "rank": r,
        "hit@1": float(r == 1),
        "hit@2": float(r is not None and r <= 2),
        "rr": 1.0 / r if r else 0.0,
        "ndcg@3": ndcg_at_k(ranked, relevant, 3),
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@2": sum(r["hit@2"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "ndcg@3": sum(r["ndcg@3"] for r in rows) / n,
    }


# ── Runner ────────────────────────────────────────────────────────────────

def keyword_ranker(query: str) -> list[str]:
    # rul_cycles=inf keeps the emergency rule out of it: only the ranking is measured
    return [p.id for p in retrieve_procedures(query, rul_cycles=float("inf"), top_k=RANK_DEPTH)]


def evaluate(rankers: dict, queries: list[dict]) -> dict:
    results = {}
    for name, rank_fn in rankers.items():
        per_query = []
        for q in queries:
            ranked = rank_fn(q["query"])
            per_query.append({**q, "ranked": ranked, **score_query(ranked, q["relevant"])})
        by_cat = defaultdict(list)
        for row in per_query:
            by_cat[row["category"]].append(row)
        results[name] = {
            "overall": aggregate(per_query),
            "by_category": {c: aggregate(by_cat[c]) for c in CATEGORIES if by_cat[c]},
            "per_query": per_query,
        }
    return results


def to_markdown(results: dict) -> str:
    lines = ["| Retriever | Hit@1 | Hit@2 | MRR | nDCG@3 |", "|---|---|---|---|---|"]
    for name, r in results.items():
        o = r["overall"]
        lines.append(f"| {name} | {o['hit@1']:.2f} | {o['hit@2']:.2f} | {o['mrr']:.3f} | {o['ndcg@3']:.3f} |")
    lines += ["", "MRR by query type:", "",
              "| Retriever | " + " | ".join(
                  f"{c} ({results[next(iter(results))]['by_category'][c]['n']})" for c in CATEGORIES) + " |",
              "|---" * (len(CATEGORIES) + 1) + "|"]
    for name, r in results.items():
        cells = [f"{r['by_category'][c]['mrr']:.3f}" for c in CATEGORIES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def misses(results: dict, name: str, k: int = 2) -> list[str]:
    out = []
    for row in results[name]["per_query"]:
        if not row["hit@2" if k == 2 else "hit@1"]:
            out.append(f"{row['qid']} [{row['category']}] {row['query']!r} -> {row['ranked'][:3]} "
                       f"(want {primary_id(row['relevant'])})")
    return out


def log_to_mlflow(results: dict, markdown: str):
    import mlflow
    from config import MLFLOW_ARTIFACT_LOCATION, MLFLOW_TRACKING_URI

    experiment = "prognosticai-retrieval"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="procedure-retrieval"):
        mlflow.log_param("n_queries", next(iter(results.values()))["overall"]["n"])
        for name, r in results.items():
            mlflow.log_metrics({f"{name}_{m.replace('@', '_at_')}": v
                                for m, v in r["overall"].items() if m != "n"})
        mlflow.log_text(markdown, "retrieval_metrics.md")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate procedure retrieval")
    p.add_argument("--es-url", help="Elasticsearch URL; without it only the keyword baseline runs")
    p.add_argument("--index", default="prognosticai-procedures-eval")
    p.add_argument("--queries", default=str(QUERIES_PATH))
    p.add_argument("--out", help="write the markdown report here")
    p.add_argument("--json-out", help="write full results (per query) here")
    p.add_argument("--mlflow", action="store_true", help="log the metrics to MLflow")
    p.add_argument("--gate", action="store_true",
                   help="exit 1 if hybrid MRR is below the keyword baseline")
    args = p.parse_args(argv)

    queries = load_queries(Path(args.queries))
    rankers = {"keyword": keyword_ranker}
    if args.es_url:
        from embeddings import MiniLMEmbedder
        from es_retriever import ElasticsearchRetriever

        es = ElasticsearchRetriever(url=args.es_url, index=args.index,
                                    embedder=MiniLMEmbedder(), timeout=30)
        es.ensure_index(MAINTENANCE_PROCEDURES)
        rankers["bm25"] = lambda q: es.bm25(q, RANK_DEPTH)
        rankers["dense"] = lambda q: es.dense(q, RANK_DEPTH)
        rankers["hybrid"] = lambda q: es.hybrid(q, RANK_DEPTH)

    results = evaluate(rankers, queries)
    report = to_markdown(results)
    print(f"{len(queries)} queries\n\n{report}")
    for name in results:
        missed = misses(results, name)
        print(f"\n{name}: {len(missed)} queries with the correct procedure outside the top 2")
        for m in missed:
            print(f"  {m}")

    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.mlflow:
        log_to_mlflow(results, report)

    if args.gate and "hybrid" in results:
        base, hyb = results["keyword"]["overall"]["mrr"], results["hybrid"]["overall"]["mrr"]
        if hyb < base:
            print(f"\nGATE FAILED: hybrid MRR {hyb:.3f} is below keyword baseline {base:.3f}")
            return 1
        print(f"\nGate passed: hybrid MRR {hyb:.3f} >= keyword baseline {base:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
