"""
es_retriever.py — Elasticsearch retrieval over the maintenance procedures
=========================================================================
Three ways to rank procedures for a failure-mode description:

  bm25    lexical match (multi_match over title, failure modes, ATA chapter, steps, parts)
  dense   kNN over all-MiniLM-L6-v2 embeddings (dense_vector, cosine, HNSW)
  hybrid  both lists fused with Reciprocal Rank Fusion (RRF)

RRF is done here in Python rather than with Elasticsearch's rrf retriever, so it works on
any Elasticsearch 8.x licence level and the fusion is easy to read and test.

Talks to Elasticsearch over its REST API with `requests` (already a project dependency),
so no extra client library is needed.
"""

import hashlib
import json
from dataclasses import asdict

import requests

from config import ES_API_KEY, ES_INDEX, ES_TIMEOUT_S, ES_URL, RRF_K
from knowledge_base import MaintenanceProcedure

# The emergency procedure is attached by rule (RUL below a threshold), never by ranking,
# exactly as in knowledge_base.retrieve_procedures. It is indexed but filtered out of searches.
EMERGENCY_PROCEDURE_ID = "MRO-006"
CANDIDATES = 10           # how deep each list goes before fusion (the corpus is smaller than this)

BM25_FIELDS = ["title^2", "failure_modes^3", "ata_chapter^2", "steps", "parts"]


def procedure_text(proc: MaintenanceProcedure) -> str:
    """The text that gets embedded for dense search."""
    return (
        f"{proc.title}. Failure modes: {', '.join(proc.failure_modes)}. {proc.ata_chapter}. "
        f"Steps: {' '.join(proc.steps)} Parts: {', '.join(proc.parts_to_preorder)}."
    )


def kb_fingerprint(procedures: list[MaintenanceProcedure], embed_model: str | None) -> str:
    """Changes whenever a procedure or the embedding model changes, so the index is rebuilt."""
    payload = json.dumps([asdict(p) for p in procedures], sort_keys=True) + str(embed_model)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank of d), rank from 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class ElasticsearchRetriever:
    """
    Pass an embedder for dense and hybrid search. Without one the index has no vector
    field and only bm25 is available, which keeps a lexical-only setup dependency-free.
    """

    def __init__(self, url: str = ES_URL, index: str = ES_INDEX, embedder=None,
                 api_key: str | None = ES_API_KEY, timeout: float = ES_TIMEOUT_S, session=None):
        self.url = url.rstrip("/")
        self.index = index
        self.embedder = embedder
        self.timeout = timeout
        self.session = session or requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"ApiKey {api_key}"

    # ── HTTP ──────────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, body=None, ndjson: str | None = None,
                 ok=(200, 201)) -> dict:
        headers = dict(self.headers)
        data = None
        if ndjson is not None:
            headers["Content-Type"] = "application/x-ndjson"
            data = ndjson
        elif body is not None:
            data = json.dumps(body)
        r = self.session.request(method, f"{self.url}{path}", data=data, headers=headers,
                                 timeout=self.timeout)
        if r.status_code not in ok:
            raise RuntimeError(f"Elasticsearch {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else {}

    # ── Indexing ──────────────────────────────────────────────────────────

    def _mapping(self, fingerprint: str) -> dict:
        props = {
            "id": {"type": "keyword"},
            "title": {"type": "text"},
            "ata_chapter": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "failure_modes": {"type": "text"},
            "steps": {"type": "text"},
            "parts": {"type": "text"},
            "severity": {"type": "keyword"},
        }
        if self.embedder is not None:
            props["embedding"] = {
                "type": "dense_vector", "dims": self.embedder.dims, "index": True,
                "similarity": "cosine", "index_options": {"type": "hnsw"},
            }
        return {
            "settings": {
                "number_of_shards": 1, "number_of_replicas": 0,
                "analysis": {"analyzer": {"default": {"type": "english"}}},
            },
            "mappings": {
                "_meta": {"kb_fingerprint": fingerprint,
                          "embed_model": getattr(self.embedder, "name", None)},
                "properties": props,
            },
        }

    def _current_fingerprint(self) -> str | None:
        try:
            resp = self._request("GET", f"/{self.index}/_mapping", ok=(200,))
        except RuntimeError:
            return None
        meta = next(iter(resp.values()), {}).get("mappings", {}).get("_meta", {})
        return meta.get("kb_fingerprint")

    def ensure_index(self, procedures: list[MaintenanceProcedure]) -> bool:
        """Create and fill the index if missing or out of date. Returns True if it (re)built."""
        fingerprint = kb_fingerprint(procedures, getattr(self.embedder, "name", None))
        if self._current_fingerprint() == fingerprint:
            return False
        self._request("DELETE", f"/{self.index}", ok=(200, 404))
        self._request("PUT", f"/{self.index}", self._mapping(fingerprint))

        vectors = None
        if self.embedder is not None:
            vectors = self.embedder.embed([procedure_text(p) for p in procedures])
        lines = []
        for i, p in enumerate(procedures):
            doc = {
                "id": p.id, "title": p.title, "ata_chapter": p.ata_chapter,
                "failure_modes": p.failure_modes, "steps": p.steps,
                "parts": p.parts_to_preorder, "severity": p.severity,
            }
            if vectors is not None:
                doc["embedding"] = vectors[i].tolist()
            lines.append(json.dumps({"index": {"_index": self.index, "_id": p.id}}))
            lines.append(json.dumps(doc))
        resp = self._request("POST", "/_bulk?refresh=true", ndjson="\n".join(lines) + "\n")
        if resp.get("errors"):
            raise RuntimeError(f"Bulk indexing failed: {json.dumps(resp)[:300]}")
        return True

    # ── Search ────────────────────────────────────────────────────────────

    @staticmethod
    def _exclude_emergency() -> dict:
        return {"bool": {"must_not": [{"term": {"id": EMERGENCY_PROCEDURE_ID}}]}}

    def _ids(self, resp: dict) -> list[str]:
        return [h["_source"]["id"] for h in resp.get("hits", {}).get("hits", [])]

    def bm25(self, query: str, size: int = CANDIDATES) -> list[str]:
        body = {
            "size": size,
            "_source": ["id"],
            "query": {"bool": {
                "must": {"multi_match": {"query": query, "fields": BM25_FIELDS,
                                         "type": "best_fields", "tie_breaker": 0.3}},
                "must_not": [{"term": {"id": EMERGENCY_PROCEDURE_ID}}],
            }},
        }
        return self._ids(self._request("POST", f"/{self.index}/_search", body, ok=(200,)))

    def dense(self, query: str, size: int = CANDIDATES) -> list[str]:
        if self.embedder is None:
            raise RuntimeError("Dense search needs an embedder")
        body = {
            "size": size,
            "_source": ["id"],
            "knn": {
                "field": "embedding",
                "query_vector": self.embedder.embed_one(query),
                "k": size,
                "num_candidates": max(50, size),
                "filter": self._exclude_emergency(),
            },
        }
        return self._ids(self._request("POST", f"/{self.index}/_search", body, ok=(200,)))

    def hybrid(self, query: str, size: int = CANDIDATES, rrf_k: int = RRF_K) -> list[str]:
        fused = rrf_fuse([self.bm25(query, CANDIDATES), self.dense(query, CANDIDATES)], k=rrf_k)
        return [doc_id for doc_id, _ in fused[:size]]

    def search(self, query: str, size: int, mode: str = "hybrid") -> list[str]:
        if mode == "bm25":
            return self.bm25(query, size)
        if mode == "dense":
            return self.dense(query, size)
        if mode == "hybrid":
            return self.hybrid(query, size)
        raise ValueError(f"Unknown search mode: {mode!r} (use bm25, dense or hybrid)")
