"""
Tests for the optional Elasticsearch retrieval add-on. All offline: Elasticsearch is
replaced by a fake HTTP session, and the embedder by a fixed-vector stub.
The live tests are in test_es_integration.py.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

import procedure_search
import retrieval_eval as RE
from es_retriever import ElasticsearchRetriever, kb_fingerprint, procedure_text, rrf_fuse
from knowledge_base import MAINTENANCE_PROCEDURES, retrieve_procedures

QUERIES = [
    "HPC (High Pressure Compressor) Degradation — blade erosion, tip clearance loss",
    "Bearing Wear — oil contamination, vibration",
    "General Engine Degradation",
    "something with no matching words at all",
]


# ── Fakes ─────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.content = json.dumps(payload).encode()
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records requests; answers searches with a fixed list of ids."""

    def __init__(self, hit_ids=("MRO-001", "MRO-004"), fingerprint=None):
        self.calls = []
        self.hit_ids = list(hit_ids)
        self.fingerprint = fingerprint

    def request(self, method, url, data=None, headers=None, timeout=None):
        body = data if (headers or {}).get("Content-Type") == "application/x-ndjson" else (
            json.loads(data) if data else None)
        self.calls.append((method, url, body))
        if url.endswith("/_mapping"):
            if self.fingerprint is None:
                return FakeResponse(404, {"error": "index_not_found"})
            return FakeResponse(200, {"idx": {"mappings": {"_meta": {"kb_fingerprint": self.fingerprint}}}})
        if url.endswith("/_search"):
            return FakeResponse(200, {"hits": {"hits": [{"_source": {"id": i}} for i in self.hit_ids]}})
        if "/_bulk" in url:
            return FakeResponse(200, {"errors": False})
        return FakeResponse(200, {"acknowledged": True})


class StubEmbedder:
    dims = 4
    name = "stub"

    def embed(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32) / 2.0

    def embed_one(self, text):
        return [0.5, 0.5, 0.5, 0.5]


class FixedRetriever:
    def __init__(self, ranked):
        self.ranked = ranked

    def search(self, query, size, mode):
        return self.ranked


# ── Default behaviour is unchanged ────────────────────────────────────────

@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("rul", [5.0, 14.9, 15.0, 80.0])
def test_default_backend_is_the_original_keyword_retriever(query, rul):
    result = procedure_search.search_procedures(query, rul)
    assert result.backend == "keyword"
    assert [p.id for p in result.procedures] == [p.id for p in retrieve_procedures(query, rul)]


def test_unreachable_elasticsearch_falls_back_to_keyword():
    dead = ElasticsearchRetriever(url="http://127.0.0.1:9", index="x", timeout=0.5)
    result = procedure_search.search_procedures(QUERIES[0], 50.0, backend="elasticsearch",
                                                retriever=dead, mode="bm25")
    assert result.backend == "keyword"
    assert "used keyword retrieval" in result.note
    assert [p.id for p in result.procedures] == [p.id for p in retrieve_procedures(QUERIES[0], 50.0)]


def test_empty_elasticsearch_result_falls_back_to_keyword():
    result = procedure_search.search_procedures(QUERIES[0], 50.0, backend="elasticsearch",
                                                retriever=FixedRetriever([]))
    assert result.backend == "keyword" and result.note


@pytest.mark.parametrize("rul,emergency", [(14.9, True), (15.0, False), (100.0, False)])
def test_emergency_rule_matches_keyword_retriever(rul, emergency):
    result = procedure_search.search_procedures("anything", rul, backend="elasticsearch",
                                                retriever=FixedRetriever(["MRO-004", "MRO-001", "MRO-002"]),
                                                mode="hybrid")
    ids = [p.id for p in result.procedures]
    assert result.backend == "elasticsearch:hybrid"
    assert ids == (["MRO-006"] if emergency else []) + ["MRO-004", "MRO-001"]
    keyword_has_emergency = retrieve_procedures("anything", rul)[0].id == "MRO-006"
    assert keyword_has_emergency == emergency


# ── Elasticsearch requests ────────────────────────────────────────────────

def test_bm25_request_searches_all_fields_and_excludes_emergency():
    s = FakeSession()
    ids = ElasticsearchRetriever(url="http://es", index="idx", session=s).bm25("oil debris", 5)
    method, url, body = s.calls[-1]
    assert ids == ["MRO-001", "MRO-004"]
    assert url == "http://es/idx/_search"
    mm = body["query"]["bool"]["must"]["multi_match"]
    assert mm["query"] == "oil debris" and "failure_modes^3" in mm["fields"]
    assert body["query"]["bool"]["must_not"] == [{"term": {"id": "MRO-006"}}]


def test_dense_request_uses_knn_with_filter():
    s = FakeSession()
    ElasticsearchRetriever(url="http://es", index="idx", embedder=StubEmbedder(), session=s).dense("q", 5)
    knn = s.calls[-1][2]["knn"]
    assert knn["field"] == "embedding" and knn["k"] == 5 and knn["num_candidates"] >= 5
    assert knn["query_vector"] == [0.5, 0.5, 0.5, 0.5]
    assert knn["filter"]["bool"]["must_not"] == [{"term": {"id": "MRO-006"}}]


def test_dense_without_embedder_is_an_error():
    with pytest.raises(RuntimeError):
        ElasticsearchRetriever(url="http://es", session=FakeSession()).dense("q")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        ElasticsearchRetriever(url="http://es", session=FakeSession()).search("q", 2, "fuzzy")


def test_api_key_is_sent():
    r = ElasticsearchRetriever(url="http://es", api_key="abc", session=FakeSession())
    assert r.headers["Authorization"] == "ApiKey abc"


def test_ensure_index_builds_once_with_vectors():
    s = FakeSession()
    r = ElasticsearchRetriever(url="http://es", index="idx", embedder=StubEmbedder(), session=s)
    assert r.ensure_index(MAINTENANCE_PROCEDURES) is True
    put = next(b for m, u, b in s.calls if m == "PUT")
    assert put["mappings"]["properties"]["embedding"]["dims"] == 4
    bulk = next(b for m, u, b in s.calls if "/_bulk" in u)
    docs = [json.loads(line) for line in bulk.strip().split("\n")[1::2]]
    assert len(docs) == len(MAINTENANCE_PROCEDURES) and all(len(d["embedding"]) == 4 for d in docs)

    s2 = FakeSession(fingerprint=kb_fingerprint(MAINTENANCE_PROCEDURES, "stub"))
    r2 = ElasticsearchRetriever(url="http://es", index="idx", embedder=StubEmbedder(), session=s2)
    assert r2.ensure_index(MAINTENANCE_PROCEDURES) is False
    assert not any(m in ("PUT", "DELETE") for m, _, _ in s2.calls)


def test_bm25_only_index_has_no_vector_field():
    s = FakeSession()
    ElasticsearchRetriever(url="http://es", index="idx", session=s).ensure_index(MAINTENANCE_PROCEDURES)
    put = next(b for m, u, b in s.calls if m == "PUT")
    assert "embedding" not in put["mappings"]["properties"]


def test_fingerprint_changes_when_a_procedure_changes():
    edited = [replace(MAINTENANCE_PROCEDURES[0], man_hours=9.0)] + MAINTENANCE_PROCEDURES[1:]
    assert kb_fingerprint(edited, "m") != kb_fingerprint(MAINTENANCE_PROCEDURES, "m")
    assert kb_fingerprint(MAINTENANCE_PROCEDURES, "a") != kb_fingerprint(MAINTENANCE_PROCEDURES, "b")


def test_procedure_text_contains_the_searchable_parts():
    p = MAINTENANCE_PROCEDURES[0]
    text = procedure_text(p)
    assert p.title in text and p.ata_chapter in text and p.failure_modes[0] in text


# ── Fusion and metrics ────────────────────────────────────────────────────

def test_rrf_fuse():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "a"]], k=60)
    scores = dict(fused)
    assert fused[0][0] == "b"
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert rrf_fuse([["x"], ["y"]], k=60)[0][0] == "x"      # tie broken by id, deterministic


def test_rrf_includes_documents_found_by_only_one_list():
    assert {d for d, _ in rrf_fuse([["a"], ["b", "c"]])} == {"a", "b", "c"}


def test_metrics():
    rel = {"B": 2, "C": 1}
    s = RE.score_query(["A", "B", "C"], rel)
    assert s["rank"] == 2 and s["hit@1"] == 0 and s["hit@2"] == 1 and s["rr"] == 0.5
    assert RE.score_query(["B", "C", "A"], rel)["ndcg@3"] == pytest.approx(1.0)
    assert RE.score_query(["A", "D"], rel)["rr"] == 0.0


def test_keyword_ranker_ranks_every_procedure_except_emergency():
    ranked = RE.keyword_ranker("HPC degradation")
    assert len(ranked) == 5 and "MRO-006" not in ranked and ranked[0] == "MRO-001"


# ── The eval set itself ───────────────────────────────────────────────────

def test_eval_queries_are_well_formed():
    queries = RE.load_queries()
    ids = {p.id for p in MAINTENANCE_PROCEDURES} - {"MRO-006"}
    assert len(queries) >= 40
    assert len({q["qid"] for q in queries}) == len(queries)
    assert len({q["query"].lower() for q in queries}) == len(queries)
    for q in queries:
        assert q["category"] in RE.CATEGORIES
        assert set(q["relevant"]) <= ids
        assert sorted(q["relevant"].values(), reverse=True)[0] == 2
        assert list(q["relevant"].values()).count(2) == 1
    assert {RE.primary_id(q["relevant"]) for q in queries} == ids     # every procedure is tested


def test_eval_runs_keyword_baseline():
    results = RE.evaluate({"keyword": RE.keyword_ranker}, RE.load_queries())
    o = results["keyword"]["overall"]
    assert 0 < o["mrr"] <= 1 and set(results["keyword"]["by_category"]) == set(RE.CATEGORIES)
    assert "| keyword |" in RE.to_markdown(results)
