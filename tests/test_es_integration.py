"""
Live tests against a real Elasticsearch. Skipped unless PROGNOSTIC_TEST_ES_URL is set
(the retrieval-eval CI job sets it). Dense and hybrid tests also need the embedding model
(python embeddings.py --download).
"""

import os

import pytest

from knowledge_base import MAINTENANCE_PROCEDURES

ES_URL = os.getenv("PROGNOSTIC_TEST_ES_URL")
pytestmark = pytest.mark.skipif(not ES_URL, reason="set PROGNOSTIC_TEST_ES_URL to run")


def _embedder_or_skip():
    try:
        from embeddings import MiniLMEmbedder
        return MiniLMEmbedder()
    except (ImportError, FileNotFoundError) as e:
        pytest.skip(f"embedding model not available: {e}")


def test_bm25_index_and_search():
    from es_retriever import ElasticsearchRetriever
    es = ElasticsearchRetriever(url=ES_URL, index="prognosticai-test-bm25", timeout=30)
    es.ensure_index(MAINTENANCE_PROCEDURES)
    assert es.ensure_index(MAINTENANCE_PROCEDURES) is False          # second call is a no-op
    assert es.bm25("ATA 72-30", 5)[0] == "MRO-001"
    assert es.bm25("spectrometric oil analysis", 5)[0] == "MRO-004"
    assert "MRO-006" not in es.bm25("imminent failure critical degradation ground aircraft", 10)


def test_dense_and_hybrid_search():
    from es_retriever import ElasticsearchRetriever
    es = ElasticsearchRetriever(url=ES_URL, index="prognosticai-test-dense",
                                embedder=_embedder_or_skip(), timeout=30)
    es.ensure_index(MAINTENANCE_PROCEDURES)
    dense = es.dense("turbine blades overheating and the ceramic coating is flaking off", 5)
    assert dense[0] == "MRO-002" and "MRO-006" not in dense and len(dense) == 5
    assert es.hybrid("metal particles showing up in the lubricant", 2)[0] == "MRO-004"


def test_agent_dispatcher_against_live_elasticsearch():
    from es_retriever import ElasticsearchRetriever
    from procedure_search import search_procedures
    es = ElasticsearchRetriever(url=ES_URL, index="prognosticai-test-bm25", timeout=30)
    es.ensure_index(MAINTENANCE_PROCEDURES)
    r = search_procedures("Bearing Wear — oil contamination, vibration", 10.0,
                          backend="elasticsearch", retriever=es, mode="bm25")
    assert r.backend == "elasticsearch:bm25" and not r.note
    assert [p.id for p in r.procedures][:2] == ["MRO-006", "MRO-004"]
