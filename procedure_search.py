"""
procedure_search.py — Picks the procedure retriever the agent uses
==================================================================
PROGNOSTIC_RETRIEVER=keyword (default) calls knowledge_base.retrieve_procedures unchanged.
PROGNOSTIC_RETRIEVER=elasticsearch ranks procedures in Elasticsearch (PROGNOSTIC_ES_MODE:
bm25, dense or hybrid). Any Elasticsearch problem (down, timeout, missing model files)
falls back to keyword retrieval, so the agent always gets procedures.

The emergency rule is the same in both paths: below 15 cycles of RUL the emergency
shutdown procedure (MRO-006) goes first, then the ranked procedures.
"""

from dataclasses import dataclass, field

import config
from knowledge_base import MAINTENANCE_PROCEDURES, MaintenanceProcedure, retrieve_procedures

EMERGENCY_RUL_CYCLES = 15     # same threshold as knowledge_base.retrieve_procedures

_BY_ID = {p.id: p for p in MAINTENANCE_PROCEDURES}
_es_retriever = None
_embedder = None


@dataclass
class ProcedureSearchResult:
    procedures: list[MaintenanceProcedure]
    backend: str                         # "keyword" or "elasticsearch:<mode>"
    note: str = ""                       # set when a fallback happened
    ranked_ids: list[str] = field(default_factory=list)


def get_es_retriever():
    """Build the Elasticsearch retriever once, and make sure the index is current."""
    global _es_retriever
    if _es_retriever is None:
        from es_retriever import ElasticsearchRetriever

        global _embedder
        if config.ES_SEARCH_MODE in ("dense", "hybrid") and _embedder is None:
            from embeddings import MiniLMEmbedder
            _embedder = MiniLMEmbedder()          # loaded once, even if Elasticsearch is down
        retriever = ElasticsearchRetriever(embedder=_embedder)
        retriever.ensure_index(MAINTENANCE_PROCEDURES)
        _es_retriever = retriever
    return _es_retriever


def search_procedures(failure_mode: str, rul_cycles: float, top_k: int = 2,
                      backend: str | None = None, retriever=None,
                      mode: str | None = None) -> ProcedureSearchResult:
    backend = backend or config.RETRIEVER
    if backend != "elasticsearch":
        return ProcedureSearchResult(retrieve_procedures(failure_mode, rul_cycles, top_k), "keyword")

    mode = mode or config.ES_SEARCH_MODE
    try:
        es = retriever or get_es_retriever()
        ranked = es.search(failure_mode, top_k, mode)
        if not ranked:
            raise LookupError("no matching procedures")
        procedures = []
        if rul_cycles < EMERGENCY_RUL_CYCLES:
            procedures.append(_BY_ID["MRO-006"])
        procedures += [_BY_ID[i] for i in ranked[:top_k] if i in _BY_ID]
        return ProcedureSearchResult(procedures, f"elasticsearch:{mode}", ranked_ids=ranked)
    except Exception as e:                                    # noqa: BLE001 - any failure falls back
        note = f"Elasticsearch retrieval failed ({type(e).__name__}: {str(e)[:160]}); used keyword retrieval"
        return ProcedureSearchResult(retrieve_procedures(failure_mode, rul_cycles, top_k),
                                     "keyword", note=note)
