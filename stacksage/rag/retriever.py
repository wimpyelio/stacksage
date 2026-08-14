"""Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion."""
import logging, os
from typing import Optional
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range
from sentence_transformers import SentenceTransformer

logger  = logging.getLogger(__name__)
COLL    = os.getenv("QDRANT_COLLECTION", "stacksage")
ES_IDX  = os.getenv("ES_INDEX", "stacksage")


class HybridRetriever:
    """BM25 (Elasticsearch) + Dense (Qdrant) with RRF fusion."""

    def __init__(self, qdrant: QdrantClient, es: Elasticsearch,
                 embedder: SentenceTransformer) -> None:
        self.qdrant   = qdrant
        self.es       = es
        self.embedder = embedder

    # ── BM25 ──────────────────────────────────────────────────────────────────
    def search_bm25(self, query: str, tags: list[str], min_score: int,
                    after_date: Optional[str], top_k: int) -> list[dict]:
        filters = [{"range": {"question_score": {"gte": min_score}}}]
        if tags:
            filters.append({"terms": {"tags_list": tags}})
        if after_date:
            filters.append({"range": {"creation_date": {"gte": after_date}}})
        body = {
            "size": top_k,
            "query": {"bool": {
                "must": {"multi_match": {
                    "query": query,
                    "fields": ["question_title^3", "answer_prose^2", "question_prose"],
                    "type":  "best_fields",
                }},
                "filter": filters,
            }},
        }
        try:
            hits = self.es.search(index=ES_IDX, body=body)["hits"]["hits"]
            return [{"doc_id": h["_source"]["doc_id"],
                     "score":  h["_score"],
                     **h["_source"]} for h in hits]
        except Exception as e:
            logger.error("BM25 error: %s", e); return []

    # ── Dense ─────────────────────────────────────────────────────────────────
    def search_dense(self, query: str, tags: list[str], min_score: int,
                     after_date: Optional[str], top_k: int) -> list[dict]:
        vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        must = [FieldCondition(key="question_score", range=Range(gte=min_score))]
        if tags:
            must.append(FieldCondition(key="tags_list", match=MatchAny(any=tags)))
        qfilter = Filter(must=must) if must else None
        try:
            hits = self.qdrant.search(collection_name=COLL, query_vector=vec,
                                      query_filter=qfilter, limit=top_k,
                                      with_payload=True)
            return [{"doc_id": h.payload["doc_id"],
                     "score":  h.score,
                     **h.payload} for h in hits]
        except Exception as e:
            logger.error("Dense search error: %s", e); return []

    # ── RRF ───────────────────────────────────────────────────────────────────
    def reciprocal_rank_fusion(self, *lists: list[dict], k: int = 60) -> list[dict]:
        """RRF: score = Σ 1/(k + rank).  Deduplicates by doc_id."""
        scores: dict[str, float] = {}
        meta:   dict[str, dict]  = {}
        for ranked in lists:
            for rank, doc in enumerate(ranked, start=1):
                did = doc["doc_id"]
                scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
                meta.setdefault(did, doc)
        return [{**meta[did], "retrieval_score": scores[did]}
                for did in sorted(scores, key=scores.__getitem__, reverse=True)]

    # ── Public ────────────────────────────────────────────────────────────────
    def retrieve(self, queries: list[str], tags: Optional[list[str]] = None,
                 min_vote_score: int = 5, after_date: Optional[str] = None,
                 top_k: int = 40) -> list[dict]:
        """Run BM25 + Dense for every sub-query, fuse with RRF, return top_k."""
        tags = tags or []
        all_lists = []
        for q in queries:
            all_lists.append(self.search_bm25(q, tags, min_vote_score, after_date, top_k))
            all_lists.append(self.search_dense(q, tags, min_vote_score, after_date, top_k))
        fused = self.reciprocal_rank_fusion(*all_lists)
        logger.debug("RRF: %d unique docs from %d lists", len(fused), len(all_lists))
        return fused[:top_k]
