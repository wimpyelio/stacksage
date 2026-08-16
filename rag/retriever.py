"""Hybrid BM25 (rank_bm25) + Dense (Qdrant local) with RRF fusion."""
import logging, os, pickle
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, Range
from sentence_transformers import SentenceTransformer

logger   = logging.getLogger(__name__)
COLL     = os.getenv("QDRANT_COLLECTION","stacksage")
BM25_PATH = Path("data/bm25_index.pkl")


class HybridRetriever:
    """BM25 (in-process) + Dense (Qdrant local) + RRF. Zero servers needed."""

    def __init__(self, qdrant: QdrantClient, embedder: SentenceTransformer) -> None:
        self.qdrant   = qdrant
        self.embedder = embedder
        self._bm25    = None
        self._records = None

    def _load_bm25(self):
        if self._bm25 is None:
            if not BM25_PATH.exists():
                raise FileNotFoundError(f"{BM25_PATH} missing — run embed_and_index.py first")
            with BM25_PATH.open("rb") as f:
                data = pickle.load(f)
            self._bm25    = data["bm25"]
            self._records = data["records"]
            logger.info("Loaded BM25 index: %d docs", len(self._records))

    def search_bm25(self, query: str, tags: list[str], min_score: int,
                    after_date: Optional[str], top_k: int) -> list[dict]:
        """Pure-Python BM25 search. Filters applied post-retrieval."""
        try:
            self._load_bm25()
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            results = []
            for idx, sc in ranked:
                if len(results) >= top_k: break
                r = self._records[idx]
                if r.get("question_score",0) < min_score: continue
                if tags and not any(t in r.get("tags_list",[]) for t in tags): continue
                if after_date and r.get("creation_date","") < after_date: continue
                results.append({**r, "score": float(sc)})
            return results
        except Exception as e:
            logger.error("BM25 error: %s", e); return []

    def search_dense(self, query: str, tags: list[str], min_score: int,
                     after_date: Optional[str], top_k: int) -> list[dict]:
        """Qdrant local ANN search."""
        vec   = self.embedder.encode(query, normalize_embeddings=True).tolist()
        must  = [FieldCondition(key="question_score", range=Range(gte=min_score))]
        if tags: must.append(FieldCondition(key="tags_list", match=MatchAny(any=tags)))
        try:
            hits = self.qdrant.search(collection_name=COLL, query_vector=vec,
                                      query_filter=Filter(must=must) if must else None,
                                      limit=top_k, with_payload=True)
            return [{**h.payload, "score": h.score} for h in hits]
        except Exception as e:
            logger.error("Dense error: %s", e); return []

    def reciprocal_rank_fusion(self, *lists: list[dict], k: int = 60) -> list[dict]:
        scores: dict[str,float] = {}
        meta:   dict[str,dict]  = {}
        for ranked in lists:
            for rank, doc in enumerate(ranked, 1):
                did = doc["doc_id"]
                scores[did] = scores.get(did,0.0) + 1.0/(k+rank)
                meta.setdefault(did, doc)
        return [{**meta[did], "retrieval_score": scores[did]}
                for did in sorted(scores, key=scores.__getitem__, reverse=True)]

    def retrieve(self, queries: list[str], tags: Optional[list[str]]=None,
                 min_vote_score: int=5, after_date: Optional[str]=None,
                 top_k: int=40) -> list[dict]:
        tags = tags or []
        all_lists = []
        for q in queries:
            all_lists.append(self.search_bm25(q, tags, min_vote_score, after_date, top_k))
            all_lists.append(self.search_dense(q, tags, min_vote_score, after_date, top_k))
        return self.reciprocal_rank_fusion(*all_lists)[:top_k]
