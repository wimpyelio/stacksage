"""Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)."""
import logging, os
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks BM25+dense candidates with a cross-encoder.  Lazy-loads model."""

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Loading reranker: %s", name)
            self._model = CrossEncoder(name)
        return self._model

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        """Score (query, title + answer_prose) pairs; return top_k sorted desc.
        Adds reranker_score to each doc.  Falls back to docs[:top_k] on error."""
        if not docs:
            return []
        try:
            model = self._get_model()
            pairs = [(query, f"{d.get('question_title','')} {d.get('answer_prose','')}".strip())
                     for d in docs]
            scores = model.predict(pairs)
            for doc, sc in zip(docs, scores):
                doc["reranker_score"] = float(sc)
            ranked = sorted(docs, key=lambda d: d["reranker_score"], reverse=True)
            return ranked[:top_k]
        except Exception as e:
            logger.error("Reranker error: %s", e)
            for d in docs:
                d.setdefault("reranker_score", 0.0)
            return docs[:top_k]
