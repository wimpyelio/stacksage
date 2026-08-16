"""Embed + index using local Qdrant (no server) + rank_bm25 (no ES)."""
import json, logging, os, pickle, uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger     = logging.getLogger(__name__)
IN_JSONL   = Path("data/processed/qa_pairs_clean.jsonl")
QDRANT_DIR = Path("data/qdrant_storage")
BM25_PATH  = Path("data/bm25_index.pkl")
BATCH_SIZE = 64
VEC_SIZE   = 384
COLL       = os.getenv("QDRANT_COLLECTION","stacksage")

_emb = _qdrant = None

def get_embedder():
    global _emb
    if _emb is None:
        name = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loading embedder: %s", name)
        _emb = SentenceTransformer(name)
    return _emb

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        _qdrant = QdrantClient(path=str(QDRANT_DIR))
        existing = [c.name for c in _qdrant.get_collections().collections]
        if COLL not in existing:
            _qdrant.create_collection(COLL,
                vectors_config=VectorParams(size=VEC_SIZE, distance=Distance.COSINE))
            logger.info("Created Qdrant collection: %s (local)", COLL)
    return _qdrant

def _doc_uuid(doc_id: str) -> str:
    return str(uuid.UUID(bytes=bytes.fromhex(doc_id)))

def process_all(in_path: Path = IN_JSONL) -> dict:
    """Embed + dual-index (Qdrant local + BM25 pickle). No servers needed."""
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} missing — run clean.py first")
    emb    = get_embedder()
    qdrant = get_qdrant()
    recs   = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    logger.info("Indexing %d records…", len(recs))

    # ── Qdrant upsert in batches ──────────────────────────────────────────────
    for i in tqdm(range(0, len(recs), BATCH_SIZE), desc="Qdrant"):
        batch = recs[i:i+BATCH_SIZE]
        vecs  = emb.encode([r["text_for_embedding"] for r in batch],
                           normalize_embeddings=True, show_progress_bar=False)
        qdrant.upsert(collection_name=COLL, wait=True, points=[
            PointStruct(id=_doc_uuid(r["doc_id"]), vector=v.tolist(), payload={
                "doc_id":         r["doc_id"],
                "question_id":    r["question_id"],
                "question_title": r["question_title"],
                "question_url":   r["question_url"],
                "answer_prose":   r.get("answer_prose",""),
                "tags_list":      r.get("tags_list",[]),
                "question_score": r.get("question_score",0),
                "answer_score":   r.get("answer_score",0),
                "creation_date":  r.get("creation_date",""),
            }) for r,v in zip(batch, vecs)
        ])

    # ── BM25 index (save to pickle) ───────────────────────────────────────────
    logger.info("Building BM25 index…")
    corpus = [
        (r.get("question_title","") + " " + r.get("answer_prose","")).lower().split()
        for r in recs
    ]
    bm25 = BM25Okapi(corpus)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "records": recs}, f)
    logger.info("BM25 index saved → %s", BM25_PATH)

    result = {"qdrant_docs": len(recs), "bm25_docs": len(recs)}
    logger.info("Done: %s", result)
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from dotenv import load_dotenv; load_dotenv()
    process_all()
