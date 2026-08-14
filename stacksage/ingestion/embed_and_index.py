"""Embed cleaned QA pairs and dual-index into Qdrant + Elasticsearch."""
import json, logging, os, uuid
from pathlib import Path
from elasticsearch import Elasticsearch, helpers
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger     = logging.getLogger(__name__)
IN_JSONL   = Path("data/processed/qa_pairs_clean.jsonl")
BATCH_SIZE = 64
VEC_SIZE   = 384
COLL       = os.getenv("QDRANT_COLLECTION", "stacksage")
ES_IDX     = os.getenv("ES_INDEX", "stacksage")

ES_MAPPING = {
    "settings": {"analysis": {"analyzer": {"en": {"type": "standard", "stopwords": "_english_"}}}},
    "mappings": {"properties": {
        "doc_id":         {"type": "keyword"},
        "question_title": {"type": "text", "analyzer": "en", "boost": 3},
        "answer_prose":   {"type": "text", "analyzer": "en", "boost": 2},
        "question_prose": {"type": "text", "analyzer": "en"},
        "tags_list":      {"type": "keyword"},
        "question_score": {"type": "integer"},
        "answer_score":   {"type": "integer"},
        "creation_date":  {"type": "date"},
        "question_url":   {"type": "keyword", "index": False},
    }},
}

_emb = _qdrant = _es = None

def get_embedder():
    global _emb
    if _emb is None:
        name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loading embedder: %s", name)
        _emb = SentenceTransformer(name)
    return _emb

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=os.getenv("QDRANT_HOST","localhost"),
                               port=int(os.getenv("QDRANT_PORT","6333")))
        existing = [c.name for c in _qdrant.get_collections().collections]
        if COLL not in existing:
            _qdrant.create_collection(COLL, vectors_config=VectorParams(size=VEC_SIZE, distance=Distance.COSINE))
            logger.info("Created Qdrant collection: %s", COLL)
    return _qdrant

def get_es():
    global _es
    if _es is None:
        _es = Elasticsearch(os.getenv("ES_HOST","http://localhost:9200"))
        if not _es.indices.exists(index=ES_IDX):
            _es.indices.create(index=ES_IDX, body=ES_MAPPING)
            logger.info("Created ES index: %s", ES_IDX)
    return _es

def _doc_uuid(doc_id: str) -> str:
    """Deterministic UUID from doc_id hex string."""
    return str(uuid.UUID(bytes=bytes.fromhex(doc_id)))

def index_batch(recs: list[dict], emb, qdrant, es) -> None:
    """Embed + upsert one batch to both stores.  Idempotent."""
    vecs = emb.encode([r["text_for_embedding"] for r in recs],
                      normalize_embeddings=True, show_progress_bar=False)
    # Qdrant
    qdrant.upsert(collection_name=COLL, wait=False, points=[
        PointStruct(id=_doc_uuid(r["doc_id"]), vector=v.tolist(), payload={
            "doc_id": r["doc_id"], "question_id": r["question_id"],
            "question_title": r["question_title"], "question_url": r["question_url"],
            "tags_list": r.get("tags_list",[]), "question_score": r.get("question_score",0),
            "answer_score": r.get("answer_score",0), "creation_date": r.get("creation_date",""),
        }) for r, v in zip(recs, vecs)
    ])
    # Elasticsearch bulk
    helpers.bulk(es, [{
        "_op_type": "index", "_index": ES_IDX, "_id": r["doc_id"],
        **{k: r.get(k) for k in [
            "doc_id","question_id","question_title","answer_prose",
            "question_prose","tags_list","question_score","answer_score",
            "creation_date","question_url"]},
    } for r in recs], raise_on_error=False)

def process_all(in_path: Path = IN_JSONL) -> dict:
    """Read cleaned JSONL, embed in batches, dual-index.
    Returns {"qdrant_docs": int, "es_docs": int}."""
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} missing — run clean.py first")
    emb = get_embedder(); qdrant = get_qdrant(); es = get_es()
    recs = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    logger.info("Indexing %d records (batch=%d)…", len(recs), BATCH_SIZE)
    for i in tqdm(range(0, len(recs), BATCH_SIZE), desc="Indexing"):
        try: index_batch(recs[i:i+BATCH_SIZE], emb, qdrant, es)
        except Exception as e: logger.error("Batch %d error: %s", i//BATCH_SIZE, e)
    es.indices.refresh(index=ES_IDX)
    result = {"qdrant_docs": qdrant.get_collection(COLL).points_count,
              "es_docs": es.count(index=ES_IDX)["count"]}
    logger.info("Done: %s", result); return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from dotenv import load_dotenv; load_dotenv()
    process_all()
