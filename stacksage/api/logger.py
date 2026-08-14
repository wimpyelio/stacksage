"""PostgreSQL query + feedback logger."""
import json, logging, os
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_DDL = ["""
CREATE TABLE IF NOT EXISTS query_logs (
    id                  SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    timestamp           TIMESTAMPTZ DEFAULT now(),
    question            TEXT,
    rewritten_queries   TEXT[],
    retrieved_doc_ids   TEXT[],
    final_doc_ids       TEXT[],
    answer              TEXT,
    judge_relevance     SMALLINT,
    judge_accuracy      SMALLINT,
    judge_completeness  SMALLINT,
    judge_reasoning     TEXT,
    rewrite_ms          INT,
    retrieval_ms        INT,
    reranker_ms         INT,
    generation_ms       INT,
    judge_ms            INT,
    total_ms            INT,
    user_feedback       SMALLINT
)""", """
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id        SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT now(),
    event     TEXT,
    details   JSONB
)"""]

_INSERT = text("""
INSERT INTO query_logs
  (session_id, question, rewritten_queries, retrieved_doc_ids, final_doc_ids,
   answer, judge_relevance, judge_accuracy, judge_completeness, judge_reasoning,
   rewrite_ms, retrieval_ms, reranker_ms, generation_ms, judge_ms, total_ms)
VALUES
  (:session_id, :question, :rewritten_queries, :retrieved_doc_ids, :final_doc_ids,
   :answer, :judge_relevance, :judge_accuracy, :judge_completeness, :judge_reasoning,
   :rewrite_ms, :retrieval_ms, :reranker_ms, :generation_ms, :judge_ms, :total_ms)
""")

_FEEDBACK = text("""
UPDATE query_logs SET user_feedback = :fb WHERE session_id = :sid
""")

_METRICS = text("""
SELECT
    count(*)                                                        AS total_queries,
    round(avg(judge_relevance)::numeric,    2)                     AS avg_relevance,
    round(avg(judge_accuracy)::numeric,     2)                     AS avg_accuracy,
    round(avg(judge_completeness)::numeric, 2)                     AS avg_completeness,
    round(avg(total_ms)::numeric,           0)                     AS avg_latency_ms,
    round(100.0 * sum(CASE WHEN user_feedback=1 THEN 1 ELSE 0 END)
          / NULLIF(count(user_feedback),0), 1)                     AS feedback_pct
FROM query_logs
""")


class QueryLogger:
    """Log RAG results + feedback to PostgreSQL.  Never raises."""

    def __init__(self) -> None:
        url = (
            f"postgresql+psycopg2://"
            f"{os.getenv('POSTGRES_USER','stacksage')}:"
            f"{os.getenv('POSTGRES_PASSWORD','stacksage')}@"
            f"{os.getenv('POSTGRES_HOST','localhost')}:"
            f"{os.getenv('POSTGRES_PORT','5432')}/"
            f"{os.getenv('POSTGRES_DB','stacksage')}"
        )
        self._engine = create_engine(url, pool_pre_ping=True)
        with self._engine.begin() as conn:
            for ddl in _DDL:
                conn.execute(text(ddl))
        logger.info("QueryLogger ready")

    def log_query(self, session_id: str, question: str, result: dict) -> None:
        js  = result.get("judge_scores", {}) or {}
        tim = result.get("timings", {})
        src = result.get("sources", [])
        cands = result.get("_candidates", [])  # set by pipeline if needed
        try:
            with self._engine.begin() as conn:
                conn.execute(_INSERT, {
                    "session_id":         session_id,
                    "question":           question,
                    "rewritten_queries":  result.get("rewritten_queries", []),
                    "retrieved_doc_ids":  [d.get("doc_id") for d in cands],
                    "final_doc_ids":      [d.get("doc_id") for d in src],
                    "answer":             result.get("answer",""),
                    "judge_relevance":    js.get("relevance"),
                    "judge_accuracy":     js.get("accuracy"),
                    "judge_completeness": js.get("completeness"),
                    "judge_reasoning":    js.get("reasoning"),
                    "rewrite_ms":         tim.get("rewrite_ms"),
                    "retrieval_ms":       tim.get("retrieval_ms"),
                    "reranker_ms":        tim.get("reranker_ms"),
                    "generation_ms":      tim.get("generation_ms"),
                    "judge_ms":           tim.get("judge_ms"),
                    "total_ms":           tim.get("total_ms"),
                })
        except Exception as e:
            logger.error("log_query failed: %s", e)

    def log_feedback(self, session_id: str, feedback: int) -> None:
        try:
            with self._engine.begin() as conn:
                conn.execute(_FEEDBACK, {"fb": feedback, "sid": session_id})
        except Exception as e:
            logger.error("log_feedback failed: %s", e)

    def get_metrics(self) -> dict:
        try:
            with self._engine.connect() as conn:
                row = conn.execute(_METRICS).mappings().one()
                return dict(row)
        except Exception as e:
            logger.error("get_metrics failed: %s", e)
            return {}
