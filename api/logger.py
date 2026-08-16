"""SQLite logger — no PostgreSQL needed."""
import logging
from pathlib import Path
from sqlalchemy import create_engine, text

logger   = logging.getLogger(__name__)
DB_PATH  = Path("data/stacksage.db")

_DDL = ["""
CREATE TABLE IF NOT EXISTS query_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP,
    question            TEXT,
    answer              TEXT,
    judge_relevance     INTEGER,
    judge_accuracy      INTEGER,
    judge_completeness  INTEGER,
    judge_reasoning     TEXT,
    rewrite_ms          INTEGER,
    retrieval_ms        INTEGER,
    reranker_ms         INTEGER,
    generation_ms       INTEGER,
    judge_ms            INTEGER,
    total_ms            INTEGER,
    user_feedback       INTEGER
)"""]

_INSERT = text("""INSERT INTO query_logs
  (session_id,question,answer,judge_relevance,judge_accuracy,judge_completeness,
   judge_reasoning,rewrite_ms,retrieval_ms,reranker_ms,generation_ms,judge_ms,total_ms)
  VALUES
  (:session_id,:question,:answer,:judge_relevance,:judge_accuracy,:judge_completeness,
   :judge_reasoning,:rewrite_ms,:retrieval_ms,:reranker_ms,:generation_ms,:judge_ms,:total_ms)
""")
_FEEDBACK = text("UPDATE query_logs SET user_feedback=:fb WHERE session_id=:sid")
_METRICS  = text("""SELECT count(*) AS total_queries,
  round(avg(judge_relevance),2)    AS avg_relevance,
  round(avg(judge_accuracy),2)     AS avg_accuracy,
  round(avg(judge_completeness),2) AS avg_completeness,
  round(avg(total_ms),0)           AS avg_latency_ms,
  round(100.0*sum(CASE WHEN user_feedback=1 THEN 1 ELSE 0 END)
        /MAX(count(user_feedback),1),1) AS feedback_pct
FROM query_logs""")


class QueryLogger:
    def __init__(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
        with self._engine.begin() as conn:
            for ddl in _DDL: conn.execute(text(ddl))
        logger.info("SQLite logger ready → %s", DB_PATH)

    def log_query(self, session_id: str, question: str, result: dict) -> None:
        js  = result.get("judge_scores",{}) or {}
        tim = result.get("timings",{})
        try:
            with self._engine.begin() as conn:
                conn.execute(_INSERT, {
                    "session_id": session_id, "question": question,
                    "answer": result.get("answer",""),
                    "judge_relevance":    js.get("relevance"),
                    "judge_accuracy":     js.get("accuracy"),
                    "judge_completeness": js.get("completeness"),
                    "judge_reasoning":    js.get("reasoning"),
                    "rewrite_ms":    tim.get("rewrite_ms"),
                    "retrieval_ms":  tim.get("retrieval_ms"),
                    "reranker_ms":   tim.get("reranker_ms"),
                    "generation_ms": tim.get("generation_ms"),
                    "judge_ms":      tim.get("judge_ms"),
                    "total_ms":      tim.get("total_ms"),
                })
        except Exception as e: logger.error("log_query: %s", e)

    def log_feedback(self, session_id: str, feedback: int) -> None:
        try:
            with self._engine.begin() as conn:
                conn.execute(_FEEDBACK, {"fb": feedback, "sid": session_id})
        except Exception as e: logger.error("log_feedback: %s", e)

    def get_metrics(self) -> dict:
        try:
            with self._engine.connect() as conn:
                return dict(conn.execute(_METRICS).mappings().one())
        except Exception as e:
            logger.error("get_metrics: %s", e); return {}

    def get_recent(self, n: int = 50) -> list[dict]:
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT {n}"
                )).mappings().all()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error("get_recent: %s", e); return []
