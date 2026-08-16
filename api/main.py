"""FastAPI application for StackSage."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models  import (QueryRequest, QueryResponse, SourceDoc,
                          JudgeScores, TimingBreakdown, FeedbackRequest,
                          FeedbackResponse, HealthResponse, MetricsResponse,
                          ServiceStatus)
from api.logger  import QueryLogger
from rag.pipeline import StackSagePipeline

logger   = logging.getLogger(__name__)
pipeline: StackSagePipeline | None = None
qlogger:  QueryLogger       | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, qlogger
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logger.info("Initialising pipeline…")
    pipeline = StackSagePipeline()
    try:
        qlogger = QueryLogger()
    except Exception as e:
        logger.warning("PostgreSQL logger unavailable: %s", e)
    logger.info("StackSage API ready")
    yield
    logger.info("Shutting down")


app = FastAPI(title="StackSage", version="0.1.0",
              description="Production-grade RAG over Stack Overflow",
              lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    result = pipeline.run(
        question=req.question,
        tags=req.tags or None,
        min_vote_score=req.min_vote_score,
        after_date=req.after_date,
        top_k=req.top_k,
    )
    if qlogger:
        qlogger.log_query(result["session_id"], req.question, result)

    js  = result.get("judge_scores") or {}
    tim = result.get("timings", {})
    sources = [
        SourceDoc(
            doc_id          = d.get("doc_id",""),
            question_title  = d.get("question_title",""),
            question_url    = d.get("question_url",""),
            tags            = d.get("tags_list",[]),
            question_score  = d.get("question_score",0),
            retrieval_score = d.get("retrieval_score",0.0),
            reranker_score  = d.get("reranker_score",0.0),
        ) for d in result.get("sources", [])
    ]
    return QueryResponse(
        answer            = result["answer"],
        sources           = sources,
        rewritten_queries = result.get("rewritten_queries", []),
        judge_scores      = JudgeScores(**{k: js.get(k) for k in
                              ("relevance","accuracy","completeness","reasoning")})
                            if any(js.get(k) for k in ("relevance","accuracy","completeness"))
                            else None,
        timings           = TimingBreakdown(
            rewrite_ms    = tim.get("rewrite_ms",0),
            retrieval_ms  = tim.get("retrieval_ms",0),
            reranker_ms   = tim.get("reranker_ms",0),
            generation_ms = tim.get("generation_ms",0),
            judge_ms      = tim.get("judge_ms",0),
            total_ms      = tim.get("total_ms",0),
        ),
        session_id        = result["session_id"],
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    if qlogger:
        qlogger.log_feedback(req.session_id, req.feedback)
    return FeedbackResponse()


@app.get("/health", response_model=HealthResponse)
async def health():
    from rag.pipeline import get_qdrant, get_es
    from ingestion.embed_and_index import get_qdrant as gq, get_es as ges
    status: dict[str, ServiceStatus] = {}

    # Qdrant
    try:
        gq().get_collections(); status["qdrant"] = ServiceStatus(ok=True)
    except Exception as e:
        status["qdrant"] = ServiceStatus(ok=False, detail=str(e))

    # Elasticsearch
    try:
        ges().cluster.health(); status["elasticsearch"] = ServiceStatus(ok=True)
    except Exception as e:
        status["elasticsearch"] = ServiceStatus(ok=False, detail=str(e))

    # Postgres
    try:
        if qlogger: qlogger.get_metrics()
        status["postgres"] = ServiceStatus(ok=True)
    except Exception as e:
        status["postgres"] = ServiceStatus(ok=False, detail=str(e))

    all_ok = all(s.ok for s in status.values())
    return HealthResponse(status="healthy" if all_ok else "degraded", **status)


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    if not qlogger:
        raise HTTPException(503, "Logger not available")
    m = qlogger.get_metrics()
    return MetricsResponse(
        total_queries          = int(m.get("total_queries") or 0),
        avg_judge_relevance    = _f(m.get("avg_relevance")),
        avg_judge_accuracy     = _f(m.get("avg_accuracy")),
        avg_judge_completeness = _f(m.get("avg_completeness")),
        avg_total_latency_ms   = _f(m.get("avg_latency_ms")),
        feedback_positive_rate = _f(m.get("feedback_pct")),
    )


def _f(v) -> float | None:
    try: return float(v) if v is not None else None
    except: return None
