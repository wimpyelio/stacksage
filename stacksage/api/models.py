"""Pydantic request / response models for the StackSage API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Inbound ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=10, max_length=1_000,
        description="Developer's technical question",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Filter results to these Stack Overflow tags",
        examples=[["python", "pandas"]],
    )
    min_vote_score: int = Field(
        5, ge=0, le=500,
        description="Minimum question vote score",
    )
    after_date: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Only include posts after this ISO date (YYYY-MM-DD)",
        examples=["2022-01-01"],
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of sources to return")


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: int = Field(..., ge=-1, le=1, description="1=thumbs up, -1=thumbs down")


# ── Outbound ───────────────────────────────────────────────────────────────────

class SourceDoc(BaseModel):
    doc_id: str
    question_title: str
    question_url: str
    tags: list[str]
    question_score: int
    retrieval_score: float = Field(description="RRF score from hybrid retrieval")
    reranker_score: float = Field(description="Cross-encoder score")


class JudgeScores(BaseModel):
    relevance: Optional[int] = Field(None, ge=1, le=5)
    accuracy: Optional[int] = Field(None, ge=1, le=5)
    completeness: Optional[int] = Field(None, ge=1, le=5)
    reasoning: Optional[str] = None


class TimingBreakdown(BaseModel):
    rewrite_ms: int
    retrieval_ms: int
    reranker_ms: int
    generation_ms: int
    judge_ms: int
    total_ms: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    rewritten_queries: list[str]
    judge_scores: Optional[JudgeScores] = None
    timings: TimingBreakdown
    session_id: str


class FeedbackResponse(BaseModel):
    status: str = "ok"


# ── Health / metrics ───────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    ok: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded"
    qdrant: ServiceStatus
    elasticsearch: ServiceStatus
    postgres: ServiceStatus


class MetricsResponse(BaseModel):
    total_queries: int
    avg_judge_relevance: Optional[float]
    avg_judge_accuracy: Optional[float]
    avg_judge_completeness: Optional[float]
    avg_total_latency_ms: Optional[float]
    feedback_positive_rate: Optional[float]
