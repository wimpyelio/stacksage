"""LLM-as-judge: score answers on relevance, accuracy, completeness."""
import json, logging
from typing import Optional
from rag.llm_client import LLMClient

logger = logging.getLogger(__name__)

_NULL = {"relevance": None, "accuracy": None, "completeness": None, "reasoning": "judge error"}

_PROMPT = """Evaluate this AI answer to a developer question.

Question: {question}
{gt_section}
AI Answer: {answer}

Score each dimension 1 (very poor) to 5 (excellent):
1. relevance   — directly addresses the question?
2. accuracy    — technically correct?
3. completeness — covers key aspects?

Return ONLY valid JSON — no text outside the object:
{{"relevance": N, "accuracy": N, "completeness": N, "reasoning": "one sentence"}}"""


class LLMJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def score(self, question: str, answer: str,
              ground_truth: Optional[str] = None) -> dict:
        """Score answer; returns _NULL on any failure (never raises)."""
        gt_section = (f"Ground Truth: {ground_truth[:300]}\n"
                      if ground_truth else "")
        prompt = _PROMPT.format(question=question, answer=answer[:1500],
                                gt_section=gt_section)
        try:
            raw = self.llm.complete(prompt, temperature=0.0, max_tokens=256)
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            # Clamp scores to 1-5
            for k in ("relevance", "accuracy", "completeness"):
                if isinstance(parsed.get(k), (int, float)):
                    parsed[k] = max(1, min(5, int(parsed[k])))
            return parsed
        except Exception as e:
            logger.warning("Judge parse failed (%s)", e)
            return _NULL
