"""Synthesise an answer from retrieved SO docs via LLM."""
import logging
from rag.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a senior software engineer answering a developer's technical question. "
    "Use ONLY the Stack Overflow answers provided. "
    "Preserve code blocks verbatim. Do NOT fabricate — say so if context is insufficient."
)

_PROMPT = """Question: {question}

--- Stack Overflow Context ---
{context}
--- End Context ---

Write a concise, accurate answer. If multiple approaches exist, compare them briefly.
End with a Sources section listing the question URLs."""

_MAX_DOC_TOKENS = 300


def _format_context(docs: list[dict]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        prose = " ".join(d.get("answer_prose", "").split()[:_MAX_DOC_TOKENS])
        code  = d.get("answer_code_blocks", [])
        code_str = ("\n".join(f"```\n{c}\n```" for c in code[:2])) if code else ""
        parts.append(
            f"[{i}] {d.get('question_title','')}\n"
            f"{prose}\n{code_str}\n"
            f"URL: {d.get('question_url','')}"
        )
    return "\n\n".join(parts)


class AnswerGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, question: str, docs: list[dict]) -> str:
        """Format context and call LLM.  Returns answer string."""
        if not docs:
            return "No relevant Stack Overflow answers found for this question."
        ctx    = _format_context(docs)
        prompt = _PROMPT.format(question=question, context=ctx)
        try:
            return self.llm.complete(prompt, system=_SYSTEM,
                                     temperature=0.2, max_tokens=1024)
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return f"Answer generation failed: {e}"
