"""Build ground-truth evaluation set from high-vote questions."""
import json, logging
from pathlib import Path

logger    = logging.getLogger(__name__)
IN_JSONL  = Path("data/processed/qa_pairs_clean.jsonl")
OUT_JSONL = Path("data/eval/ground_truth.jsonl")

def build_eval_set(
    in_path: Path = IN_JSONL, out_path: Path = OUT_JSONL,
    min_score: int = 50, n: int = 100,
) -> int:
    """Select top-n questions (score >= min_score) as eval ground truth.
    Saves: question_id, question_title, question_body, question_url,
           accepted_answer, tags_list, question_score.
    Returns records saved."""
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} missing — run clean.py first")
    candidates = sorted(
        [r for l in in_path.open(encoding="utf-8")
         if l.strip()
         for r in [json.loads(l)]
         if r.get("question_score", 0) >= min_score],
        key=lambda r: r.get("question_score", 0), reverse=True
    )[:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in candidates:
            fh.write(json.dumps({
                "question_id":    r["question_id"],
                "question_title": r["question_title"],
                "question_body":  r.get("question_prose",""),
                "question_url":   r["question_url"],
                "accepted_answer":r.get("answer_prose",""),
                "tags_list":      r.get("tags_list",[]),
                "question_score": r.get("question_score",0),
            }, ensure_ascii=False) + "\n")
    logger.info("Eval set: %d selected → %s", len(candidates), out_path)
    return len(candidates)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from dotenv import load_dotenv; load_dotenv()
    build_eval_set()
