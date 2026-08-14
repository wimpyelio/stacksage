"""RAGAS evaluation: faithfulness, answer_relevancy, context_precision, context_recall.

Usage:
    python evaluation/ragas_eval.py \
        --ground-truth data/eval/ground_truth.jsonl \
        --n-questions 50
"""
import argparse, json, logging
from pathlib import Path

logger = logging.getLogger(__name__)
OUT = Path("evaluation/results")


def run_ragas(
    ground_truth_path: str,
    pipeline,
    n: int = 50,
) -> dict:
    """
    Evaluate StackSage with RAGAS metrics.

    Metrics:
        faithfulness        — is the answer grounded in the retrieved context?
        answer_relevancy    — does the answer address the question?
        context_precision   — are top contexts relevant?
        context_recall      — does context cover the ground truth answer?

    Saves evaluation/results/ragas_report.json.
    Returns dict of metric averages.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        )
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "Install ragas + datasets:  pip install ragas datasets"
        ) from e

    gt    = _load_gt(ground_truth_path, n)
    rows  = {"question":[], "answer":[], "contexts":[], "ground_truths":[]}

    for i, item in enumerate(gt):
        try:
            result   = pipeline.run(item["question_title"])
            answer   = result.get("answer","")
            contexts = [
                f"{s.get('question_title','')} {s.get('answer_prose','')}"
                for s in result.get("sources", [])
            ]
            rows["question"].append(item["question_title"])
            rows["answer"].append(answer)
            rows["contexts"].append(contexts)
            rows["ground_truths"].append([item.get("accepted_answer","")])
        except Exception as e:
            logger.error("RAGAS item %d error: %s", i, e)
        if (i+1) % 10 == 0:
            logger.info("RAGAS: %d/%d", i+1, len(gt))

    if not rows["question"]:
        logger.error("No items evaluated"); return {}

    ds     = Dataset.from_dict(rows)
    report = evaluate(ds, metrics=[
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    ])
    result_dict = {
        "faithfulness":      round(float(report["faithfulness"]),      4),
        "answer_relevancy":  round(float(report["answer_relevancy"]),  4),
        "context_precision": round(float(report["context_precision"]), 4),
        "context_recall":    round(float(report["context_recall"]),    4),
        "n_questions":       len(rows["question"]),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"ragas_report.json").write_text(json.dumps(result_dict, indent=2))
    logger.info("RAGAS report saved → %s/ragas_report.json", OUT)

    print("\n=== RAGAS Results ===")
    for k, v in result_dict.items():
        print(f"  {k:<22}: {v}")
    return result_dict


def _load_gt(path: str, n: int) -> list[dict]:
    items = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    logger.info("Loaded %d GT items (using %d)", len(items), min(n, len(items)))
    return items[:n]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from dotenv import load_dotenv; load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--n-questions", type=int, default=50)
    args = ap.parse_args()

    from rag.pipeline import StackSagePipeline
    run_ragas(args.ground_truth, StackSagePipeline(), args.n_questions)
