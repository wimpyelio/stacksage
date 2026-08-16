"""Ingest Stack Exchange Q&A from a SEDE CSV export.

HOW TO GET THE DATA:
  1. https://data.stackexchange.com/stackoverflow/query/new
  2. Paste SEDE_SQL below, click Download Results
  3. Save as data/raw/sede_export.csv
  4. python ingestion/query_sede.py
"""
import hashlib, json, logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)
RAW_CSV   = Path("data/raw/sede_export.csv")
OUT_JSONL = Path("data/processed/qa_pairs.jsonl")

SEDE_SQL = """
SELECT TOP 50000
    q.Id, q.Title, q.Body, q.Score, q.Tags,
    CONVERT(VARCHAR(10), q.CreationDate, 23) AS CreationDate,
    a.Body AS AcceptedAnswerBody, a.Score AS AnswerScore
FROM Posts q
JOIN Posts a ON q.AcceptedAnswerId = a.Id
WHERE q.Score >= 5
  AND q.Tags LIKE '%<python>%'
  AND q.CreationDate >= '2020-01-01'
  AND q.ClosedDate IS NULL
  AND a.Score >= 3
ORDER BY q.Score DESC
"""

def _safe_int(v):
    try: return int(float(str(v)))
    except: return 0

def load_sede_csv(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Load SEDE CSV; raises FileNotFoundError with actionable message."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} missing.\nRun SEDE_SQL at "
            "https://data.stackexchange.com/stackoverflow/query/new"
        )
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df

def convert_to_jsonl(df: pd.DataFrame, out_path: Path = OUT_JSONL) -> int:
    """Write JSONL with stable doc_id (MD5 of question Id).
    Schema: doc_id, question_id, question_title, question_body,
    question_score, tags, creation_date, accepted_answer,
    answer_score, question_url.
    Returns rows written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            qid = str(row.get("Id","")).strip()
            if not qid: continue
            fh.write(json.dumps({
                "doc_id":          hashlib.md5(qid.encode()).hexdigest(),
                "question_id":     qid,
                "question_title":  str(row.get("Title","")).strip(),
                "question_body":   str(row.get("Body","")).strip(),
                "question_score":  _safe_int(row.get("Score")),
                "tags":            str(row.get("Tags","")).strip(),
                "creation_date":   str(row.get("CreationDate","")).strip(),
                "accepted_answer": str(row.get("AcceptedAnswerBody","")).strip(),
                "answer_score":    _safe_int(row.get("AnswerScore")),
                "question_url":    f"https://stackoverflow.com/q/{qid}",
            }, ensure_ascii=False) + "\n"); n += 1
    logger.info("Wrote %d records → %s", n, out_path)
    return n

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from dotenv import load_dotenv; load_dotenv()
    convert_to_jsonl(load_sede_csv())
