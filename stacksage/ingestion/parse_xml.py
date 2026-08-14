"""Stream-parse Posts.xml (Stack Exchange XML dump alternative).
Produces the same JSONL schema as query_sede.py.
Usage: python ingestion/parse_xml.py --input data/raw/Posts.xml
"""
import argparse, hashlib, json, logging, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)
OUT_JSONL   = Path("data/processed/qa_pairs.jsonl")
MIN_Q_SCORE = 5; MIN_A_SCORE = 3
AFTER_DATE  = "2020-01-01"; TARGET_TAG = "<python>"

def _safe_int(v):
    try: return int(float(str(v)))
    except: return 0

def iter_posts(xml_path: Path) -> Iterator[dict]:
    """iterparse — constant memory regardless of file size."""
    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag == "row":
            yield dict(elem.attrib); elem.clear()

def parse_posts_xml(xml_path: Path) -> list[dict]:
    posts = list(iter_posts(xml_path))
    logger.info("Parsed %d raw posts from %s", len(posts), xml_path)
    return posts

def join_qa_pairs(posts: list[dict]) -> list[dict]:
    """Join questions (PostTypeId=1) with accepted answers (PostTypeId=2)."""
    qs, ans = {}, {}
    for p in posts:
        (qs if p.get("PostTypeId")=="1" else ans if p.get("PostTypeId")=="2" else {})[p.get("Id","?")] = p
    pairs = []
    for qid, q in qs.items():
        if _safe_int(q.get("Score",0)) < MIN_Q_SCORE: continue
        if TARGET_TAG not in q.get("Tags",""): continue
        if q.get("CreationDate","")[:10] < AFTER_DATE: continue
        aid = q.get("AcceptedAnswerId")
        if not aid or aid not in ans: continue
        a = ans[aid]
        if _safe_int(a.get("Score",0)) < MIN_A_SCORE: continue
        pairs.append({
            "doc_id":          hashlib.md5(qid.encode()).hexdigest(),
            "question_id":     qid,
            "question_title":  q.get("Title",""),
            "question_body":   q.get("Body",""),
            "question_score":  _safe_int(q.get("Score")),
            "tags":            q.get("Tags",""),
            "creation_date":   q.get("CreationDate","")[:10],
            "accepted_answer": a.get("Body",""),
            "answer_score":    _safe_int(a.get("Score")),
            "question_url":    f"https://stackoverflow.com/q/{qid}",
        })
    logger.info("Joined %d QA pairs", len(pairs)); return pairs

def write_jsonl(pairs: list[dict], out_path: Path = OUT_JSONL) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for p in pairs: fh.write(json.dumps(p, ensure_ascii=False)+"\n")
    logger.info("Wrote %d records → %s", len(pairs), out_path); return len(pairs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=str(OUT_JSONL)); args = ap.parse_args()
    write_jsonl(join_qa_pairs(parse_posts_xml(Path(args.input))), Path(args.output))
