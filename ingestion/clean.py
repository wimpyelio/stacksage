"""Clean raw QA pairs: strip HTML, extract code blocks, truncate, normalise tags."""
import json, logging, re
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
IN_JSONL  = Path("data/processed/qa_pairs.jsonl")
OUT_JSONL = Path("data/processed/qa_pairs_clean.jsonl")
MAX_ANS   = 512   # tokens for embedding field
MAX_BODY  = 256


def strip_html(html: str) -> str:
    """Remove tags, decode entities, collapse whitespace."""
    return re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" ")).strip()


def extract_code_blocks(html: str) -> tuple[str, list[str]]:
    """Replace <pre><code> with [CODE] placeholder; return (stripped_html, code_list)."""
    soup = BeautifulSoup(html, "lxml")
    blocks = []
    for pre in soup.find_all("pre"):
        blocks.append(pre.get_text().strip())
        pre.replace_with(" [CODE] ")
    return str(soup), blocks


def truncate(text: str, n: int) -> str:
    toks = text.split()
    return " ".join(toks[:n]) if len(toks) > n else text


def normalize_tags(raw: str) -> list[str]:
    return [t for t in re.split(r"[<>]", raw) if t.strip()]


def clean_record(rec: dict) -> Optional[dict]:
    """Clean one record; returns None if it should be filtered out."""
    raw = rec.get("accepted_answer", "")
    if not raw.strip(): return None

    no_code, code_blocks = extract_code_blocks(raw)
    prose = truncate(strip_html(no_code), MAX_ANS)
    if len(prose.split()) < 10: return None   # answer too sparse

    q_prose = truncate(strip_html(rec.get("question_body", "")), MAX_BODY)
    tags    = normalize_tags(rec.get("tags", ""))
    title   = rec.get("question_title", "")

    embed_text = truncate(
        f"TITLE: {title} | TAGS: {' '.join(tags)} | ANSWER: {prose}", MAX_ANS
    )
    return {
        **rec,
        "answer_prose":        prose,
        "answer_code_blocks":  code_blocks,
        "question_prose":      q_prose,
        "tags_list":           tags,
        "text_for_embedding":  embed_text,
    }


def process_file(in_path: Path = IN_JSONL, out_path: Path = OUT_JSONL) -> int:
    """Clean all records; write survivors.  Idempotent.  Returns count written."""
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} missing — run query_sede.py first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = filtered = 0
    with in_path.open(encoding="utf-8") as fi, out_path.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line: continue
            total += 1
            cleaned = clean_record(json.loads(line))
            if cleaned is None: filtered += 1; continue
            fo.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
    written = total - filtered
    logger.info("Clean: %d in → %d out (%d filtered)", total, written, filtered)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    process_file()
