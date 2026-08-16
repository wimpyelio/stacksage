.PHONY: setup ingest run eval test clean

setup:
	pip install -r requirements.txt

ingest:
	python ingestion/query_sede.py
	python ingestion/clean.py
	python ingestion/embed_and_index.py
	python ingestion/build_eval_set.py

ingest-dummy:
	python scripts/make_dummy_data.py
	python ingestion/clean.py
	python ingestion/embed_and_index.py
	python ingestion/build_eval_set.py

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
	streamlit run ui/app.py

eval:
	python evaluation/evaluate.py --ground-truth data/eval/ground_truth.jsonl --mode retrieval --n-questions 50

test:
	python rag/pipeline.py

clean:
	rm -rf data/qdrant_storage data/bm25_index.pkl data/stacksage.db data/processed
