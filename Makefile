.PHONY: setup ingest-dummy ingest run eval test clean

setup:
	pip install -r requirements.txt

ingest-dummy:
	PYTHONPATH=. python scripts/make_dummy_data.py
	PYTHONPATH=. python ingestion/query_sede.py
	PYTHONPATH=. python ingestion/clean.py
	PYTHONPATH=. python ingestion/embed_and_index.py
	PYTHONPATH=. python ingestion/build_eval_set.py

ingest:
	PYTHONPATH=. python ingestion/query_sede.py
	PYTHONPATH=. python ingestion/clean.py
	PYTHONPATH=. python ingestion/embed_and_index.py
	PYTHONPATH=. python ingestion/build_eval_set.py

run:
	PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
	PYTHONPATH=. streamlit run ui/app.py

eval:
	PYTHONPATH=. python evaluation/evaluate.py \
		--ground-truth data/eval/ground_truth.jsonl \
		--mode retrieval --n-questions 20

test:
	PYTHONPATH=. python rag/pipeline.py

clean:
	rm -rf data/qdrant_storage data/bm25_index.pkl data/stacksage.db data/processed
