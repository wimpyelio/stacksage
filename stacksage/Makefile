.PHONY: setup ingest-sede ingest-xml run eval eval-ragas test clean lint

setup:
	docker compose up -d qdrant elasticsearch postgres
	@echo 'Waiting for services…'; sleep 15
	pip install -r requirements.txt

ingest-sede:
	python ingestion/query_sede.py
	python ingestion/clean.py
	python ingestion/embed_and_index.py
	python ingestion/build_eval_set.py

ingest-xml:
	python ingestion/parse_xml.py --input data/raw/Posts.xml
	python ingestion/clean.py
	python ingestion/embed_and_index.py
	python ingestion/build_eval_set.py

run:
	docker compose up

eval:
	python evaluation/evaluate.py \
		--ground-truth data/eval/ground_truth.jsonl \
		--output evaluation/results/ \
		--n-questions 100

eval-ragas:
	python evaluation/ragas_eval.py \
		--ground-truth data/eval/ground_truth.jsonl \
		--n-questions 50

test:
	python -c "from rag.pipeline import StackSagePipeline; p = StackSagePipeline(); r = p.run('How do I use async context managers in Python?'); print(r['answer'][:300])"

clean:
	docker compose down -v
	rm -rf data/qdrant_storage data/es_data data/pg_data

lint:
	python -m flake8 rag/ api/ ingestion/ evaluation/ --max-line-length=100
