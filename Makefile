.PHONY: test train infra lint

infra:
	docker compose up -d

test:
	pytest apps/api/tests libs/recsys_core/tests pipelines/training/tests -v

train:
	python pipelines/training/train_pipeline.py

lint:
	ruff check libs/ pipelines/ apps/api/app/
