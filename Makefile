.PHONY: test train full-eval parity backfill-qdrant infra lint bench

PYTHONPATH_CONFIG = PYTHONPATH=apps/api:libs/recsys_core/src:pipelines/training:pipelines/training/ml_training/src

infra:
	docker compose up -d

test:
	$(PYTHONPATH_CONFIG) pytest apps/api/tests libs/recsys_core/tests pipelines/training/tests -v

train:
	$(PYTHONPATH_CONFIG) python pipelines/training/train_pipeline.py

full-eval:
	$(PYTHONPATH_CONFIG) python pipelines/training/full_eval.py --k 10

parity:
	$(PYTHONPATH_CONFIG) python pipelines/training/parity_check.py

backfill-qdrant:
	$(PYTHONPATH_CONFIG) python pipelines/batch/backfill_qdrant.py

bench:
	k6 run benchmarks/k6_recsys.js

lint:
	ruff check libs/ pipelines/ apps/api/app/
	mypy libs/recsys_core/src pipelines/training/train_pipeline.py --ignore-missing-imports
