.PHONY: test train infra lint

PYTHONPATH_CONFIG = PYTHONPATH=apps/api:libs/recsys_core/src:pipelines/training:pipelines/training/ml_training/src

infra:
	docker compose up -d

test:
	$(PYTHONPATH_CONFIG) pytest apps/api/tests libs/recsys_core/tests pipelines/training/tests -v

train:
	$(PYTHONPATH_CONFIG) python pipelines/training/train_pipeline.py

lint:
	ruff check libs/ pipelines/ apps/api/app/
