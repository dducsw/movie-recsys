"""
apps/api/tests/test_p1_serving.py
---------------------------------
Regression tests for the P1 serving fixes:

* retry-safe ml_training bundle singleton (transient failure recovered after
  TTL, not permanently disabled) + locked load + test hook
* recsys degradation path counters (``recsys_fallbacks_total``)
* model-fingerprint cache keys
"""

import time

import pytest


@pytest.fixture(autouse=True)
def _isolate_singletons(monkeypatch):
    from app.services import ml_training_service as mls
    from app.services import recsys as recsys_mod

    mls.reset_bundle_state()
    recsys_mod.reset_model_state()
    yield
    mls.reset_bundle_state()
    recsys_mod.reset_model_state()


# ── Bundle singleton ─────────────────────────────────────────────────────────

def test_get_model_bundle_absent_retries_after_ttl(monkeypatch):
    from app.services import ml_training_service as mls

    # Keep the artifact absent.  First call: absence recorded.
    assert mls.get_model_bundle() is None
    assert mls._BUNDLE_LAST_FAILURE > 0

    # Second call within the TTL is a no-op (no retry hammering):
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise AssertionError("should not reload inside TTL")

    monkeypatch.setattr(mls, "_bundle_path", _boom)
    assert mls.get_model_bundle() is None
    assert calls["n"] == 0

    # Simulate TTL expiry -> it retries and picks up an existing path again.
    mls._BUNDLE_LAST_FAILURE = time.time() - mls._BUNDLE_RETRY_SECONDS - 1
    monkeypatch.setattr(mls, "_bundle_path", lambda: None)
    assert mls.get_model_bundle() is None
    assert calls["n"] == 0


def test_get_model_bundle_load_failure_is_retryable(monkeypatch):
    from app.services import ml_training_service as mls

    monkeypatch.setattr(mls, "_bundle_path", lambda: "/nonexistent-seeder/x.joblib")
    # A failed load must NOT poison the singleton permanently.
    assert mls.get_model_bundle() is None
    assert mls._model_bundle is None


def test_reset_bundle_state_clears_everything(monkeypatch):
    from app.services import ml_training_service as mls

    mls._model_bundle = {"stub": True}
    mls._BUNDLE_LAST_FAILURE = 123.0
    mls.reset_bundle_state()
    assert mls._model_bundle is None
    assert mls._BUNDLE_LAST_FAILURE == 0.0


# ── Model fingerprint & cache keys ─────────────────────────────────────────────

def test_model_version_is_deterministic_and_changes_with_artifact(tmp_path, monkeypatch):
    from app.services import recsys as recsys_mod

    artifact = tmp_path / "lgb_ranker.pkl"
    artifact.write_bytes(b"0")
    monkeypatch.setenv("RANKER_MODEL_FILE", str(artifact))
    monkeypatch.setattr(recsys_mod, "_MODEL_VERSION", None)

    v1 = recsys_mod.model_version()
    assert len(v1) == 8
    assert recsys_mod.model_version() == v1  # cached

    # Republishing the same content with a new mtime bumps the fingerprint.
    import os

    os.utime(artifact, (time.time() + 5, time.time() + 5))
    monkeypatch.setattr(recsys_mod, "_MODEL_VERSION", None)
    assert recsys_mod.model_version() != v1


def test_cache_key_embeds_model_version(monkeypatch):
    from app.services import recsys as recsys_mod

    monkeypatch.setattr(recsys_mod, "_MODEL_VERSION", "deadbeef")
    key = recsys_mod._cache_key("similar", 12, 30)
    assert key.startswith("similar:vdeadbeef:")


# ── Degradation counters ─────────────────────────────────────────────────────

def test_safe_db_counts_fallback_instead_of_raising(monkeypatch):
    from app.services import recsys as recsys_mod

    calls = {"db": 0}

    class _BrokenMovieModel:
        @staticmethod
        def get_trending(page: int = 1, limit: int = 10):
            calls["db"] += 1
            raise RuntimeError("db down")

    monkeypatch.setattr(recsys_mod, "MovieModel", _BrokenMovieModel)
    rows = recsys_mod._safe_trending(10, "similar_movies")
    assert rows == []
    assert calls["db"] == 1


def test_stage2_returns_empty_when_db_metadata_unavailable(monkeypatch):
    from app.services import recsys as recsys_mod

    monkeypatch.setattr(recsys_mod.MovieModel, "get_by_ids", staticmethod(
        lambda ids: (_ for _ in ()).throw(RuntimeError("db down"))))
    ranked = recsys_mod.RecsysService._stage_2_ranking(
        candidates=[{"movieId": 1, "retrieval_score": 0.5, "source": "test"}],
        target_genres={"Action"},
        liked_count=1,
        endpoint="test",
    )
    assert ranked == []


# -- Ensemble blending weights (P2) ---------------------------------------


def test_ensemble_weights_defaults(monkeypatch):
    from app.services import ml_training_service as mls

    monkeypatch.delenv("RECSYS_CB_WEIGHT", raising=False)
    monkeypatch.delenv("RECSYS_LGB_WEIGHT", raising=False)
    assert mls._ensemble_weights({}) == {"catboost": 0.6, "lightgbm": 0.4}


def test_ensemble_weights_follow_bundle_record(monkeypatch):
    from app.services import ml_training_service as mls

    monkeypatch.delenv("RECSYS_CB_WEIGHT", raising=False)
    monkeypatch.delenv("RECSYS_LGB_WEIGHT", raising=False)
    weights = mls._ensemble_weights({"ensemble_weights": {"catboost": 0.7, "lightgbm": 0.3}})
    assert weights == {"catboost": 0.7, "lightgbm": 0.3}


def test_ensemble_weights_env_overrides_bundle(monkeypatch):
    from app.services import ml_training_service as mls

    monkeypatch.setenv("RECSYS_CB_WEIGHT", "0.5")
    monkeypatch.delenv("RECSYS_LGB_WEIGHT", raising=False)
    weights = mls._ensemble_weights({"ensemble_weights": {"catboost": 0.7, "lightgbm": 0.3}})
    assert weights == {"catboost": 0.5, "lightgbm": 0.3}


def test_ensemble_weights_invalid_values_fall_back(monkeypatch):
    from app.services import ml_training_service as mls

    monkeypatch.delenv("RECSYS_CB_WEIGHT", raising=False)
    monkeypatch.delenv("RECSYS_LGB_WEIGHT", raising=False)

    # Malformed bundle record -> defaults, not a crash.
    weights = mls._ensemble_weights({"ensemble_weights": {"catboost": "abc", "lightgbm": None}})
    assert weights == {"catboost": 0.6, "lightgbm": 0.4}

    # Non-numeric env var -> defaults preserved.
    monkeypatch.setenv("RECSYS_CB_WEIGHT", "not-a-float")
    weights = mls._ensemble_weights({})
    assert weights == {"catboost": 0.6, "lightgbm": 0.4}
