"""
apps/api/app/chatbot/redis_checkpointer.py
-----------------------------------------
Hybrid Redis + InMemory CheckpointSaver for LangGraph.
Persists multi-turn session state across multiple backend replicas via Redis.
Fails-open to standard InMemorySaver if Redis is unavailable.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.services.redis_factory import get_redis

logger = logging.getLogger(__name__)

CHECKPOINT_TTL_SECONDS = 86400 * 7  # 7 days


class RedisMemorySaver(InMemorySaver):
    """
    LangGraph CheckpointSaver backed by Redis.
    Syncs checkpoints and writes to Redis key `chat:{thread_id}:checkpoint_data`.
    Enables load balancing across multiple uvicorn workers and replicas without losing conversation state.
    """

    def _sync_from_redis(self, thread_id: str) -> None:
        """Hydrate in-memory thread storage from Redis if available."""
        r = get_redis(db=0, timeout=0.3)
        if not r:
            return
        try:
            raw = r.get(f"chat:{thread_id}:checkpoint_data")
            if raw:
                data = pickle.loads(raw)
                if isinstance(data, dict):
                    if "storage" in data and thread_id not in self.storage:
                        self.storage[thread_id] = data["storage"]
                    if "writes" in data:
                        self.writes.update(data["writes"])
                    if "blobs" in data:
                        self.blobs.update(data["blobs"])
        except Exception as exc:
            logger.debug("Failed hydrating checkpoint from Redis for thread %s: %s", thread_id, exc)

    def _sync_to_redis(self, thread_id: str) -> None:
        """Persist thread's checkpoint data to Redis."""
        r = get_redis(db=0, timeout=0.3)
        if not r:
            return
        try:
            storage_data = dict(self.storage.get(thread_id, {}))
            writes_data = {k: v for k, v in self.writes.items() if k[0] == thread_id}
            blobs_data = {k: v for k, v in self.blobs.items() if k[0] == thread_id}
            payload = {
                "storage": storage_data,
                "writes": writes_data,
                "blobs": blobs_data,
            }
            r.setex(f"chat:{thread_id}:checkpoint_data", CHECKPOINT_TTL_SECONDS, pickle.dumps(payload))
        except Exception as exc:
            logger.debug("Failed persisting checkpoint to Redis for thread %s: %s", thread_id, exc)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id and thread_id not in self.storage:
            self._sync_from_redis(thread_id)
        return super().get_tuple(config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        res = super().put(config, checkpoint, metadata, new_versions)
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            self._sync_to_redis(thread_id)
        return res

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        super().put_writes(config, writes, task_id)
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            self._sync_to_redis(thread_id)
