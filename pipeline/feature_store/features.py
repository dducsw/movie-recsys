"""
pipeline/feature_store/features.py
-----------------------------------
Feast feature definitions for MovieNex recommendation system.
Defines Entities (movie, user) and Feature Views (movie_stats, user_stats).
"""

from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    PushSource,
    ValueType,
)
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float32, Int64, String

# 1. Define Entities
movie_entity = Entity(
    name="movie_id",
    value_type=ValueType.INT64,
    join_keys=["movieId"],
    description="Unique identifier for each movie item in the catalog",
)

user_entity = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    join_keys=["userId"],
    description="Unique identifier for each user profile",
)

# 2. Define Batch Sources (PostgreSQL)
movie_stats_source = PostgreSQLSource(
    name="movie_stats_source",
    query="""
        SELECT 
            movieid AS "movieId",
            popularity::float4 AS popularity,
            vote_average::float4 AS vote_average,
            COALESCE(vote_count, 0)::int8 AS vote_count,
            COALESCE(NULLIF(SUBSTRING(release_date, 1, 4), ''), '2010')::int8 AS release_year,
            NOW() AS event_timestamp,
            NOW() AS created_timestamp
        FROM movies
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

user_stats_source = PostgreSQLSource(
    name="user_stats_source",
    query="""
        SELECT 
            COALESCE(user_id, 0)::int8 AS "userId",
            COUNT(*)::float4 AS user_activity,
            (AVG(rating) - 3.5)::float4 AS user_bias,
            NOW() AS event_timestamp,
            NOW() AS created_timestamp
        FROM user_ratings
        WHERE user_id IS NOT NULL
        GROUP BY user_id
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Define Feature Views
movie_stats_view = FeatureView(
    name="movie_stats",
    entities=[movie_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="popularity", dtype=Float32),
        Field(name="vote_average", dtype=Float32),
        Field(name="vote_count", dtype=Int64),
        Field(name="release_year", dtype=Int64),
    ],
    online=True,
    source=movie_stats_source,
)

user_stats_view = FeatureView(
    name="user_stats",
    entities=[user_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="user_activity", dtype=Float32),
        Field(name="user_bias", dtype=Float32),
    ],
    online=True,
    source=user_stats_source,
)
