import pandas as pd
import yaml

with open("configs.yaml") as f:
    CFG = yaml.safe_load(f)

def load_all():
    P = CFG["paths"]
    crawled = pd.read_csv(P["crawled"])
    links   = pd.read_csv(P["links"])
    movies  = pd.read_csv(P["movies"])
    ratings = pd.read_csv(P["ratings"])  # userId,movieId,rating,timestamp

    # Normalize column types
    crawled["movieId"] = pd.to_numeric(crawled["movieId"], errors="coerce")
    crawled = crawled.dropna(subset=["movieId"])
    crawled["movieId"] = crawled["movieId"].astype(int)

    links["movieId"] = links["movieId"].astype(int)
    links["tmdbId"]  = pd.to_numeric(links["tmdbId"], errors="coerce")

    movies["movieId"] = movies["movieId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["userId"]  = ratings["userId"].astype(int)
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="coerce")

    # Merge crawled metadata with links (by tmdbId) to enrich ml movie ids
    enriched = (
        links.merge(crawled, left_on="tmdbId", right_on="movieId",
                    how="left", suffixes=("_ml", "_crawl"))
             .rename(columns={"movieId_ml": "movieId"})
             .drop(columns=["movieId_crawl"])
    )
    # Fall back to the small movies.csv metadata for unmatched rows
    enriched = enriched.merge(movies[["movieId","genres"]],
                              on="movieId", how="left", suffixes=("","_ml"))

    # Prefer crawled genres when present, else use ML genres
    enriched["genres"] = enriched["genres"].fillna(enriched["genres_ml"])
    enriched = enriched.drop(columns=["genres_ml"])

    return crawled, links, movies, ratings, enriched