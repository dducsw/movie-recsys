import re, json, ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import yaml

with open("configs.yaml") as f:
    CFG = yaml.safe_load(f)

GENRE_LIST = CFG["features"]["genre_list"]

# ---------- Helpers ----------
def parse_list(x):
    """Parse strings that look like lists or JSON-ish arrays."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if not isinstance(x, str):
        return []
    s = x.strip()
    if not s or s.lower() in ("nan","none","[]"):
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]
    except Exception:
        # fallback: split by comma/pipe
        return [t.strip() for t in re.split(r"[,|;]", s) if t.strip()]

def extract_names(lst, key="name"):
    out = []
    for item in lst:
        if isinstance(item, dict) and key in item:
            out.append(item[key])
        elif isinstance(item, str):
            out.append(item)
    return out

def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

# ---------- Movie features ----------
def build_movie_features(enriched: pd.DataFrame) -> pd.DataFrame:
    df = enriched.copy()

    # release_date -> year, month, decade
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]  = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_decade"] = (df["release_year"] // 10 * 10).astype("Float64")

    # numeric
    for c in ["popularity","vote_average","vote_count"]:
        df[c] = df[c].apply(safe_float)

    # adult
    df["adult"] = df["adult"].astype(str).str.lower().isin(["true","1","yes"])
    df["adult"] = df["adult"].astype(int)

    # genres -> multi-hot
    df["genres_list"] = df["genres"].apply(parse_list).apply(
        lambda gs: [g for g in gs if g in GENRE_LIST])
    mlb = MultiLabelBinarizer(classes=GENRE_LIST)
    genre_oh = pd.DataFrame(mlb.fit_transform(df["genres_list"]),
                            columns=[f"genre_{g}" for g in mlb.classes_],
                            index=df.index)
    df["n_genres"] = df["genres_list"].apply(len)

    # director / cast / keywords -> top-N frequency encoding
    df["director_list"] = df["director"].apply(parse_list).apply(
        lambda x: extract_names(x, "name") if x and isinstance(x[0], dict) else x)
    df["cast_list"]     = df["cast"].apply(parse_list).apply(extract_names)
    df["keywords_list"] = df["keywords"].apply(parse_list).apply(extract_names)

    df["n_cast"]     = df["cast_list"].apply(len)
    df["n_keywords"] = df["keywords_list"].apply(len)

    # primary (first) director / first actor as categorical
    df["primary_director"] = df["director_list"].apply(
        lambda x: x[0] if len(x) else "<unknown>")
    df["primary_actor"] = df["cast_list"].apply(
        lambda x: x[0] if len(x) else "<unknown>")

    # overview length & simple keyword density
    df["overview_len"] = df["overview"].fillna("").apply(len)

    out = pd.concat([df, genre_oh], axis=1)
    return out

# ---------- User features (computed on the TRAIN ratings only) ----------
def build_user_features(train_ratings: pd.DataFrame,
                         movie_feats: pd.DataFrame) -> pd.DataFrame:
    mf = movie_feats[["movieId","popularity","vote_average",
                      "release_year"] + [c for c in movie_feats.columns
                                         if c.startswith("genre_")]]
    r = train_ratings.merge(mf, on="movieId", how="left")

    g = r.groupby("userId").agg(
        user_avg_rating   = ("rating","mean"),
        user_std_rating   = ("rating","std"),
        user_n_ratings    = ("rating","count"),
        user_avg_pop      = ("popularity","mean"),
        user_avg_vote     = ("vote_average","mean"),
        user_avg_year     = ("release_year","mean"),
    ).reset_index()
    g["user_std_rating"] = g["user_std_rating"].fillna(0)

    # user genre preference vector (mean of genre_* weighted by rating)
    genre_cols = [c for c in r.columns if c.startswith("genre_")]
    for c in genre_cols:
        r[c] = r[c] * r["rating"]
    gpref = r.groupby("userId")[genre_cols].mean().reset_index()
    gpref.columns = ["userId"] + [f"user_pref_{c}" for c in genre_cols]
    g = g.merge(gpref, on="userId", how="left")
    return g

# ---------- Build the training matrix ----------
def build_dataset(ratings: pd.DataFrame, movie_feats: pd.DataFrame,
                  user_feats: pd.DataFrame) -> pd.DataFrame:
    mf_keep = ["movieId","popularity","vote_average","vote_count","adult",
               "release_year","release_month","release_decade",
               "n_genres","n_cast","n_keywords","overview_len",
               "primary_director","primary_actor"] + \
              [c for c in movie_feats.columns if c.startswith("genre_")]
    data = ratings.merge(movie_feats[mf_keep], on="movieId", how="left")
    data = data.merge(user_feats, on="userId", how="left")

    # user-movie interaction feature: does the movie's release year match user avg
    data["year_match"] = (data["release_year"] - data["user_avg_year"]).abs()

    # genre overlap between user preference and movie genres
    genre_cols = [c for c in movie_feats.columns if c.startswith("genre_")]
    pref_cols  = [f"user_pref_{c}" for c in genre_cols]
    overlap = (data[genre_cols].values * data[pref_cols].values).sum(axis=1)
    data["genre_match_score"] = overlap
    return data