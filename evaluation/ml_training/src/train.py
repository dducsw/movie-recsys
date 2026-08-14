import numpy as np
import pandas as pd
import joblib
import json
import time
import os
import yaml
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_loader import load_all
from features import build_movie_features, build_user_features, build_dataset
from models import train_lgbm, train_catboost, prepare_lgb_prediction_data

with open("configs.yaml", "r") as f:
    CFG = yaml.safe_load(f)

def time_split_per_user(ratings, n_test=5, n_val=5, min_ratings=15, seed=42):
    train_idx, val_idx, test_idx = [], [], []
    for uid, g in ratings.groupby("userId"):
        g = g.sort_values("timestamp")
        if len(g) < min_ratings:
            train_idx.extend(g.index.tolist()); continue
        idx = g.index.tolist()
        test_idx.extend(idx[-n_test:])
        val_idx.extend(idx[-(n_test+n_val):-n_test])
        train_idx.extend(idx[:-(n_test+n_val)])
    return (ratings.loc[train_idx].reset_index(drop=True),
            ratings.loc[val_idx].reset_index(drop=True),
            ratings.loc[test_idx].reset_index(drop=True))

def rmse_mae(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae

def evaluate_ranking_metrics(eval_df, k=10, relevance_threshold=4.0):
    """
    Calculate Hit Ratio@k, NDCG@k, and MRR.
    """
    hit_ratios, ndcgs, rrs = [], [], []
    
    for uid, g in eval_df.groupby("userId"):
        if len(g) < 1: continue
        g = g.sort_values("pred", ascending=False)
        
        # Binary relevance: 1 if actual rating >= threshold, else 0
        rel = (g["rating"] >= relevance_threshold).astype(int).tolist()
        
        # Skip users who have no relevant items in the test set
        if sum(rel) == 0:
            continue
            
        # Top K predictions
        topk_rel = rel[:k]
        
        # 1. Hit Ratio@K: 1 if there is at least one hit in top K, else 0
        hit = 1 if sum(topk_rel) > 0 else 0
        hit_ratios.append(hit)
        
        # 2. NDCG@K
        dcg = sum([r / np.log2(i + 2) for i, r in enumerate(topk_rel)])
        ideal_rel = sorted(rel, reverse=True)[:k]
        idcg = sum([r / np.log2(i + 2) for i, r in enumerate(ideal_rel)])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
        
        # 3. MRR (Mean Reciprocal Rank): 1 / rank of the first relevant item
        rr = 0.0
        for i, r in enumerate(rel):
            if r == 1:
                rr = 1.0 / (i + 1)
                break
        rrs.append(rr)
        
    return {
        f"Hit Ratio@{k}": float(np.mean(hit_ratios)) if hit_ratios else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "MRR": float(np.mean(rrs)) if rrs else 0.0
    }

def plot_and_save_comparison(
    metrics_lgb: dict,
    metrics_cb: dict,
    lgb_time: float,
    cb_time: float,
    lgb_inf_time: float,
    cb_inf_time: float,
    save_path: str = "model_comparison.png"
):
    """
    Plots a multi-panel comparison between LightGBM and CatBoost (YetiRank)
    and saves the figure to disk.
    """
    # Set visual styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = ["#2b5c8f", "#d95f02"]  # Blue for LightGBM, Orange for CatBoost
    models = ["LightGBM", "CatBoost\n(YetiRank)"]

    # ----------------------------------------------------
    # Subplot 1: Ranking Quality Metrics (Higher is Better)
    # ----------------------------------------------------
    ranking_metrics = ["Hit Ratio@10", "NDCG@10", "MRR"]
    lgb_scores = [metrics_lgb[m] for m in ranking_metrics]
    cb_scores = [metrics_cb[m] for m in ranking_metrics]

    x = np.arange(len(ranking_metrics))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, lgb_scores, width, label="LightGBM", color=colors[0], edgecolor="black", alpha=0.85)
    rects2 = axes[0].bar(x + width/2, cb_scores, width, label="CatBoost (YetiRank)", color=colors[1], edgecolor="black", alpha=0.85)

    axes[0].set_title("Ranking Quality (Higher is Better)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ranking_metrics, fontsize=10)
    axes[0].set_ylim(0, max(max(lgb_scores), max(cb_scores)) * 1.15)
    axes[0].legend(loc="upper left")
    axes[0].bar_label(rects1, fmt="%.4f", padding=3, fontsize=8.5)
    axes[0].bar_label(rects2, fmt="%.4f", padding=3, fontsize=8.5)

    # ----------------------------------------------------
    # Subplot 2: Training Time (Lower is Better)
    # ----------------------------------------------------
    train_times = [lgb_time, cb_time]
    rects_train = axes[1].bar(models, train_times, color=colors, width=0.45, edgecolor="black", alpha=0.85)

    axes[1].set_title("Training Time (Lower is Better)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Seconds (s)")
    axes[1].set_ylim(0, max(train_times) * 1.18)
    axes[1].bar_label(rects_train, fmt="%.2fs", padding=3, fontsize=9)

    # ----------------------------------------------------
    # Subplot 3: Inference Latency (Lower is Better)
    # ----------------------------------------------------
    inf_times = [lgb_inf_time, cb_inf_time]
    rects_inf = axes[2].bar(models, inf_times, color=colors, width=0.45, edgecolor="black", alpha=0.85)

    axes[2].set_title("Inference Latency (Lower is Better)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Seconds per User (s/user)")
    axes[2].set_ylim(0, max(inf_times) * 1.18)
    axes[2].bar_label(rects_inf, fmt="%.4fs", padding=3, fontsize=9)

    # Overall figure adjustments
    plt.suptitle("Offline A/B Comparison: LightGBM vs. CatBoost (YetiRank)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save high-resolution figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved successfully to: {save_path}")
    plt.show()

def main():
    print(">>> Loading data ...")
    crawled, links, movies, ratings, enriched = load_all()

    print(">>> Building movie features ...")
    movie_feats = build_movie_features(enriched)

    print(">>> Splitting ratings (time-based per user) ...")
    s = CFG["split"]
    train_r, val_r, test_r = time_split_per_user(
        ratings, n_test=s["n_holdout_per_user"], n_val=s["n_holdout_per_user"],
        min_ratings=2*s["n_holdout_per_user"] + s["min_user_ratings"],
        seed=s["random_state"])

    print(">>> Building user features on train ...")
    user_feats = build_user_features(train_r, movie_feats)

    print(">>> Assembling datasets ...")
    train = build_dataset(train_r, movie_feats, user_feats)
    val   = build_dataset(val_r,   movie_feats, user_feats)
    test  = build_dataset(test_r,  movie_feats, user_feats)

    y_tr  = train["rating"].astype(float).values
    y_val = val["rating"].astype(float).values
    y_te  = test["rating"].astype(float).values

    # Load tuned parameters if tuning was run
    tuned_params = None
    if os.path.exists("best_params.json"):
        print(">>> Found best_params.json. Using tuned hyperparameters.")
        with open("best_params.json") as f:
            tuned_params = json.load(f)

    # -------- LightGBM --------
    print(">>> Training LightGBM ...")
    t0 = time.time()
    lgb_params = tuned_params["lgbm"] if tuned_params else None

    lgb_model, lgb_cols, lgb_category_maps = train_lgbm(
        train.drop(columns=["rating", "timestamp"]).copy(),
        y_tr,
        val.drop(columns=["rating", "timestamp"]).copy(),
        y_val,
        params=lgb_params
    )
    lgb_time = time.time() - t0

    X_val_lgb = prepare_lgb_prediction_data(val, lgb_cols, lgb_category_maps)
    X_test_lgb = prepare_lgb_prediction_data(test, lgb_cols, lgb_category_maps)

    pred_val_lgb = lgb_model.predict(X_val_lgb, num_iteration=lgb_model.best_iteration)
    
    # Measure Inference Time per user for LightGBM
    t_inf_lgb = time.time()
    pred_te_lgb = lgb_model.predict(X_test_lgb, num_iteration=lgb_model.best_iteration)
    lgb_inf_time = (time.time() - t_inf_lgb) / test["userId"].nunique()

    rmse_v_lgb, _ = rmse_mae(y_val, pred_val_lgb)
    rmse_t_lgb, mae_t_lgb = rmse_mae(y_te, pred_te_lgb)
    print(f"[LGBM] val RMSE={rmse_v_lgb:.4f} | test RMSE={rmse_t_lgb:.4f} MAE={mae_t_lgb:.4f} | time={lgb_time:.1f}s")

    # -------- CatBoost --------
    print(">>> Training CatBoost ...")
    t0 = time.time()
    cb_params = tuned_params["catboost"] if tuned_params else None

    cb_model, cb_cols = train_catboost(
        train.drop(columns=["rating", "timestamp"]).copy(),
        y_tr,
        val.drop(columns=["rating", "timestamp"]).copy(),
        y_val,
        params=cb_params
    )
    cb_time = time.time() - t0

    pred_val_cb = cb_model.predict(val[cb_cols])
    
    # Measure Inference Time per user for CatBoost
    t_inf_cb = time.time()
    pred_te_cb = cb_model.predict(test[cb_cols])
    cb_inf_time = (time.time() - t_inf_cb) / test["userId"].nunique()

    rmse_v_cb, _ = rmse_mae(y_val, pred_val_cb)
    rmse_t_cb, mae_t_cb = rmse_mae(y_te, pred_te_cb)
    print(f"[CB]   val RMSE={rmse_v_cb:.4f} | test RMSE={rmse_t_cb:.4f} MAE={mae_t_cb:.4f} | time={cb_time:.1f}s")

    # -------- Select Best Model --------
    best_model_name = "LightGBM" if rmse_v_lgb < rmse_v_cb else "CatBoost"
    
    # -------- Calculate Ranking Metrics --------
    eval_df_lgb = test[["userId","movieId","rating"]].copy()
    eval_df_lgb["pred"] = pred_te_lgb
    metrics_lgb = evaluate_ranking_metrics(eval_df_lgb, k=10)

    eval_df_cb = test[["userId","movieId","rating"]].copy()
    eval_df_cb["pred"] = pred_te_cb
    metrics_cb = evaluate_ranking_metrics(eval_df_cb, k=10)

    # -------- Print A/B Comparison Table --------
    print("\n=== OFFLINE EVALUATION RESULTS (A/B COMPARISON) ===")
    print("| Metric | LightGBM Ranker | CatBoost Ranker (YetiRank) |")
    print("| :--- | :--- | :--- |")
    print(f"| **Hit Ratio@10** | {metrics_lgb['Hit Ratio@10']:.4f} | {metrics_cb['Hit Ratio@10']:.4f} |")
    print(f"| **NDCG@10** | {metrics_lgb['NDCG@10']:.4f} | {metrics_cb['NDCG@10']:.4f} |")
    print(f"| **MRR** | {metrics_lgb['MRR']:.4f} | {metrics_cb['MRR']:.4f} |")
    print(f"| **Train Time (sec)** | {lgb_time:.2f}s | {cb_time:.2f}s |")
    print(f"| **Inference Time (sec/user)** | {lgb_inf_time:.4f}s | {cb_inf_time:.4f}s |")

    print(f"\n>>> Best model based on Validation RMSE: {best_model_name}")

    plot_and_save_comparison(
        metrics_lgb=metrics_lgb,
        metrics_cb=metrics_cb,
        lgb_time=lgb_time,
        cb_time=cb_time,
        lgb_inf_time=lgb_inf_time,
        cb_inf_time=cb_inf_time,
        save_path="ranking_comparison_results.png"
    )

    # Save artifacts (save both, but note the best one)
    joblib.dump({
        "lgbm": lgb_model, "lgbm_cols": lgb_cols, 
        "lgb_category_maps": lgb_category_maps,  # Saved for recommend.py
        "catboost": cb_model, "cb_cols": cb_cols,
        "user_feats": user_feats, "movie_feats": movie_feats,
        "config": CFG,
        "best_model": best_model_name
    }, "models.joblib")
    print(">>> Saved models.joblib")

if __name__ == "__main__":
    main()