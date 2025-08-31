import os
import re
import json
import math
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# --------- Config ----------

CSV_PATH = os.getenv("CSV_PATH", "artifacts/v1/tracks_for_rec.csv")
TOPN_DEFAULT = 20
N_CANDIDATES = 10000       # initial candidate pool (after hard filters)
MMR_LAMBDA = 0.5           # 0.4–0.6 for more diversity
SEED = 42
np.random.seed(SEED)

# Track recently recommended songs to add penalty
_recent_recommendations = {}

# Features we’ll index
FEATURES = [
    "energy","tempo","danceability","loudness","liveness","valence",
    "speechiness","instrumentalness","mode","key","duration_ms",
    "acousticness","track_popularity"
]

# ---------- Prototypes (intent atoms) ----------
# Each prototype defines: seed phrases (for embedding), a target feature profile, per-feature weights, and optional hard constraints
PROTOS = {
  "heartbreak_sad": {
    "seeds": ["heartbroken", "lost love", "lonely and tender", "sad breakup songs", "melancholic and reflective"],
    "target": {"valence":0.15,"energy":0.20,"tempo":70,"acousticness":0.75,"danceability":0.30},
    "weights": {"valence":5,"energy":4,"tempo":2.5,"acousticness":3.5,"danceability":2.0},
    "hard": {}
  },
  "exam_stress": {
    "seeds": ["exam stress", "anxious study", "need calm focus", "deep work instrumentals", "concentration music"],
    "target": {"valence":0.50,"energy":0.30,"tempo":88,"instrumentalness":0.75,"speechiness":0.10},
    "weights": {"instrumentalness":3,"speechiness":2,"tempo":1.2,"energy":1.5,"valence":1.2},
    "hard": {"speechiness_max":0.25}
  },
  "focus_lofi": {
    "seeds": ["lofi beats", "study with me", "focus mode", "coding music", "no vocals please"],
    "target": {"valence":0.55,"energy":0.28,"tempo":85,"instrumentalness":0.8,"speechiness":0.08,"acousticness":0.5},
    "weights": {"instrumentalness":3,"speechiness":2.5,"tempo":1.2,"energy":1.3,"acousticness":1.5},
    "hard": {"speechiness_max":0.2}
  },
  "workout_hype": {
    "seeds": ["hype me up", "gym pump", "competitive match energy", "adrenaline rush", "powerful beats"],
    "target": {"valence":0.85,"energy":0.95,"tempo":150,"danceability":0.85,"loudness":-3},
    "weights": {"energy":5,"tempo":4,"danceability":3.5,"loudness":2.5},
    "hard": {}
  },
  "party_dance": {
    "seeds": ["party tonight", "dance floor", "friday night out", "club vibes", "let's dance"],
    "target": {"valence":0.85,"energy":0.8,"tempo":125,"danceability":0.8},
    "weights": {"valence":1,"energy":2.5,"tempo":2.5,"danceability":3},
    "hard": {}
  },
  "chill_evening": {
    "seeds": ["chill evening", "late night calm", "relaxing mellow", "unwind after work", "calm waves"],
    "target": {"valence":0.6,"energy":0.15,"tempo":65,"acousticness":0.8},
    "weights": {"valence":2,"energy":4,"tempo":3,"acousticness":4},
    "hard": {}
  },
  "angry_edgy": {
    "seeds": ["angry", "aggressive mood", "rage", "edgy intense", "thrash it"],
    "target": {"valence":0.3,"energy":0.9,"tempo":150,"loudness":-2},
    "weights": {"valence":1,"energy":3,"tempo":2.5,"loudness":2},
    "hard": {}
  },
  "happy_sunny": {
    "seeds": ["happy and bright", "sunny vibes", "feel good", "good mood", "optimistic upbeat"],
    "target": {"valence":0.95,"energy":0.8,"tempo":125,"danceability":0.8},
    "weights": {"valence":4,"energy":3,"tempo":2.5,"danceability":3.5},
    "hard": {}
  },
  "romantic": {
    "seeds": ["romantic date", "soft love songs", "warm and intimate", "sweet slow dance"],
    "target": {"valence":0.7,"energy":0.4,"tempo":90,"acousticness":0.5},
    "weights": {"valence":2,"energy":1,"tempo":1,"acousticness":1.5},
    "hard": {}
  },
  "nostalgic": {
    "seeds": ["nostalgic", "retro memories", "throwback feels", "bittersweet past"],
    "target": {"valence":0.55,"energy":0.5,"tempo":110,"danceability":0.55},
    "weights": {"valence":1.5,"danceability":1.2,"tempo":1.2,"energy":1.2},
    "hard": {}
  },
  "night_drive": {
    "seeds": ["night drive", "neon city", "synthwave", "cruising at night"],
    "target": {"valence":0.6,"energy":0.55,"tempo":115,"danceability":0.6,"liveness":0.15},
    "weights": {"tempo":1.5,"danceability":1.4,"energy":1.3,"liveness":1.2},
    "hard": {}
  },
  "calm_sleepy": {
    "seeds": ["sleepy calm", "soothing lullaby", "ambient sleep", "wind down"],
    "target": {"valence":0.5,"energy":0.15,"tempo":65,"acousticness":0.7},
    "weights": {"energy":3,"tempo":1.5,"acousticness":2},
    "hard": {"speechiness_max":0.2}
  }
}

# ---------- Sentence Transformer ----------
from sentence_transformers import SentenceTransformer
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
st_model = SentenceTransformer(ST_MODEL_NAME)

def embed(texts: List[str]) -> np.ndarray:
    # L2-normalized embeddings
    return st_model.encode(texts, normalize_embeddings=True)

# Precompute prototype embeddings (average of seed phrases)
for name, cfg in PROTOS.items():
    proto_emb = embed(cfg["seeds"]).mean(axis=0)
    proto_emb = proto_emb / (np.linalg.norm(proto_emb) + 1e-9)
    PROTOS[name]["embedding"] = proto_emb

# ---------- Data ingestion & scaling ----------
def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, np.ndarray]:
    df = pd.read_csv(csv_path)
    # Basic sanity: required columns
    missing = [c for c in FEATURES + ["track_name","track_artist"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df.dropna(subset=FEATURES + ["track_name","track_artist"]).copy()

    # Optional bounds/clipping
    df["tempo"] = df["tempo"].clip(40, 220)
    df["loudness"] = df["loudness"].clip(-35, 5)
    df["duration_ms"] = df["duration_ms"].clip(30_000, 600_000)
    df["track_popularity"] = df["track_popularity"].clip(0, 100)

    X = df[FEATURES].copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    meta = df[["track_name","track_artist"]].copy()
    return df, meta, scaler, Xs

DATA_DF, META_DF, SCALER, X_SCALED = load_dataset(CSV_PATH)
MEANS = SCALER.mean_
STDS = np.sqrt(SCALER.var_ + 1e-12)  # avoid /0

def inv_scale(Xs: np.ndarray) -> np.ndarray:
    return Xs * STDS + MEANS

X_ORIG = inv_scale(X_SCALED)  # for hard constraint checks quickly

# ---------- Intent blending ----------
def softmax(x: np.ndarray, temp: float = 0.15) -> np.ndarray:
    # cosine sims → soft weights
    x = np.asarray(x)
    # temperature scaling (higher temp → more uniform)
    z = (x - x.max()) / max(temp, 1e-6)
    ex = np.exp(z)
    return ex / (ex.sum() + 1e-12)

def prompt_to_intent(prompt: str, temperature: float = 0.15):
    q = embed([prompt])[0]
    keys = list(PROTOS.keys())
    sims = np.array([np.dot(q, PROTOS[k]["embedding"]) for k in keys])  # cosine since normalized
    mix = softmax(sims, temp=temperature)

    blended_target: Dict[str, float] = {}
    blended_weights: Dict[str, float] = {}
    hard: Dict[str, float] = {}

    for idx, k in enumerate(keys):
        alpha = float(mix[idx])
        cfg = PROTOS[k]
        for f, v in cfg["target"].items():
            blended_target[f] = blended_target.get(f, 0.0) + alpha * v
        for f, v in cfg["weights"].items():
            # additive weighting for better intent mixing
            blended_weights[f] = blended_weights.get(f, 0.0) + alpha * v
        # merge hard constraints conservatively
        h = cfg.get("hard", {})
        if "instrumentalness_min" in h:
            hard["instrumentalness_min"] = max(hard.get("instrumentalness_min", 0.0), h["instrumentalness_min"])
        if "speechiness_max" in h:
            hard["speechiness_max"] = min(hard.get("speechiness_max", 1.0), h["speechiness_max"])
        if "loudness_min" in h:
            hard["loudness_min"] = max(hard.get("loudness_min", -60.0), h["loudness_min"])

    # Keyword overrides (simple, optional)
    t = prompt.lower()
    if re.search(r"\b(instrumental|no lyrics|lo[- ]?fi)\b", t):
        hard["instrumentalness_min"] = max(hard.get("instrumentalness_min", 0.0), 0.6)
        blended_target["speechiness"] = 0.1
        blended_weights["instrumentalness"] = max(blended_weights.get("instrumentalness", 1.0), 2.0)
        blended_weights["speechiness"]   = max(blended_weights.get("speechiness", 1.0), 2.0)
    if any(word in t for word in ["metal","rock","punk","hardcore"]):
        blended_target["mode"] = 0
        hard["loudness_min"] = max(hard.get("loudness_min", -60.0), -8)

    mixture = dict(zip(keys, mix.tolist()))
    return blended_target, blended_weights, hard, mixture, sims.tolist()

# ---------- Query vector & distances ----------
DEFAULTS = {"valence":0.6,"energy":0.5,"tempo":110,"danceability":0.5,"acousticness":0.3,
            "instrumentalness":0.1,"speechiness":0.2,"loudness":-8,"liveness":0.2,"mode":1,"key":5,
            "duration_ms":210_000,"track_popularity":50}

def build_query_scaled(target: Dict[str, float]) -> np.ndarray:
    # Create scaled vector in feature order, filling missing with defaults
    merged = {**DEFAULTS, **target}
    q = np.zeros(len(FEATURES), dtype=float)
    for i, name in enumerate(FEATURES):
        v = merged.get(name, DEFAULTS.get(name, 0.0))
        q[i] = (v - MEANS[i]) / (STDS[i] + 1e-9)
    return q

def apply_hard_constraints(mask: np.ndarray, hard: Dict[str, float]) -> np.ndarray:
    # work in original space (unscaled)
    def fmin(name, val):
        if name in FEATURES:
            i = FEATURES.index(name)
            return X_ORIG[:, i] >= val
        return np.ones_like(mask, dtype=bool)

    def fmax(name, val):
        if name in FEATURES:
            i = FEATURES.index(name)
            return X_ORIG[:, i] <= val
        return np.ones_like(mask, dtype=bool)

    if "instrumentalness_min" in hard:
        mask &= fmin("instrumentalness", hard["instrumentalness_min"])
    if "speechiness_max" in hard:
        mask &= fmax("speechiness", hard["speechiness_max"])
    if "loudness_min" in hard:
        mask &= fmin("loudness", hard["loudness_min"])
    return mask

def weighted_dist_matrix(Xc_scaled: np.ndarray, q_scaled: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    w = np.ones(len(FEATURES))
    for k, wt in weights.items():
        if k in FEATURES:
            w[FEATURES.index(k)] = float(wt)
    sw = np.sqrt(w)  # Euclidean on scaled * sqrt(weights)
    diffs = (Xc_scaled - q_scaled) * sw
    d = np.linalg.norm(diffs, axis=1)
    return d

# ---------- Diversification (MMR) ----------
def cosine_sim_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # rows are vectors
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

def mmr_select(X_scaled_sub: np.ndarray, relevance: np.ndarray, k: int, lam: float = 0.7) -> List[int]:
    # Maximal Marginal Relevance: iteratively pick items balancing relevance & diversity
    n = len(relevance)
    if n == 0:
        return []
    selected = []
    candidates = set(range(n))
    # Precompute similarities
    sims = cosine_sim_rows(X_scaled_sub, X_scaled_sub)

    # Start with random selection from top-3 to add variety
    top3_idx = np.argpartition(relevance, -3)[-3:]
    first = int(np.random.choice(top3_idx))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < min(k, n) and candidates:
        best_score = -1e9
        best_cand = None
        for i in candidates:
            # diversity term: max similarity to already selected
            div = np.max(sims[i, selected]) if selected else 0.0
            score = lam * relevance[i] - (1 - lam) * div
            if score > best_score:
                best_score = score
                best_cand = i
        selected.append(best_cand)
        candidates.remove(best_cand)
    return selected

def add_recency_penalty(relevance: np.ndarray, idx_all: np.ndarray, penalty: float = 0.3) -> np.ndarray:
    """Add penalty to recently recommended songs"""
    global _recent_recommendations
    penalized = relevance.copy()
    
    for i, track_idx in enumerate(idx_all):
        track_key = f"{META_DF.iloc[track_idx]['track_name']}_{META_DF.iloc[track_idx]['track_artist']}"
        if track_key in _recent_recommendations:
            # Apply exponential decay penalty based on how recently it was recommended
            calls_ago = _recent_recommendations[track_key]
            penalty_factor = penalty * (0.7 ** calls_ago)  # decay over time
            penalized[i] -= penalty_factor
    return penalized

def update_recent_recommendations(results: List[Dict]) -> None:
    """Update recent recommendations tracking"""
    global _recent_recommendations
    # Age existing entries
    for key in list(_recent_recommendations.keys()):
        _recent_recommendations[key] += 1
        # Remove entries older than 10 calls
        if _recent_recommendations[key] > 10:
            del _recent_recommendations[key]
    
    # Add new recommendations
    for track in results:
        track_key = f"{track['track_name']}_{track['track_artist']}"
        _recent_recommendations[track_key] = 0

# ---------- Main recommend ----------
def recommend_from_prompt(prompt: str, topn: int = TOPN_DEFAULT) -> Dict:
    t0 = time.time()
    target, weights, hard, mixture, sims = prompt_to_intent(prompt)

    q_scaled = build_query_scaled(target)

    # Hard constraints
    mask = np.ones(len(X_SCALED), dtype=bool)
    mask = apply_hard_constraints(mask, hard)

    # Reduce to a candidate set quickly (optional: use random subsample if too large)
    idx_all = np.nonzero(mask)[0]
    if len(idx_all) == 0:
        return {
            "prompt": prompt,
            "mixture": mixture,
            "hard": hard,
            "results": [],
            "timing_ms": round((time.time()-t0)*1000, 1),
            "notes": "No items after hard-constraints."
        }

    # Compute weighted distance on the filtered pool
    Xc = X_SCALED[idx_all]
    d = weighted_dist_matrix(Xc, q_scaled, weights)

    # Popularity blend: pop ∈ [0..1]
    pop_i = FEATURES.index("track_popularity")
    pop = (X_ORIG[idx_all, pop_i] - 0) / (100.0 + 1e-9)

    # Higher is better, reduced popularity weight
    rel = -d + 0.05 * pop
    
    # Add recency penalty to avoid repetitive recommendations
    rel = add_recency_penalty(rel, idx_all)

    # Take top candidates by rel first
    k0 = min(N_CANDIDATES, len(idx_all))
    top_idx = np.argpartition(rel, -k0)[-k0:]
    # Sort them
    top_idx = top_idx[np.argsort(-rel[top_idx])]

    # Diversify with MMR on the candidate subset
    Xsub = Xc[top_idx]
    relsub = rel[top_idx]
    take = mmr_select(Xsub, relsub, k=topn, lam=MMR_LAMBDA)

    final_rows = [idx_all[top_idx[i]] for i in take]
    out = []
    for j in final_rows:
        out.append({
            "track_name": str(META_DF.iloc[j]["track_name"]),
            "track_artist": str(META_DF.iloc[j]["track_artist"])
        })
    
    # Update recent recommendations tracking
    update_recent_recommendations(out)

    return {
        "prompt": prompt,
        "mixture": mixture,     # prototype mixture weights
        "hard": hard,           # applied hard constraints
        "results": out,
        "timing_ms": round((time.time()-t0)*1000, 1)
    }
