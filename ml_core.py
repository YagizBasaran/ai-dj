# ml_engine.py
import os, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

# ---------- Module-level state ----------
_ARTIFACTS_DIR = None
_rf_bundle = None          # {"model": RandomForestClassifier, "scaler": StandardScaler}
_numeric_features = None   # list[string]
_tracks_df = None          # pandas.DataFrame

# ---------- Public getters (optional) for Debugggingggg ----------
def tracks_df() -> Optional[pd.DataFrame]:
    return _tracks_df

def numeric_features() -> Optional[list]:
    return _numeric_features

def has_model() -> bool:
    return (
        _rf_bundle is not None
        and isinstance(_rf_bundle, dict)
        and "model" in _rf_bundle
        and "scaler" in _rf_bundle
        and _numeric_features is not None
        and _tracks_df is not None
        and len(_tracks_df) > 0
    )

# ---------- Load artifacts ----------
def load_artifacts(artifacts_dir: Optional[str] = None) -> None:
    global _ARTIFACTS_DIR, _rf_bundle, _numeric_features, _tracks_df

    model_version = artifacts_dir or os.getenv("MODEL_VERSION", "v2")
    
    # Shared files
    shared_path = Path("artifacts/shared")
    model_path = Path("artifacts/models") / model_version / "model.pkl"

    try:
        _rf_bundle = joblib.load(model_path) if model_path.exists() else None
        
        feats_path = shared_path / "numeric_features.json"
        if feats_path.exists():
            with open(feats_path, "r", encoding="utf-8") as f:
                _numeric_features = json.load(f)
        else:
            _numeric_features = None

        tracks_path = shared_path / "tracks_for_rec.csv"
        _tracks_df = pd.read_csv(tracks_path) if tracks_path.exists() else None
        
    except Exception as e:
        print(f"[WARN] Could not load artifacts: {e}")
        _rf_bundle = None
        _numeric_features = None
        _tracks_df = None

# Eager-load once on import (optional). can also call load_artifacts() from app.py.
# load_artifacts()

# ---------- Helpers ----------
def _class_index(model, target_label: str) -> Optional[int]:
    try:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return None
        hits = np.where(classes == target_label)[0]
        return int(hits[0]) if len(hits) else None
    except Exception:
        return None

def _normalize_0_1(series: pd.Series) -> pd.Series:
    vmin, vmax = series.min(), series.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - vmin) / (vmax - vmin)

def _diversify_by_artist(df: pd.DataFrame, artist_col="track_artist", top_k=20, per_artist_cap=2) -> pd.DataFrame:
    out = []
    counts = {}
    for _, row in df.iterrows():
        a = str(row.get(artist_col, "")).strip().lower()
        c = counts.get(a, 0)
        if c < per_artist_cap:
            out.append(row)
            counts[a] = c + 1
        if len(out) >= top_k:
            break
    return pd.DataFrame(out)

# Optional: light mapping to handle minor label mismatches between parser/model
_LABEL_SYNONYMS = {
    "energetic": "hype",
    "calm": "chill",
    "relaxed": "chill",
    "rage": "angry",
    "love": "romantic",
    "general": "happy",
}

def _resolve_model_label(target: str, model_classes) -> str:
    """
    Returns a label that exists in model_classes. Tries exact, then synonyms, else fallback.
    """
    if target in model_classes:
        return target
    mapped = _LABEL_SYNONYMS.get(target, target)
    if mapped in model_classes:
        return mapped
    # final fallback
    return "happy" if "happy" in model_classes else model_classes[0]

def mood_from_prompt(text: str) -> str:
    t = (text or "").lower()
    # expanded sad patterns
    if any(k in t for k in [
        "heartbroken","broken","broke up","break up","breakup","cry","tears","depressed","sad"
    ]):
        return "sad"
    if any(k in t for k in ["gym","hype","pump","workout","competitive","match"]):
        return "hype"
    if any(k in t for k in ["romantic","date","love"]):
        return "romantic"
    if any(k in t for k in ["study","focus","concentrate"]):
        return "focus"
    if any(k in t for k in ["angry","rage"]):
        return "angry"
    if any(k in t for k in ["chill","calm","lofi","relax"]):
        return "chill"
    if any(k in t for k in ["happy","good mood","feel good"]):
        return "happy"
    return "happy"

def rank_candidates_with_rf(candidates: pd.DataFrame, target_mood: str, top_k: int = 20) -> pd.DataFrame:
    if not has_model():
        sub = candidates.sort_values("track_popularity", ascending=False)
        return sub.drop_duplicates(subset=["track_name", "track_artist"]).head(top_k)

    model = _rf_bundle["model"]
    scaler = _rf_bundle["scaler"]
    classes = list(getattr(model, "classes_", []))
    target = _resolve_model_label(str(target_mood), classes)
    mood_idx = _class_index(model, target)

    # --- NEW: fill any missing numeric feature columns with 0.0 and keep going
    df = candidates.copy()
    for col in _numeric_features:
        if col not in df.columns:
            df[col] = 0.0
    # ensure numeric dtype
    X = df[_numeric_features].astype(float).to_numpy()
    try:
        Xs = scaler.transform(X)
    except Exception:
        Xs = X

    proba = model.predict_proba(Xs)
    ml_score = proba[:, mood_idx] if proba.ndim == 2 else proba

    df["ml_score"] = ml_score
    df["pop_norm"] = _normalize_0_1(df["track_popularity"].fillna(0))
    # push ML to dominate during dev so you SEE the difference
    df["final_score"] = 0.95 * df["ml_score"] + 0.05 * df["pop_norm"]

    df = df.sort_values("final_score", ascending=False)
    df = df.drop_duplicates(subset=["track_name", "track_artist"], keep="first")
    df = _diversify_by_artist(df, artist_col="track_artist", top_k=top_k, per_artist_cap=2)
    return df.head(top_k)


def recommend_by_mood(mood: str, top_k: int = 20):
    if _tracks_df is None or len(_tracks_df) == 0:
        return []

    # BROAD pool; later we’ll add era/genre filters here instead of mood
    pool = _tracks_df
    # For speed in dev:
    # pool = _tracks_df.sample(n=min(2000, len(_tracks_df)), random_state=42)

    ranked = rank_candidates_with_rf(pool, target_mood=mood, top_k=top_k)

    cols = ["track_name", "track_artist", "mood", "track_popularity"]
    for extra in ["ml_score", "final_score"]:
        if extra in ranked.columns:
            cols.append(extra)
    return ranked[cols].to_dict(orient="records")



# print(">>> ARTIFACTS DIR:", _ARTIFACTS_DIR)
# print(">>> MODEL PATH EXISTS:", (Path(_ARTIFACTS_DIR) / "model.pkl").exists())
