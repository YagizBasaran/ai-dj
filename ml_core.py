"""
ML Core - Music Recommendation Engine
Handles mood detection, ML-based recommendations, and direct search functionality
"""

import os
import json
import re
import joblib
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import google.generativeai as genai

# ============================================================================
# MODULE-LEVEL STATE
# ============================================================================

_ARTIFACTS_DIR = None
_rf_bundle = None
_numeric_features = None
_tracks_df = None

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ============================================================================
# PUBLIC GETTERS (for debugging)
# ============================================================================

def tracks_df() -> Optional[pd.DataFrame]:
    """Returns the loaded tracks dataframe"""
    return _tracks_df

def numeric_features() -> Optional[list]:
    """Returns the list of numeric features used by the model"""
    return _numeric_features

def has_model() -> bool:
    """Check if all required ML artifacts are loaded"""
    return (
        _rf_bundle is not None
        and isinstance(_rf_bundle, dict)
        and "model" in _rf_bundle
        and "scaler" in _rf_bundle
        and _numeric_features is not None
        and _tracks_df is not None
        and len(_tracks_df) > 0
    )

# ============================================================================
# ARTIFACT LOADING
# ============================================================================

def load_artifacts(artifacts_dir: Optional[str] = None) -> None:
    """
    Load ML artifacts (model, scaler, features, tracks dataset)

    Args:
        artifacts_dir: Model version directory (default: v2)
    """
    global _ARTIFACTS_DIR, _rf_bundle, _numeric_features, _tracks_df

    model_version = artifacts_dir or os.getenv("MODEL_VERSION", "v2")

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

        tracks_path = shared_path / "combined_dataset 2.csv"
        _tracks_df = pd.read_csv(tracks_path) if tracks_path.exists() else None

    except Exception as e:
        print(f"[WARN] Could not load artifacts: {e}")
        _rf_bundle = None
        _numeric_features = None
        _tracks_df = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _class_index(model, target_label: str) -> Optional[int]:
    """Get the index of a target label in model classes"""
    try:
        classes = getattr(model, "classes_", None)
        if classes is None:
            return None
        hits = np.where(classes == target_label)[0]
        return int(hits[0]) if len(hits) else None
    except Exception:
        return None

def _normalize_0_1(series: pd.Series) -> pd.Series:
    """Normalize a pandas series to 0-1 range"""
    vmin, vmax = series.min(), series.max()
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - vmin) / (vmax - vmin)

def _diversify_by_artist(df: pd.DataFrame, artist_col="track_artist", top_k=20, per_artist_cap=2) -> pd.DataFrame:
    """
    Diversify results by limiting tracks per artist

    Args:
        df: DataFrame with tracks
        artist_col: Column name for artist
        top_k: Total number of tracks to return
        per_artist_cap: Maximum tracks per artist
    """
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

# ============================================================================
# MOOD DETECTION
# ============================================================================

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
    Resolve target mood to a valid model class label
    Tries exact match first, then synonyms, then fallback
    """
    if target in model_classes:
        return target
    mapped = _LABEL_SYNONYMS.get(target, target)
    if mapped in model_classes:
        return mapped
    return "happy" if "happy" in model_classes else model_classes[0]

def mood_from_prompt(text: str) -> str:
    """
    Use Gemini API to classify user prompt into one of 8 predefined moods

    Args:
        text: User input text

    Returns:
        One of: sad, angry, happy, hype, romantic, chill, focus, live
    """
    if not text or not text.strip():
        return "happy"

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
Analyze the following user input and classify it into EXACTLY ONE of these 8 moods:
- sad: melancholy, heartbreak, depression, crying, loss
- angry: rage, frustration, mad, furious, upset
- happy: joyful, cheerful, upbeat, positive, good mood
- hype: energetic, pump up, workout, competitive, intense, excited
- romantic: love, date, intimate, passionate, couple time
- chill: relaxed, calm, laid-back, lofi, peaceful, easy-going
- focus: study, concentrate, work, productivity, deep work
- live: party, celebration, social, dancing, going out

User input: "{text}"

Respond with ONLY the mood word (sad, angry, happy, hype, romantic, chill, focus, or live). No explanation needed.
"""

        response = model.generate_content(prompt)
        detected_mood = response.text.strip().lower()

        valid_moods = ["sad", "angry", "happy", "hype", "romantic", "chill", "focus", "live"]
        if detected_mood in valid_moods:
            return detected_mood
        else:
            return "other unexpected"

    except Exception as e:
        print(f"[WARN] Gemini API error: {e}")
        t = (text or "").lower()
        if any(k in t for k in ["heartbroken","broken","broke up","break up","breakup","cry","tears","depressed","sad"]):
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
        if any(k in t for k in ["party","celebration","dancing","going out"]):
            return "live"
        if any(k in t for k in ["happy","good mood","feel good"]):
            return "happy"
        return "other"

# ============================================================================
# ML RANKING
# ============================================================================

def rank_candidates_with_rf(candidates: pd.DataFrame, target_mood: str, top_k: int = 20) -> pd.DataFrame:
    """
    Rank candidate tracks using Random Forest model

    Args:
        candidates: DataFrame with candidate tracks
        target_mood: Target mood for ranking
        top_k: Number of top tracks to return

    Returns:
        DataFrame with ranked tracks and scores
    """
    if not has_model():
        sub = candidates.sort_values("track_popularity", ascending=False)
        return sub.drop_duplicates(subset=["track_name", "track_artist"]).head(top_k)

    model = _rf_bundle["model"]
    scaler = _rf_bundle["scaler"]
    classes = list(getattr(model, "classes_", []))
    target = _resolve_model_label(str(target_mood), classes)
    mood_idx = _class_index(model, target)

    df = candidates.copy()
    for col in _numeric_features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[_numeric_features].astype(float).to_numpy()
    try:
        Xs = scaler.transform(X)
    except Exception:
        Xs = X

    proba = model.predict_proba(Xs)
    ml_score = proba[:, mood_idx] if proba.ndim == 2 else proba

    df["ml_score"] = ml_score
    df["pop_norm"] = _normalize_0_1(df["track_popularity"].fillna(0))
    df["final_score"] = 0.95 * df["ml_score"] + 0.05 * df["pop_norm"]

    df = df.sort_values("final_score", ascending=False)
    df = df.drop_duplicates(subset=["track_name", "track_artist"], keep="first")
    df = _diversify_by_artist(df, artist_col="track_artist", top_k=top_k, per_artist_cap=2)
    return df.head(top_k)

# ============================================================================
# DIRECT SEARCH
# ============================================================================

def search_direct_matches(query: str, top_k: int = 2) -> List[Dict]:
    """
    Search for direct matches in track name, artist name, or album name
    Uses word tokenization for better natural language matching

    Args:
        query: User search query
        top_k: Number of matches to return

    Returns:
        List of matched tracks with scores
    """
    print(f"\n=== DIRECT SEARCH DEBUG ===")
    print(f"Query received: '{query}'")
    print(f"Tracks available: {len(_tracks_df) if _tracks_df is not None else 0}")

    if _tracks_df is None or len(_tracks_df) == 0:
        print("No tracks loaded!")
        return []

    if not query or not query.strip():
        print("Empty query!")
        return []

    query_lower = query.strip().lower()
    query_lower = re.sub(r'[^\w\s]', '', query_lower)

    stop_words = {'i', 'am', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
                  'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
                  'so', 'very', 'really', 'just', 'feel', 'feeling', 'want', 'like'}

    words = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]
    meaningful_words = [w for w in words if w not in stop_words]

    if not meaningful_words:
        search_queries = [query_lower]
    else:
        search_queries = [query_lower] + meaningful_words

    print(f"Search queries: {search_queries}")

    df = _tracks_df.copy()
    df['track_name_lower'] = df['track_name'].fillna('').astype(str).str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df['track_artist_lower'] = df['track_artist'].fillna('').astype(str).str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df['track_album_name_lower'] = df.get('track_album_name', pd.Series([''] * len(df))).fillna('').astype(str).str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))

    def calculate_combined_score(text: str, queries: list) -> float:
        """Calculate match score for text against multiple queries"""
        if not text:
            return 0.0

        max_score = 0.0

        full_query = queries[0] if queries else ""
        if full_query and full_query in text:
            if text == full_query:
                return 100.0
            elif text.startswith(full_query):
                return 80.0
            else:
                max_score = 50.0

        if len(queries) > 1:
            individual_words = queries[1:]
            words_matched = sum(1 for word in individual_words if word in text)
            total_words = len(individual_words)

            if words_matched > 0:
                word_score = 60.0 * (words_matched / total_words)
                if words_matched >= 2:
                    word_score += 20.0
                max_score = max(max_score, word_score)

        return max_score

    df['name_score'] = df['track_name_lower'].apply(lambda x: calculate_combined_score(x, search_queries))
    df['artist_score'] = df['track_artist_lower'].apply(lambda x: calculate_combined_score(x, search_queries))
    df['album_score'] = df['track_album_name_lower'].apply(lambda x: calculate_combined_score(x, search_queries))

    df['match_score'] = (
        df['name_score'] * 2.0 +
        df['artist_score'] * 1.5 +
        df['album_score'] * 1.0
    )

    matches = df[df['match_score'] > 0].copy()
    print(f"Found {len(matches)} potential matches")

    if len(matches) == 0:
        print("No matches found!")
        return []

    print(f"Top 5 matches:")
    for idx, row in matches.head(5).iterrows():
        print(f"  - {row['track_name']} by {row['track_artist']} (score: {row['match_score']:.1f})")

    if 'track_popularity' in matches.columns:
        pop_normalized = _normalize_0_1(matches['track_popularity'].fillna(0))
        matches['final_score'] = matches['match_score'] + (pop_normalized * 5.0)
    else:
        matches['final_score'] = matches['match_score']

    matches = matches.sort_values('final_score', ascending=False)
    matches = matches.drop_duplicates(subset=['track_name', 'track_artist'], keep='first')

    top_matches = matches.head(top_k)
    print(f"Returning {len(top_matches)} results")

    cols = ['track_name', 'track_artist', 'mood', 'track_popularity']
    if 'track_album_name' in top_matches.columns:
        cols.append('track_album_name')
    for extra in ['match_score', 'final_score']:
        if extra in top_matches.columns:
            cols.append(extra)

    available_cols = [c for c in cols if c in top_matches.columns]
    results = top_matches[available_cols].to_dict(orient='records')

    for r in results:
        r['is_direct_match'] = True

    print(f"=== END DIRECT SEARCH DEBUG ===\n")
    return results

# ============================================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================================

def recommend_by_mood_with_preference(mood: str, preference: str = "balanced", top_k: int = 10, user_query: str = "", user_preferences: dict = None) -> List[Dict]:
    """
    Main recommendation function with Discovery/Mainstream preference + Direct Search + User Preferences

    Args:
        mood: Detected mood
        preference: "discovery", "mainstream", or "balanced"
        top_k: Number of mood-based recommendations
        user_query: Original user input for direct search
        user_preferences: User's liked moods and disliked songs from localStorage

    Returns:
        List with up to 2 direct matches + 10 mood-based recommendations
    """
    if _tracks_df is None or len(_tracks_df) == 0:
        return []

    user_preferences = user_preferences or {}
    liked_moods = user_preferences.get('liked_moods', [])
    disliked_songs = user_preferences.get('disliked_songs', [])

    direct_matches = []
    if user_query and user_query.strip():
        direct_matches = search_direct_matches(user_query, top_k=2)
        print(f"[DEBUG] Found {len(direct_matches)} direct matches for query: '{user_query}'")

    pool = _tracks_df

    if disliked_songs:
        pool = pool[~pool['track_name'].str.lower().isin([s.lower() for s in disliked_songs])]
        print(f"[DEBUG] Filtered out {len(disliked_songs)} disliked songs")

    ranked = rank_candidates_with_rf(pool, target_mood=mood, top_k=50)

    # Apply subtle boost to liked moods (move them up slightly, not to the top)
    if liked_moods and len(liked_moods) > 0:
        # Create a small boost score for liked moods (0.2 means ~20% boost)
        ranked['liked_mood_boost'] = ranked['mood'].isin(liked_moods).astype(float) * 0.2
        ranked = ranked.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle first
        ranked = ranked.sort_values('liked_mood_boost', ascending=False).reset_index(drop=True)
        ranked = ranked.drop('liked_mood_boost', axis=1)
        print(f"[DEBUG] Applied subtle boost to tracks from liked moods: {liked_moods}")

    if direct_matches:
        direct_track_ids = [(m['track_name'], m['track_artist']) for m in direct_matches]
        ranked = ranked[~ranked[['track_name', 'track_artist']].apply(tuple, axis=1).isin(direct_track_ids)]

    if len(ranked) == 0:
        return direct_matches

    popularity_threshold = ranked["track_popularity"].median()
    popular_songs = ranked[ranked["track_popularity"] >= popularity_threshold]
    non_popular_songs = ranked[ranked["track_popularity"] < popularity_threshold]

    popular_songs = popular_songs.sample(frac=1, random_state=None).reset_index(drop=True)
    non_popular_songs = non_popular_songs.sample(frac=1, random_state=None).reset_index(drop=True)

    mood_based_results = []
    if preference == "discovery":
        mood_based_results.extend(non_popular_songs.head(7).to_dict(orient="records"))
        mood_based_results.extend(popular_songs.head(3).to_dict(orient="records"))
    elif preference == "mainstream":
        mood_based_results.extend(popular_songs.head(7).to_dict(orient="records"))
        mood_based_results.extend(non_popular_songs.head(3).to_dict(orient="records"))
    else:
        mood_based_results = ranked.head(top_k).to_dict(orient="records")

    mood_based_results = mood_based_results[:top_k]

    final_results = direct_matches + mood_based_results

    print(f"[DEBUG] Returning {len(direct_matches)} direct + {len(mood_based_results)} mood = {len(final_results)} total")
    return final_results
