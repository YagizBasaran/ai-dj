# ml_engine.py
import os, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
import google.generativeai as genai

import re  # For improved direct search
from typing import Optional, List, Dict #direct search imports

# ---------- Module-level state ----------
_ARTIFACTS_DIR = None
_rf_bundle = None          # {"model": RandomForestClassifier, "scaler": StandardScaler}
_numeric_features = None   # list[string]
_tracks_df = None          # pandas.DataFrame

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

        # tracks_path = shared_path / "tracks_for_rec.csv"
        tracks_path = shared_path / "combined_dataset 2.csv"
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
    """
    Use Gemini API to classify user prompt into one of 8 predefined moods.
    Returns one of: sad, angry, happy, hype, romantic, chill, focus, live
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

        # Validate the response is one of our predefined moods
        valid_moods = ["sad", "angry", "happy", "hype", "romantic", "chill", "focus", "live"]
        if detected_mood in valid_moods:
            return detected_mood
        else:
            # Fallback if Gemini returns something unexpected
            return "other unexpected" #again gemini errored unexpectedly

    except Exception as e:
        print(f"[WARN] Gemini API error: {e}")
        # Fallback to simple keyword matching if API fails
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
        return "other" #Other means gemini errored and no keywords matched

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

#CLASSIC FIRST VERSION, !!!!UNUSED!!!!
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

# ---------- NEW: Direct Search Function ---------- Comes before emotion based search
def search_direct_matches(query: str, top_k: int = 2) -> List[Dict]:
    """
    Searches for direct matches in track name, artist name, or album name.
    Uses word tokenization for better natural language matching.
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
    
    # Extract meaningful words (ignore common stop words)
    stop_words = {'i', 'am', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
                  'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
                  'so', 'very', 'really', 'just', 'feel', 'feeling', 'want', 'like'}
    
    # Split into words and filter
    words = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]
    meaningful_words = [w for w in words if w not in stop_words]
    
    # If no meaningful words, try the full query
    if not meaningful_words:
        search_queries = [query_lower]
    else:
        # Try combinations: full phrase, then individual meaningful words
        search_queries = [query_lower] + meaningful_words
    
    print(f"Search queries: {search_queries}")
    
    # Make a copy to work with
    df = _tracks_df.copy()
    
    # Create searchable columns
    df['track_name_lower'] = df['track_name'].fillna('').astype(str).str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df['track_artist_lower'] = df['track_artist'].fillna('').astype(str).str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df['track_album_name_lower'] = df.get('track_album_name', pd.Series([''] * len(df))).fillna('').astype(str).str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    
    # Score function that handles multiple search terms
    def calculate_combined_score(text: str, queries: list) -> float:
        if not text:
            return 0.0
        
        max_score = 0.0
        
        # First, try the full original query (highest priority)
        full_query = queries[0] if queries else ""
        if full_query and full_query in text:
            # Bonus for full phrase match
            if text == full_query:
                return 100.0
            elif text.startswith(full_query):
                return 80.0
            else:
                max_score = 50.0
        
        # Then try individual words, but score based on HOW MANY words match
        if len(queries) > 1:
            individual_words = queries[1:]  # Skip the full query
            words_matched = sum(1 for word in individual_words if word in text)
            total_words = len(individual_words)
            
            if words_matched > 0:
                # Score based on percentage of words matched
                # If all words match: 60 points, if half match: 30 points
                word_score = 60.0 * (words_matched / total_words)
                
                # Bonus if multiple words match (not just one)
                if words_matched >= 2:
                    word_score += 20.0
                
                max_score = max(max_score, word_score)
        
        return max_score
    
    # Score each row
    df['name_score'] = df['track_name_lower'].apply(lambda x: calculate_combined_score(x, search_queries))
    df['artist_score'] = df['track_artist_lower'].apply(lambda x: calculate_combined_score(x, search_queries))
    df['album_score'] = df['track_album_name_lower'].apply(lambda x: calculate_combined_score(x, search_queries))
    
    # Combine scores (prioritize track name > artist > album)
    df['match_score'] = (
        df['name_score'] * 2.0 +
        df['artist_score'] * 1.5 +
        df['album_score'] * 1.0
    )
    
    # Filter to only matches
    matches = df[df['match_score'] > 0].copy()
    print(f"Found {len(matches)} potential matches")
    
    if len(matches) == 0:
        print("No matches found!")
        return []
    
    # Show top matches for debugging
    print(f"Top 5 matches:")
    for idx, row in matches.head(5).iterrows():
        print(f"  - {row['track_name']} by {row['track_artist']} (score: {row['match_score']:.1f})")
    
    # Add popularity bonus
    if 'track_popularity' in matches.columns:
        pop_normalized = _normalize_0_1(matches['track_popularity'].fillna(0))
        matches['final_score'] = matches['match_score'] + (pop_normalized * 5.0)
    else:
        matches['final_score'] = matches['match_score']
    
    # Sort and deduplicate
    matches = matches.sort_values('final_score', ascending=False)
    matches = matches.drop_duplicates(subset=['track_name', 'track_artist'], keep='first')
    
    # Get top_k results
    top_matches = matches.head(top_k)
    print(f"Returning {len(top_matches)} results")
    
    # Format output
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

#Current new one 7/3 ratio for popularity preference
#Current new one 7/3 ratio for popularity preference + DIRECT SEARCH
def recommend_by_mood_with_preference(mood: str, preference: str = "balanced", top_k: int = 10, user_query: str = ""):
    """
    Recommend songs with Discovery/Classic preference + Direct Search matches
    
    Args:
        mood: detected mood
        preference: "discovery", "mainstream", or "balanced"
        top_k: number of mood-based recommendations (default 10)
        user_query: original user input for direct search
    
    Returns:
        List with up to 2 direct matches at top + 10 mood-based recommendations
    """
    if _tracks_df is None or len(_tracks_df) == 0:
        return []
    
    # 1. FIRST: Get direct matches (if query provided)
    direct_matches = []
    if user_query and user_query.strip():
        direct_matches = search_direct_matches(user_query, top_k=2)
        print(f"[DEBUG] Found {len(direct_matches)} direct matches for query: '{user_query}'")
    
    # 2. SECOND: Get mood-based recommendations
    pool = _tracks_df
    ranked = rank_candidates_with_rf(pool, target_mood=mood, top_k=50)  # Get more candidates
    
    # If we have direct matches, exclude them from mood-based results to avoid duplicates
    if direct_matches:
        direct_track_ids = [(m['track_name'], m['track_artist']) for m in direct_matches]
        ranked = ranked[~ranked[['track_name', 'track_artist']].apply(tuple, axis=1).isin(direct_track_ids)]
    
    if len(ranked) == 0:
        # If no mood matches, just return direct matches
        return direct_matches
    
    # Define popularity threshold
    popularity_threshold = ranked["track_popularity"].median()
    
    # Split into popular and non-popular
    popular_songs = ranked[ranked["track_popularity"] >= popularity_threshold]
    non_popular_songs = ranked[ranked["track_popularity"] < popularity_threshold]
    
    # Shuffle for randomness
    popular_songs = popular_songs.sample(frac=1, random_state=None).reset_index(drop=True)
    non_popular_songs = non_popular_songs.sample(frac=1, random_state=None).reset_index(drop=True)
    
    # Apply preference logic
    mood_based_results = []
    if preference == "discovery":
        # 7 non-popular + 3 popular
        mood_based_results.extend(non_popular_songs.head(7).to_dict(orient="records"))
        mood_based_results.extend(popular_songs.head(3).to_dict(orient="records"))
    elif preference == "mainstream":
        # 7 popular + 3 non-popular  
        mood_based_results.extend(popular_songs.head(7).to_dict(orient="records"))
        mood_based_results.extend(non_popular_songs.head(3).to_dict(orient="records"))
    else:  # balanced
        mood_based_results = ranked.head(top_k).to_dict(orient="records")
    
    # Limit to top_k
    mood_based_results = mood_based_results[:top_k]
    
    # 3. COMBINE: Direct matches at top + mood-based below
    final_results = direct_matches + mood_based_results
    
    print(f"[DEBUG] Returning {len(direct_matches)} direct + {len(mood_based_results)} mood = {len(final_results)} total")
    return final_results

# print(">>> ARTIFACTS DIR:", _ARTIFACTS_DIR)
# print(">>> MODEL PATH EXISTS:", (Path(_ARTIFACTS_DIR) / "model.pkl").exists())
