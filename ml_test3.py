# # AI DJ – Flask app (No anchors/keywords). Pure LLM → feature vector → nearest tracks
# #
# # Reduced label set: we drop track_id, number, duration_ms, explicit, key, time_signature.
# # Remaining feature set is continuous/categorical audio features useful for similarity.

# from __future__ import annotations
# import argparse
# import json
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional

# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics.pairwise import cosine_similarity

# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None

# ###############################################################################
# # 1) Reduced label set
# ###############################################################################
# REDUCED_FEATURES: List[str] = [
#     "danceability",
#     "energy",
#     "valence",
#     "acousticness",
#     "instrumentalness",
#     "speechiness",
#     "liveness",
#     "tempo",   # scaled BPM
#     "mode"     # 0/1 minor/major
# ]

# RETURN_COLS = [
#     "track_name", "artists", "album_name", "track_genre", "popularity",
#     "danceability", "energy", "valence", "acousticness", "instrumentalness",
#     "speechiness", "liveness", "tempo", "mode"
# ]

# ###############################################################################
# # 2) LLM extraction – strict JSON, no defaults
# ###############################################################################

# LLM_SYSTEM = (
#     "You are a precise music feature extractor. Convert the user's free text prompt "
#     "into continuous numeric audio feature values without using defaults or rounding. "
#     "If the prompt does not provide enough information to infer a feature, return null. "
#     "Never fabricate values."
# )

# def build_user_prompt(prompt: str, features: List[str], minmax: Dict[str, Tuple[float, float]]) -> str:
#     spec = {
#         "notes": [
#             "Return a JSON object with exactly these feature keys.",
#             "Values should be continuous and within the dataset range.",
#             "Use null if the prompt does not imply the feature.",
#             "tempo is BPM if provided, otherwise null if not clear.",
#             "mode is 0 for minor, 1 for major; null if uncertain.",
#         ],
#         "features": {f: {
#             "range": list(minmax.get(f, (0.0, 1.0))) if f != "mode" else [0,1],
#             "type": "number" if f != "mode" else "integer"
#         } for f in features}
#     }
#     return json.dumps({
#         "instruction": "Extract numeric values, or null when uncertain.",
#         "feature_spec": spec,
#         "prompt": prompt,
#         "output_schema": {f: None for f in features},
#         "output_must_be_json_object_with_only_these_keys": features,
#     }, ensure_ascii=False)

# @dataclass
# class Recommender:
#     df: pd.DataFrame
#     features: List[str]
#     scaler: MinMaxScaler
#     track_matrix: np.ndarray
#     feature_minmax: Dict[str, Tuple[float, float]]
#     openai_client: Optional[OpenAI]
#     model_name: str = "gpt-4o-mini"

#     @classmethod
#     def from_csv(cls, csv_path: str, features: List[str]) -> "Recommender":
#         df = pd.read_csv(csv_path)
#         missing = [c for c in features if c not in df.columns]
#         if missing:
#             raise ValueError(f"Missing columns in CSV: {missing}")
#         scaler = MinMaxScaler()
#         X = df[features].astype(float).values
#         X_scaled = scaler.fit_transform(X)
#         feature_minmax = {f: (float(df[f].min()), float(df[f].max())) for f in features}
#         client = OpenAI() if OpenAI is not None else None
#         return cls(df=df, features=features, scaler=scaler, track_matrix=X_scaled,
#                    feature_minmax=feature_minmax, openai_client=client)

#     def prompt_to_vector(self, prompt: str) -> Dict[str, Optional[float]]:
#         if self.openai_client is None:
#             raise RuntimeError("OpenAI SDK not available. Install 'openai' and set OPENAI_API_KEY.")
#         user_payload = build_user_prompt(prompt, self.features, self.feature_minmax)
#         resp = self.openai_client.chat.completions.create(
#             model=self.model_name,
#             temperature=0,
#             messages=[
#                 {"role": "system", "content": LLM_SYSTEM},
#                 {"role": "user", "content": user_payload},
#             ],
#             response_format={"type": "json_object"},
#         )
#         raw = resp.choices[0].message.content
#         try:
#             data = json.loads(raw)
#         except Exception:
#             data = {f: None for f in self.features}
#         out: Dict[str, Optional[float]] = {}
#         for f in self.features:
#             v = data.get(f, None)
#             if v is None:
#                 out[f] = None
#                 continue
#             try:
#                 out[f] = float(v)
#             except Exception:
#                 out[f] = None
#         return out

#     def recommend(self, prompt: str, k: int = 10) -> Dict:
#         feat_map = self.prompt_to_vector(prompt)
#         dims = [i for i, f in enumerate(self.features) if feat_map.get(f) is not None]
#         if not dims:
#             return {
#                 "prompt": prompt,
#                 "used_features": [],
#                 "vector": {},
#                 "results": [],
#                 "warning": "LLM did not provide any features; refine your prompt.",
#             }
#         vec = []
#         for f in self.features:
#             v = feat_map.get(f)
#             if v is None:
#                 vec.append(np.nan)
#             else:
#                 if f == "mode":
#                     v = np.clip(v, 0, 1)
#                     vec.append(v)
#                 else:
#                     fmin, fmax = self.feature_minmax[f]
#                     val = (v - fmin) / (fmax - fmin + 1e-9)
#                     vec.append(float(np.clip(val, 0.0, 1.0)))
#         vec = np.array(vec, dtype=float)
#         X = self.track_matrix[:, dims]
#         q = vec[dims]
#         sims = cosine_similarity([q], X)[0]
#         idx = np.argsort(-sims)[:k]
#         results = self.df.iloc[idx][RETURN_COLS].copy()
#         results["similarity"] = sims[idx]
#         return {
#             "prompt": prompt,
#             "used_features": [self.features[i] for i in dims],
#             "vector": {f: feat_map.get(f) for f in self.features},
#             "results": results.to_dict(orient="records"),
#         }

# ###############################################################################
# # Flask app
# ###############################################################################

# def create_app(csv_path: str) -> Flask:
#     rec = Recommender.from_csv(csv_path, REDUCED_FEATURES)
#     app = Flask(__name__)

#     @app.get("/health")
#     def health():
#         return {
#             "status": "ok",
#             "num_tracks": int(rec.df.shape[0]),
#             "features": rec.features,
#             "minmax": rec.feature_minmax,
#             "model": rec.model_name,
#         }

#     @app.get("/features")
#     def features():
#         return {"features": rec.features, "minmax": rec.feature_minmax}

#     @app.get("/recommend")
#     def recommend():
#         q = request.args.get("query", "").strip()
#         if not q:
#             return jsonify({"error": "query parameter is required"}), 400
#         k = int(request.args.get("k", 10))
#         out = rec.recommend(q, k=k)
#         return jsonify(out)

#     return app

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", required=True, help="Path to SpotifyFeatures.csv")
#     parser.add_argument("--host", default="127.0.0.1")
#     parser.add_argument("--port", type=int, default=5000)
#     args = parser.parse_args()
#     app = create_app(args.csv)
#     app.run(host=args.host, port=args.port, debug=True)

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Google Gemini API
import google.generativeai as genai


# Only 9 features used for vector similarity (no track_genre)
REDUCED_FEATURES_DEFAULT: List[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",   # scaled BPM
]

RETURN_COLS_DEFAULT = [
    "artists", "album_name", "track_name", "popularity", "danceability", 
    "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "track_genre"
]

# Predefined track genres - will be populated later
PREDEFINED_TRACK_GENRES: List[str] = [
    "acoustic",
    "afrobeat",
    "alt-rock",
    "alternative",
    "ambient",
    "anime",
    "black-metal",
    "bluegrass",
    "blues",
    "brazil",
    "breakbeat",
    "british",
    "cantopop",
    "chicago-house",
    "children",
    "chill",
    "classical",
    "club",
    "comedy",
    "country",
    "dance",
    "dancehall",
    "death-metal",
    "deep-house",
    "detroit-techno",
    "disco",
    "disney",
    "drum-and-bass",
    "dub",
    "dubstep",
    "edm",
    "electro",
    "electronic",
    "emo",
    "folk",
    "forro",
    "french",
    "funk",
    "garage",
    "german",
    "gospel",
    "goth",
    "grindcore",
    "groove",
    "grunge",
    "guitar",
    "happy",
    "hard-rock",
    "hardcore",
    "hardstyle",
    "heavy-metal",
    "hip-hop",
    "honky-tonk",
    "house",
    "idm",
    "indian",
    "indie",
    "indie-pop",
    "industrial",
    "iranian",
    "j-dance",
    "j-idol",
    "j-pop",
    "j-rock",
    "jazz",
    "k-pop",
    "kids",
    "latin",
    "latino",
    "malay",
    "mandopop",
    "metal",
    "metalcore",
    "minimal-techno",
    "mpb",
    "new-age",
    "opera",
    "pagode",
    "party",
    "piano",
    "pop",
    "pop-film",
    "power-pop",
    "progressive-house",
    "psych-rock",
    "punk",
    "punk-rock",
    "r-n-b",
    "reggae",
    "reggaeton",
    "rock",
    "rock-n-roll",
    "rockabilly",
    "romance",
    "sad",
    "salsa",
    "samba",
    "sertanejo",
    "show-tunes",
    "singer-songwriter",
    "ska",
    "sleep",
    "songwriter",
    "soul",
    "spanish",
    "study",
    "swedish",
    "synth-pop",
    "tango",
    "techno",
    "trance",
    "trip-hop",
    "turkish",
    "world-music"
]

LLM_SYSTEM_FEATURES = (
    "You are a precise music feature extractor with deep understanding of musical characteristics. "
    "Analyze the user's prompt semantically to extract 9 specific audio features. Each feature "
    "represents a measurable aspect of music that affects how a song feels and sounds. "
    "Consider the emotional context, energy level, musical style, and descriptive words in the prompt. "
    "Map these semantic meanings to numeric values within the specified ranges. "
    "Only return null if the prompt provides absolutely no relevant information for a feature."
)

LLM_SYSTEM_GENRE = (
    "You are an expert music genre classifier with deep knowledge of musical styles and characteristics. "
    "Analyze the user's prompt semantically to understand the musical style, mood, cultural context, "
    "and descriptive elements they're looking for. Consider keywords like instruments, cultural references, "
    "energy levels, time periods, and emotional descriptions. Match these semantic elements to the most "
    "appropriate genre from the provided list. If multiple genres could apply, choose the most specific one. "
    "Return the exact genre name from the list, or null if no reasonable match can be determined."
)

def _build_user_prompt(
    prompt: str,
    features: List[str],
    minmax: Dict[str, Tuple[float, float]],
) -> str:
    feature_descriptions = {
        "danceability": "How suitable the track is for dancing (0.0-0.98). Higher values = more danceable, rhythmic, steady beat.",
        "energy": "Perceived energy and intensity (0.0-1.0). Higher values = louder, faster, noisier, more energetic.",
        "loudness": "Overall loudness in dB (-49.5 to 4.53). Higher values = louder tracks.",
        "speechiness": "Presence of spoken words (0.0-0.96). Higher values = more speech-like, rap, talk shows.",
        "acousticness": "Acoustic vs electric instruments (0.0-1.0). Higher values = more acoustic, organic sound.",
        "instrumentalness": "Lack of vocals (0.0-1.0). Higher values = fewer vocals, more instrumental.",
        "liveness": "Live performance detection (0.0-1.0). Higher values = more likely recorded live.",
        "valence": "Musical positivity/mood (0.0-0.99). Higher values = happier, euphoric, positive.",
        "tempo": "Beats per minute (0-243 BPM). Actual tempo of the track."
    }
    
    spec = {
        "task": "Analyze the prompt semantically and map it to these 9 musical features",
        "semantic_guidelines": [
            "Consider emotional words (happy→high valence, sad→low valence)",
            "Energy words (energetic→high energy, chill→low energy)",
            "Musical styles (acoustic→high acousticness, electronic→low acousticness)",
            "Vocal descriptions (instrumental→high instrumentalness, singing→low instrumentalness)",
            "Tempo indicators (fast→high tempo, slow→low tempo)",
            "Dance context (party→high danceability, study→low danceability)",
            "Live context (concert→high liveness, studio→low liveness)"
        ],
        "features": {
            f: {
                "description": feature_descriptions[f],
                "range": list(minmax.get(f, (0.0, 1.0))),
                "type": "number"
            } for f in features
        },
        "instructions": [
            "Analyze the semantic meaning of the prompt",
            "Map emotional and descriptive elements to appropriate feature values",
            "Use the full range of each feature based on semantic intensity",
            "Return null only if the prompt gives no relevant information for that feature",
            "Be consistent: similar prompts should yield similar feature vectors"
        ]
    }
    return json.dumps({
        "user_prompt": prompt,
        "feature_specification": spec,
        "required_output": "JSON object with exact feature names as keys",
        "output_schema": {f: None for f in features}
    }, ensure_ascii=False, indent=2)

def _build_genre_prompt(prompt: str, genres: List[str]) -> str:
    return json.dumps({
        "task": "Analyze the user prompt semantically to identify the most appropriate music genre",
        "user_prompt": prompt,
        "semantic_analysis_guidelines": [
            "Look for explicit genre mentions (rock, pop, jazz, etc.)",
            "Identify musical instruments (guitar→rock, piano→classical, synthesizer→electronic)",
            "Consider cultural/language references (spanish→spanish, korean→k-pop)",
            "Analyze energy/mood descriptors (energetic→edm/rock, chill→ambient/study)",
            "Look for time period indicators (80s→synth-pop, vintage→blues/jazz)",
            "Consider activity context (workout→edm/hip-hop, study→ambient/chill)",
            "Identify vocal styles (rap→hip-hop, operatic→opera)"
        ],
        "available_genres": genres,
        "instructions": [
            "First understand what the user is describing musically",
            "Match semantic elements to genre characteristics from the available_genres list",
            "You MUST choose from the exact genres provided in available_genres list",
            "Return the exact genre name as it appears in the list",
            "Return null only if absolutely no reasonable match exists"
        ],
        "output_format": "Exact genre name as string from available_genres list, or null"
    }, ensure_ascii=False, indent=2)


@dataclass
class AIDJRecommender:
    """
    Pure LLM → feature-vector → nearest-tracks recommender.
    - No anchors, no keyword tables, no defaults.
    - Only uses the subset of features the LLM provides (others remain missing).
    """
    df: pd.DataFrame
    features: List[str]
    scaler: MinMaxScaler
    track_matrix: np.ndarray
    feature_minmax: Dict[str, Tuple[float, float]]
    return_cols: List[str]
    model_name: str = "gemini-1.5-flash"

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        features: Optional[List[str]] = None,
        return_cols: Optional[List[str]] = None,
        model_name: str = "gemini-1.5-flash",
    ) -> "AIDJRecommender":
        features = features or REDUCED_FEATURES_DEFAULT
        return_cols = return_cols or RETURN_COLS_DEFAULT

        df = pd.read_csv(csv_path)
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        # Fit per-feature min-max scaler (→ [0,1]) and cache min/max for inverse mapping
        scaler = MinMaxScaler()
        X = df[features].astype(float).values
        X_scaled = scaler.fit_transform(X)
        feature_minmax = {f: (float(df[f].min()), float(df[f].max())) for f in features}

        # Configure Gemini API - requires GOOGLE_API_KEY in env
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return cls(
            df=df,
            features=features,
            scaler=scaler,
            track_matrix=X_scaled,
            feature_minmax=feature_minmax,
            return_cols=return_cols,
            model_name=model_name,
        )

    # ---- Detect genre from prompt ----
    def prompt_to_genre(self, prompt: str) -> Optional[str]:
        """Extract genre from prompt using Gemini API"""
        if not PREDEFINED_TRACK_GENRES:
            return None  # No genres available
            
        user_payload = _build_genre_prompt(prompt, PREDEFINED_TRACK_GENRES)
        full_prompt = f"{LLM_SYSTEM_GENRE}\n\n{user_payload}"
        
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0
                )
            )
            genre = response.text.strip().strip('"').strip("'")
            
            # Validate against predefined genres
            if genre and genre in PREDEFINED_TRACK_GENRES:
                return genre
            return None
        except Exception as e:
            print(f"Gemini API error for genre detection: {e}")
            return None

    # ---- Prompt → raw feature map (floats or None). No defaults. ----
    def prompt_to_feature_map(self, prompt: str) -> Dict[str, Optional[float]]:
        user_payload = _build_user_prompt(prompt, self.features, self.feature_minmax)
        
        # Create the full prompt for Gemini
        full_prompt = f"{LLM_SYSTEM_FEATURES}\n\n{user_payload}"
        
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            raw = response.text
            
            try:
                data = json.loads(raw)
            except Exception:
                # Fail-safe: return no features rather than guessing
                return {f: None for f in self.features}

            out: Dict[str, Optional[float]] = {}
            for f in self.features:
                v = data.get(f, None)
                if v is None:
                    out[f] = None
                    continue
                    
                try:
                    val = float(v)
                    # Validate and clamp values to proper ranges
                    out[f] = self._validate_and_clamp_feature(f, val)
                except Exception:
                    out[f] = None
            return out
        except Exception as e:
            print(f"Gemini API error: {e}")
            return {f: None for f in self.features}
    
    def _validate_and_clamp_feature(self, feature_name: str, value: float) -> float:
        """Validate and clamp feature values to their proper ranges"""
        feature_ranges = {
            "danceability": (0.0, 0.98),
            "energy": (0.0, 1.0),
            "loudness": (-49.5, 4.53),
            "speechiness": (0.0, 0.96),
            "acousticness": (0.0, 1.0),
            "instrumentalness": (0.0, 1.0),
            "liveness": (0.0, 1.0),
            "valence": (0.0, 0.99),
            "tempo": (0.0, 243.0)
        }
        
        min_val, max_val = feature_ranges.get(feature_name, (0.0, 1.0))
        return float(np.clip(value, min_val, max_val))

    # ---- Scale a feature map to [0,1] using dataset min/max (no rounding). ----
    def _scale_feature_map(self, feat_map: Dict[str, Optional[float]]) -> np.ndarray:
        vec = []
        for f in self.features:
            v = feat_map.get(f)
            if v is None:
                vec.append(np.nan)
            else:
                # All features are normalized to [0,1] range
                fmin, fmax = self.feature_minmax[f]
                norm = (v - fmin) / (fmax - fmin + 1e-9)
                vec.append(float(np.clip(norm, 0.0, 1.0)))
        return np.array(vec, dtype=float)

    # ---- Enhanced recommendation with better semantic analysis ----
    def recommend(self, prompt: str, k: int = 10) -> Dict:
        # First, detect genre from prompt
        detected_genre = self.prompt_to_genre(prompt)
        
        # Get feature vector from prompt
        feat_map = self.prompt_to_feature_map(prompt)
        vec = self._scale_feature_map(feat_map)

        # Use only provided dimensions
        dims = np.where(~np.isnan(vec))[0]
        if dims.size == 0:
            return {
                "prompt": prompt,
                "detected_genre": detected_genre,
                "used_features": [],
                "vector": {f: feat_map.get(f) for f in self.features},
                "results": [],
                "warning": "No features provided by LLM; refine your prompt.",
            }

        # Filter dataset by genre if detected
        if detected_genre and 'track_genre' in self.df.columns:
            genre_mask = self.df['track_genre'] == detected_genre
            genre_indices = np.where(genre_mask)[0]
            
            if len(genre_indices) > 0:
                # Use only songs from the detected genre
                q = vec[dims]
                X = self.track_matrix[genre_indices][:, dims]
                sims = cosine_similarity([q], X)[0]
                top_relative_idx = np.argsort(-sims)[:k]
                top_idx = genre_indices[top_relative_idx]
                
                out = self.df.iloc[top_idx][self.return_cols].copy()
                out["similarity"] = sims[top_relative_idx]
                
                return {
                    "prompt": prompt,
                    "detected_genre": detected_genre,
                    "used_features": [self.features[i] for i in dims],
                    "vector": {f: feat_map.get(f) for f in self.features},
                    "results": out.to_dict(orient="records"),
                    "genre_filtered": True,
                    "total_genre_songs": len(genre_indices)
                }
            else:
                warning = f"No songs found for detected genre '{detected_genre}'"
        else:
            warning = "No genre detected" if not detected_genre else "Genre column not found"

        # Fallback: search across all songs if no genre filter applied
        q = vec[dims]
        X = self.track_matrix[:, dims]
        sims = cosine_similarity([q], X)[0]
        top_idx = np.argsort(-sims)[:k]
        out = self.df.iloc[top_idx][self.return_cols].copy()
        out["similarity"] = sims[top_idx]

        return {
            "prompt": prompt,
            "detected_genre": detected_genre,
            "used_features": [self.features[i] for i in dims],
            "vector": {f: feat_map.get(f) for f in self.features},
            "results": out.to_dict(orient="records"),
            "genre_filtered": False,
            "warning": warning if 'warning' in locals() else None
        }

    # ---- Optional helpers ----
    def get_features(self) -> List[str]:
        return list(self.features)

    def get_minmax(self) -> Dict[str, Tuple[float, float]]:
        return dict(self.feature_minmax)

