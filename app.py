"""
AI DJ - Flask Application
Music recommendation system using ML-based mood detection and YouTube integration
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import requests

from ml_core import (
    load_artifacts,
    mood_from_prompt,
    recommend_by_mood_with_preference,
    has_model,
    tracks_df,
    numeric_features,
    _rf_bundle
)

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = Flask(__name__)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ============================================================================
# INITIALIZATION & DEBUG
# ============================================================================

def debug_file_system():
    """Debug file system to verify artifact paths"""
    print(f"\n=== FILE SYSTEM DEBUG ===")
    print(f"Current working directory: {os.getcwd()}")

    shared_path = Path("artifacts/shared")
    model_path = Path("artifacts/models/v2/model.pkl")

    print(f"Shared path exists: {shared_path.exists()}")
    print(f"Model path exists: {model_path.exists()}")

    if shared_path.exists():
        print(f"Contents of shared/: {os.listdir(shared_path)}")

    models_path = Path("artifacts/models")
    if models_path.exists():
        print(f"Contents of models/: {os.listdir(models_path)}")
        v2_path = models_path / "v2"
        if v2_path.exists():
            print(f"Contents of models/v2/: {os.listdir(v2_path)}")

    print(f"=== END FILE SYSTEM DEBUG ===\n")

def debug_loaded_artifacts():
    """Debug loaded ML artifacts"""
    print(f"\n=== LOADING DEBUG ===")
    print(f"Model loaded: {has_model()}")
    print(f"Tracks loaded: {len(tracks_df()) if tracks_df() is not None else 'None'}")
    print(f"Features: {numeric_features()}")

    if has_model() and _rf_bundle is not None:
        model = _rf_bundle["model"]
        print(f"Model classes: {list(model.classes_)}")
    print(f"=== END LOADING DEBUG ===\n")

debug_file_system()
load_artifacts()
debug_loaded_artifacts()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def search_youtube_music(query, max_results=5):
    """
    Search for music videos on YouTube using the YouTube Data API

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of dictionaries containing video information
    """
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            'part': 'snippet',
            'q': query + " music",
            'type': 'video',
            'maxResults': max_results,
            'key': YOUTUBE_API_KEY,
            'videoCategoryId': '10',  # Music category
            'order': 'relevance'
        }

        response = requests.get(url, params=params)
        data = response.json()

        results = []
        if 'items' in data:
            for item in data['items']:
                results.append({
                    'id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'][:100] + "...",
                    'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                    'channel': item['snippet']['channelTitle']
                })

        return results
    except Exception as e:
        print(f"YouTube API Error: {e}")
        return []

# ============================================================================
# ROUTES - Pages
# ============================================================================

@app.route('/')
def intro_page():
    """Landing page with app introduction"""
    return render_template('intro.html')

@app.route('/home')
def home():
    """Main AI DJ interface"""
    return render_template('home.html')

# ============================================================================
# ROUTES - API Endpoints
# ============================================================================

@app.route('/process_request', methods=['POST'])
def process_request():
    """
    Main API endpoint for processing user music requests
    Returns YouTube videos based on mood detection and ML recommendations
    """
    try:
        user_input = request.json.get('input', '')
        preference = request.json.get('preference', 'balanced')
        user_preferences = request.json.get('user_preferences', {})

        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        detected_mood = mood_from_prompt(user_input)
        recs = recommend_by_mood_with_preference(
            detected_mood,
            preference,
            top_k=10,
            user_query=user_input,
            user_preferences=user_preferences
        )

        direct_matches = [r for r in recs if r.get('is_direct_match', False)]
        mood_matches = [r for r in recs if not r.get('is_direct_match', False)]

        all_results = []

        for rec in direct_matches[:3]:
            search_query = f"{rec['track_name']} {rec['track_artist']}"
            youtube_results = search_youtube_music(search_query, 1)
            if youtube_results:
                result = youtube_results[0]
                result['source'] = 'direct_match'
                result['track_info'] = rec
                all_results.append(result)

        for rec in mood_matches[:7]:
            search_query = f"{rec['track_name']} {rec['track_artist']}"
            youtube_results = search_youtube_music(search_query, 1)
            if youtube_results:
                result = youtube_results[0]
                result['source'] = 'mood_match'
                result['track_info'] = rec
                all_results.append(result)

        return jsonify({
            'mood': detected_mood,
            'preference': preference,
            'results': all_results,
            'direct_matches_count': len(direct_matches),
            'mood_matches_count': len(mood_matches),
            'top_match': all_results[0] if all_results else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/ml-test", methods=["GET", "POST"])
def ml_test():
    """
    ML testing interface for debugging recommendations
    Shows raw ML output without YouTube integration
    """
    prompt = ""
    detected_mood = None
    recs = []
    preference = "balanced"

    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        preference = request.form.get("preference", "balanced")
        detected_mood = mood_from_prompt(prompt)
        recs = recommend_by_mood_with_preference(
            detected_mood,
            preference,
            top_k=10,
            user_query=prompt
        )

    return render_template(
        "ml_test.html",
        prompt=prompt,
        detected_mood=detected_mood,
        recs=recs,
        preference=preference
    )

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True)
