import os, json
from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
from pathlib import Path
import numpy as np

from ml_core import (
    load_artifacts,
    mood_from_prompt,
    recommend_by_mood,
    recommend_by_mood_with_preference,
    has_model,
    tracks_df,
    numeric_features
)

app = Flask(__name__)

# Debug file system BEFORE loading
print(f"=== FILE SYSTEM DEBUG ===")
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

print(f"=== END FILE SYSTEM DEBUG ===")

# NOW load the artifacts
load_artifacts()  # This was missing!

# Debug loading results
print(f"=== LOADING DEBUG ===")
print(f"Model loaded: {has_model()}")
print(f"Tracks loaded: {len(tracks_df()) if tracks_df() is not None else 'None'}")
print(f"Features: {numeric_features()}")

if has_model():
    from ml_core import _rf_bundle
    model = _rf_bundle["model"]
    print(f"Model classes: {list(model.classes_)}")
print(f"=== END LOADING DEBUG ===")

# You'll need to get a free YouTube Data API key from Google Cloud Console
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def search_youtube_music(query, max_results=5):
    """
    Search for music videos on YouTube
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

# Routes for explanatory pages
@app.route('/')
def intro_page():
    return render_template('intro.html')

@app.route('/home')
def home():
    return render_template('home.html')

# NEW: JSON API endpoint for ML recommendations
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """
    API endpoint that returns ML recommendations as JSON
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        preference = data.get('preference', 'balanced')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Get mood from prompt
        detected_mood = mood_from_prompt(prompt)
        
        # Get recommendations using ML
        recommendations = recommend_by_mood_with_preference(
            mood=detected_mood, 
            preference=preference, 
            top_k=10,  # Total recommendations
            user_query=prompt  # For direct search
        )
        
        # Separate direct matches from mood-based
        direct_matches = [r for r in recommendations if r.get('is_direct_match', False)]
        mood_based = [r for r in recommendations if not r.get('is_direct_match', False)]
        
        # Function to add YouTube thumbnail to a track
        def add_youtube_thumbnail(track):
            try:
                search_query = f"{track['track_name']} {track['track_artist']}"
                youtube_results = search_youtube_music(search_query, 1)
                if youtube_results and len(youtube_results) > 0:
                    track['thumbnail'] = youtube_results[0]['thumbnail']
                    track['youtube_id'] = youtube_results[0]['id']
                else:
                    track['thumbnail'] = None
                    track['youtube_id'] = None
            except:
                track['thumbnail'] = None
                track['youtube_id'] = None
            return track
        
        # Get YouTube video for the TOP MOOD-BASED SONG
        top_match_video = None
        if len(mood_based) > 0:
            top_song = mood_based[0]
            search_query = f"{top_song['track_name']} {top_song['track_artist']}"
            youtube_results = search_youtube_music(search_query, 1)
            if youtube_results:
                top_match_video = youtube_results[0]
                top_match_video['track_info'] = {
                    'name': top_song['track_name'],
                    'artist': top_song['track_artist']
                }
        
        # Add thumbnails to first 5 mood-based songs
        mood_based_with_thumbs = []
        for i, track in enumerate(mood_based[:5]):
            mood_based_with_thumbs.append(add_youtube_thumbnail(track))
        
        # Add thumbnails to direct matches (max 2)
        direct_with_thumbs = []
        for track in direct_matches[:2]:
            direct_with_thumbs.append(add_youtube_thumbnail(track))
        
        # Format response
        response = {
            'mood': detected_mood,
            'moodBased': mood_based_with_thumbs,  # Now limited to 5 with thumbnails
            'directSearch': direct_with_thumbs,  # Max 2 with thumbnails
            'topMatch': top_match_video
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in api_recommend: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_request', methods=['POST'])
def process_request():
    # This is the old endpoint, kept for compatibility
    try:
        user_input = request.json.get('input', '')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Process with mock ML (replace with your actual ML later)
        ml_result = process_user_input(user_input)
        
        # Search for music based on ML result
        all_results = []
        for search_term in ml_result['search_terms'][:2]:  # Limit API calls
            results = search_youtube_music(search_term, 3)
            all_results.extend(results)
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
            if len(unique_results) >= 8:
                break
        
        return jsonify({
            'mood': ml_result['mood'],
            'confidence': ml_result['confidence'],
            'results': unique_results,
            'top_match': unique_results[0] if unique_results else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/ml-test", methods=["GET", "POST"])
def ml_test():
    prompt = ""
    detected_mood = None
    recs = []
    preference = "balanced"  # default
    
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        preference = request.form.get("preference", "balanced")
        detected_mood = mood_from_prompt(prompt)
        
        # Use the new function with preference
        recs = recommend_by_mood_with_preference(detected_mood, preference, top_k=10, user_query=prompt)

    return render_template(
        "ml_test.html",
        prompt=prompt,
        detected_mood=detected_mood,
        recs=recs,
        preference=preference
    )

# Helper function for the old endpoint
def process_user_input(user_input):
    """
    This is the old mock function, kept for compatibility
    """
    user_input = user_input.lower()
    
    # Simple keyword-based mock processing
    if any(word in user_input for word in ['sad', 'depressed', 'broken', 'heartbreak']):
        mood = "sad"
        search_terms = ["sad songs", "emotional ballads", "heartbreak music"]
    elif any(word in user_input for word in ['happy', 'energetic', 'pump', 'workout', 'competitive']):
        mood = "energetic"
        search_terms = ["upbeat music", "workout songs", "motivational music"]
    elif any(word in user_input for word in ['chill', 'relax', 'calm', 'study']):
        mood = "chill"
        search_terms = ["chill music", "lo-fi beats", "relaxing songs"]
    else:
        mood = "general"
        search_terms = [user_input, "popular music"]
    
    return {
        'mood': mood,
        'search_terms': search_terms,
        'confidence': 0.85
    }

if __name__ == '__main__':
    app.run(debug=True)