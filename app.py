# @app.get("/health")
# def health():
#     return {"ok": True}

# @app.get("/")
# def home():
#     return render_template("intro.html")

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
)
load_artifacts(os.getenv("ARTIFACTS_DIR", "artifacts/v1"))

app = Flask(__name__)

# You'll need to get a free YouTube Data API key from Google Cloud Console
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


#Serhat:
# Mock ML processing function (replace with your actual ML later)
def process_user_input(user_input):
    """
    This is where ML algorithms will go later.
    For now, it returns mock results based on keywords.
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

@app.route('/process_request', methods=['POST'])
def process_request():
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
    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        detected_mood = mood_from_prompt(prompt)
        recs = recommend_by_mood(detected_mood, top_k=10)

    return render_template(
        "ml_test.html",
        prompt=prompt,
        detected_mood=detected_mood,
        recs=recs
    )

@app.route("/ml-test2", methods=["GET", "POST"])
def ml_test2():
    prompt = ""
    recommendations = []
    timing_ms = 0
    mixture = {}
    hard_constraints = {}
    
    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        if prompt:
            from ml_serhat import recommend_from_prompt
            result = recommend_from_prompt(prompt, topn=30)
            recommendations = result.get("results", [])
            timing_ms = result.get("timing_ms", 0)
            mixture = result.get("mixture", {})
            hard_constraints = result.get("hard", {})
    
    return render_template("ml_test2.html", 
                         prompt=prompt,
                         recommendations=recommendations,
                         timing_ms=timing_ms,
                         mixture=mixture,
                         hard_constraints=hard_constraints)

    

if __name__ == '__main__':
    app.run(debug=True)
