# @app.get("/health")
# def health():
#     return {"ok": True}

# @app.get("/")
# def home():
#     return render_template("intro.html")

import os, json, joblib
from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# You'll need to get a free YouTube Data API key from Google Cloud Console
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts/v1"))
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "numeric_features.json"
TRACKS_PATH = ARTIFACTS_DIR / "tracks_for_rec.csv"

# Load artifacts at process start
rf_model = joblib.load(MODEL_PATH)
numeric_features = json.loads(FEATURES_PATH.read_text())
tracks_df = pd.read_csv(TRACKS_PATH)

# Following 2 functions are for ML
# Simple prompt→mood mapper (I will replace this with LLM later)
def mood_from_prompt(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["heartbroken","broken","cry","tears","depressed","sad"]):
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
    return "happy"  #!!!

def recommend_by_mood(mood: str, top_k: int = 20):
    # Filter by mood
    sub = tracks_df.loc[tracks_df["mood"] == mood]
    if sub.empty:
        sub = tracks_df.loc[tracks_df["mood"] == "happy"]

    # Sort and de-duplicate exact same song/artist pairs, then cap
    sub = (
        sub.sort_values("track_popularity", ascending=False)
           .drop_duplicates(subset=["track_name", "track_artist"], keep="first")
           .head(top_k)
    )

    return sub[["track_name","track_artist","mood","track_popularity"]].to_dict(orient="records")


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



if __name__ == '__main__':
    app.run(debug=True)
