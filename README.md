# AI-DJ 🎧 🎧 🎧 🎧 🎧 
AI-powered music recommendation system that understands your emotions and suggests music accordingly using machine learning and natural language processing.
https://ai-dj-y2v1.onrender.com
## Table of Contents
1. [Features](#1-features-)
2. [Technical Architecture](#2-technical-architecture-)
3. [Prerequisites](#3-prerequisites-)
4. [Quick Start](#4-quick-start-)
5. [Dev Notes](#5-devs-notes-)

# 1) Features 🤖

### Core Features (No Authentication Required)

**1. Emotion-Based Music Selection**
- Natural language mood detection using Google Gemini AI
- Converts user prompts into mood categories: sad, angry, happy, hype, romantic, chill, focus, live etc.
- Machine learning-powered song recommendations using **Random Forest** classifier
- Trained on 82,000+ songs with audio features (energy, valence, tempo, danceability, etc.)

**2. Direct Search Capabilities**
- Intelligent search for specific songs, artists, or albums.
- Word tokenization and fuzzy matching for better natural language queries.
- *Dual search approach*: ML-based recommendations + direct YouTube matches.

**3. Discovery Modes**
- **Discovery Mode**: 70% lesser-known songs, 30% popular (find hidden gems)
- **Mainstream Mode**: 70% popular songs, 30% lesser-known (safe choices)
- **Balanced Mode**: Equal mix based on ML confidence scores

**4. YouTube Integration**
- Automatic YouTube video fetching for all recommendations
- Embedded player with autoplay
- Video thumbnails for all songs

### Premium Features (Requires Firebase setup but check it out at https://ai-dj-y2v1.onrender.com)

**5. User Authentication**
- Secure user registration and login system
- Password hashing with bcrypt
- Session-based authentication using Flask sessions

**6. Time-of-Day Pattern Learning**
- Tracks user listening habits across 5 time sections:
  - Morning Generation (6am-12pm)
  - In Job (12pm-5pm)
  - After Job (5pm-8pm)
  - Evening Generation (8pm-12am)
  - Night Generation (12am-6am)
- Learns mood preferences for each time section
- Provides personalized recommendations based on historical listening patterns
- Displays pattern-based song suggestions when you return at the same time

# 2) Technical Architecture 🔧

### Backend Stack
- **Framework**: Flask 3.0.3 (Python web framework)
- **ML Engine**: scikit-learn 1.4.2 with Random Forest Classifier
- **Database**: Firebase Firestore (optional, for user data)
- **LLM Service**: Google Gemini 2.5 Flash (mood detection)
- **Containerization**: Docker + Docker Compose

### ML Model Details
- **Algorithm**: Random Forest with 200 estimators
- **Features**: 13 audio features (energy, tempo, danceability, loudness, liveness, valence, speechiness, instrumentalness, mode, key, duration, acousticness, popularity)
- **Classes**: 8 mood categories
- **Preprocessing**: StandardScaler normalization
- **Training Split**: 70% train, 15% validation, 15% test

### Frontend Stack
- Pure HTML, CSS (no frameworks)
- Real-time chat-like interface
- YouTube embedded player

### Data Pipeline
1. User prompt → Gemini API → Mood classification
2. Mood → Random Forest model → Probability scores (nearest vector)
3. Scores (95%) + Popularity (5%) → Final ranking algorithm
4. Top songs → YouTube API → Video metadata + thumbnails
5. Results → Frontend rendering

# 3) Prerequisites ✔

### Required
- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **YouTube Data API v3 Key** ([Get it here](https://console.cloud.google.com/))
- **Google Gemini API Key** ([Get it here](https://aistudio.google.com/app/api-keys))

### Optional (for authentication features)

Create Firebase Project

   - Go to https://console.firebase.google.com/
   - Click "Add project"
   - Enter project name
   - Click "Create project"

Enable Firestore Database

   - In Firebase Console, click "Build" → "Firestore Database"
   - Click "Create database"
   - Choose "Start in test mode" (for development)
   - Select a location close to your users
   - Click "Enable"

Generate Service Account Key

   - Click gear icon → "Project settings"
   - Go to "Service accounts" tab
   - Click "Generate new private key"
   - Click "Generate key"
   - Save the downloaded JSON file as `firebase-key.json` in project root

Add this to .env file lastly, SECRET_KEY="change-this-to-a-random-secret-key-in-productionx"

# 4) Quick Start 🎶

### Minimal Setup (5 minutes)
```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai-dj.git
cd ai-dj

# 2. Create environment file
cp .env.example .env

# 3. Edit .env and add your API keys
nano .env  # or use any text editor

# Add these lines:
YOUTUBE_API_KEY=your_youtube_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here

# 4. Run with Docker
docker compose up --build dev

# 5. Open browser
# Navigate to http://localhost:5000
```

# 5) Devs Notes :)

## 1. Random Forest + No LLM (16.08.2025)
1. Labels are handmade, more precise and more detailed labels MUST be made.
2. After labels, a LLM MUST interpret the prompt and communicate with ML model.
- To test this version go to `http://localhost:5000/ml-test` after running dev version.

### 1.1 Random Forest v2 + No LLM (10.09.2025) at understaning the user prompt
1. Serhat's Gemini algorithm could be used.
2. Might require better labeling at the model (Google Colab .ipynb file)
3. Creates scores (%95 ML model + %5 popularity)
- To test this version go to `http://localhost:5000/ml-test` after running dev version.

## 2. Random Forest + Gemini 2.5 Flash LLM (30.09.2025)
1. Gemini now understands the prompt and decides a mood.
2. RM model still can be improved.
3. Direct search added (artist, song or album name)
4. Previous updates also added multi search (Discovery vs Mainstream)
- To test this version go to `http://localhost:5000/ml-test` after running dev version.

