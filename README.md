# AI-DJ 🎧 🎧 🎧 🎧 🎧 

Lightweight Flask backend for AI-DJ, fully Dockerized with dev hot-reload and production-ready Gunicorn setup.

# 1. Clone the Repo 📥 
```bash
git clone https://github.com/YagizBasaran/ai-dj.git
cd ai-dj
```

# 2. How to start?

## 2.1. Development (Instant Reload)
```bash 
docker compose up --build dev
```
- Visit: http://localhost:5000
- Edit files locally → refresh → changes show immediately

## 2.2. Production / Deployment
```bash 
docker compose up --build web
```
- Visit: http://localhost:8000
- For code changes: ```docker compose up --build web``` again because no auto-refresh here (Gunicorn)

## 2.3. Extra Files
- Add .env file and put your YOUTUBE_API_KEY="..." for generating the key follow these steps:

How to Get Free YouTube API Key:

- Go to Google Cloud Console
- Create a new project or select existing
- Enable "YouTube Data API v3"
- Create credentials (API key)
