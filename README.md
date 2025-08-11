# AI-DJ 🎧 🎧 🎧 🎧 🎧 

Lightweight Flask backend for AI-DJ, fully Dockerized with dev hot-reload and production-ready Gunicorn setup.

# 1. Clone the Repo 📥 
```bash
git clone <your-repo-url>
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
