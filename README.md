# AI-DJ 🎧 🎧 🎧 🎧 🎧 

Lightweight Flask backend for AI-DJ, fully Dockerized with dev hot-reload and production-ready Gunicorn setup.

# 1. Clone the Repo 📥 
```bash
git clone https://github.com/YagizBasaran/ai-dj.git
cd ai-dj
```

# 2. How to start?

## 2.1. Development (Instant Reload)

- Download Docker to your computer and open it. Then from your terminal (Git Bash prefered), run the command below in the project folder.

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
- Add .env file and put your Gemini key as GOOGLE_API_KEY= "..." and "YOUTUBE_API_KEY="..." for generating the Youtube Api key follow these steps:

How to Get Free YouTube API Key:
- Go to Google Cloud Console
- Create a new project or select existing
- Enable "YouTube Data API v3"
- Create credentials (API key)

How to Get Free Google Gemini API Key:
- https://aistudio.google.com/app/api-keys

# 3. ML Model

## 3.1. Random Forest + No LLM (16.08.2025)
1. Labels are handmade, more precise and more detailed labels MUST be made.
2. After labels, a LLM MUST interpret the prompt and communicate with ML model.
- To test this version go to `http://localhost:5000/ml-test` after running dev version.

### 3.1.1 Random Forest v2 + No LLM (10.09.2025) at understaning the user prompt
1. Serhat's Gemini algorithm could be used.
2. Might require better labeling at the model (Google Colab .ipynb file)
3. Creates scores (%95 ML model + %5 popularity)
- To test this version go to `http://localhost:5000/ml-test` after running dev version.

## 3.2 Random Forest + Gemini 2.5 Flash LLM (30.09.2025)
1. Gemini now understands the prompt and decides a mood.
2. RM model still can be improved.
3. Direct search added (artist, song or album name)
4. Previous updates also added multi search (Discovery vs Mainstream)
- To test this version go to `http://localhost:5000/ml-test` after running dev version.

