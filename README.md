# Taste Profile Demo (Render-ready)

Single service: FastAPI serves the API **and** the frontend (static index.html + local images).

## Local run
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
uvicorn app.app:app --reload --port 8080
```
Open http://127.0.0.1:8080

## Deploy to Render
1. Create a new GitHub repository and push this folder.
2. In Render, click **New +** → **Web Service** → **Connect repository**.
3. Confirm these settings (already in render.yaml):
   - Build Command: `pip install -r app/requirements.txt`
   - Start Command: `uvicorn app.app:app --host 0.0.0.0 --port $PORT`
4. Deploy. Your public URL will look like `https://taste-profile-demo.onrender.com`.

No extra static hosting needed: images are served at `/img/*` and the app serves `/` (index.html).
