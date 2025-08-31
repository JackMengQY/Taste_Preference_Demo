
import os, json, sqlite3, time, math, random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

DB_PATH = Path(os.environ.get("DB_PATH", "/tmp/taste.db"))
ATTRS = ["sweet","salty","spicy","sour","bitter","umami","creamy","crispy","chewy","greasy","fresh_clean","protein_forward"]
D = len(ATTRS)

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS images(id INTEGER PRIMARY KEY, url TEXT, name TEXT, attrs TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, w TEXT, b REAL, n INTEGER DEFAULT 0, updated_at REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS responses(id INTEGER PRIMARY KEY, user_id TEXT, ts REAL, left_image_id INTEGER, right_image_id INTEGER, choice TEXT)")
    conn.commit()
    conn.close()

def seed_images_if_empty():
    conn = db()
    cur = conn.cursor()
    n = cur.execute("SELECT COUNT(*) as n FROM images").fetchone()["n"]
    if n == 0:
        def u(name): return f"/img/{name}"
        data = [
            (u("bigmac.png"), "Big Mac", [0.15,0.7,0.05,0.05,0.05,0.7,0.6,0.3,0.2,0.85,0.1,0.4]),
            (u("neapolitan_pizza.png"), "Neapolitan Pizza", [0.15,0.55,0.1,0.2,0.05,0.7,0.4,0.4,0.3,0.5,0.3,0.4]),
            (u("spicy_hotpot.png"), "Spicy Hotpot", [0.1,0.55,0.9,0.1,0.1,0.8,0.3,0.2,0.2,0.6,0.2,0.5]),
            (u("fish_and_chips.png"), "Fish & Chips", [0.05,0.75,0.05,0.05,0.05,0.5,0.2,0.9,0.2,0.8,0.1,0.3]),
            (u("chicken_caesar_salad.png"), "Chicken Caesar Salad", [0.05,0.35,0.0,0.05,0.05,0.4,0.3,0.2,0.2,0.1,0.9,0.7]),
            (u("pho.png"), "Pho with Herbs", [0.05,0.35,0.15,0.1,0.05,0.7,0.1,0.1,0.1,0.1,0.9,0.6]),
            (u("matcha_parfait.png"), "Matcha Parfait", [0.8,0.05,0.0,0.1,0.35,0.2,0.9,0.2,0.2,0.2,0.7,0.2]),
            (u("birria_tacos.png"), "Birria Tacos", [0.1,0.6,0.7,0.1,0.05,0.8,0.3,0.5,0.3,0.6,0.2,0.6]),
            (u("mapo_tofu.png"), "Mapo Tofu", [0.05,0.55,0.85,0.1,0.05,0.8,0.2,0.1,0.1,0.4,0.2,0.4]),
            (u("yogurt_granola.png"), "Yogurt & Granola", [0.55,0.05,0.0,0.1,0.05,0.2,0.8,0.2,0.2,0.1,0.8,0.3]),
            (u("sashimi.png"), "Sashimi Platter", [0.0,0.2,0.0,0.05,0.05,0.6,0.2,0.1,0.3,0.05,0.95,0.9]),
            (u("buffalo_wings.png"), "Buffalo Wings", [0.05,0.6,0.75,0.05,0.05,0.7,0.2,0.6,0.2,0.7,0.15,0.5])
        ]
        cur.executemany("INSERT INTO images(url,name,attrs) VALUES(?,?,?)", data)
        conn.commit()
    conn.close()

def load_images():
    conn = db()
    rows = conn.execute("SELECT id,url,name,attrs FROM images").fetchall()
    conn.close()
    return [{"id": r["id"], "url": r["url"], "name": r["name"], "x": np.array(json.loads(r["attrs"]), float)} for r in rows]

def load_responses(uid):
    conn = db()
    rows = conn.execute("SELECT left_image_id,right_image_id,choice FROM responses WHERE user_id=?", (uid,)).fetchall()
    conn.close()
    return [{"left": r["left_image_id"], "right": r["right_image_id"], "choice": r["choice"]} for r in rows]

def sigmoid(z): return 1.0/(1.0+math.exp(-z))

def train_user_w(uid, images, lr=0.5, l2=0.01, epochs=250):
    idx = {i["id"]: i for i in images}
    R = load_responses(uid)
    if not R: return np.zeros(D), 0.0
    X, y = [], []
    for r in R:
        l, rgt = idx.get(r["left"]), idx.get(r["right"])
        if l is None or rgt is None: continue
        X.append(l["x"] - rgt["x"])
        y.append(1 if r["choice"] == "left" else 0)
    if not X: return np.zeros(D), 0.0
    X, y = np.vstack(X), np.array(y, float)
    w, b = np.zeros(D), 0.0
    for _ in range(epochs):
        p = sigmoid(X.dot(w) + b)
        w -= lr * (X.T.dot(p - y)/len(y) + l2*w)
        b -= lr * np.mean(p - y)
    return w, b

class NextPairResponse(BaseModel):
    left_id: int; right_id: int; left_url: str; right_url: str; left_name: str; right_name: str

class ClickPayload(BaseModel):
    user_id: str; left_id: int; right_id: int; choice: str

init_db()
seed_images_if_empty()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"ok": True, "attrs": len(ATTRS)}

@app.get("/images")
def images_list():
    ims = load_images()
    return {"count": len(ims), "images": [{"id": i["id"], "name": i["name"], "url": i["url"]} for i in ims]}

@app.get("/next", response_model=NextPairResponse)
def next_pair(user_id: str):
    ims = load_images()
    if len(ims) < 2: raise HTTPException(400, "Not enough images.")
    w, b = train_user_w(user_id, ims)
    best, best_unc = None, 999
    for _ in range(60):
        L, R = random.sample(ims, 2)
        p = sigmoid(float((L["x"] - R["x"]).dot(w) + b))
        unc = abs(p - 0.5)
        if unc < best_unc:
            best_unc, best = unc, (L, R)
    L, R = best
    return NextPairResponse(left_id=L["id"], right_id=R["id"], left_url=L["url"], right_url=R["url"], left_name=L["name"], right_name=R["name"])

@app.post("/click")
def click(payload: ClickPayload):
    if payload.choice not in ("left","right"):
        raise HTTPException(400, "Invalid choice")
    conn = db()
    conn.execute("INSERT OR IGNORE INTO users(id, w, b, n, updated_at) VALUES(?,?,?,?,?)", (payload.user_id, json.dumps([0.0]*D), 0.0, 0, time.time()))
    conn.execute("INSERT INTO responses(user_id, ts, left_image_id, right_image_id, choice) VALUES(?,?,?,?,?)",
                 (payload.user_id, time.time(), payload.left_id, payload.right_id, payload.choice))
    conn.execute("UPDATE users SET n=COALESCE(n,0)+1, updated_at=? WHERE id=?", (time.time(), payload.user_id))
    conn.commit()
    conn.close()
    return {"ok": True}

# Serve static files after routes
BASE_DIR = Path(__file__).parent
app.mount("/img", StaticFiles(directory=BASE_DIR / "static" / "img"), name="img")
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
