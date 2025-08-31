import os, sqlite3, json, time, math, random
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "taste.db"

ATTRS = ["sweet","salty","spicy","sour","bitter","umami","creamy","crispy","chewy","greasy","fresh_clean","protein_forward"]
DISPLAY = {
    "sweet": "Sweet",
    "salty": "Salty",
    "spicy": "Spicy",
    "sour": "Sour / Tangy",
    "bitter": "Bitter",
    "umami": "Umami / Savory",
    "creamy": "Creamy",
    "crispy": "Crispy",
    "chewy": "Chewy",
    "greasy": "Rich / Greasy",
    "fresh_clean": "Fresh & Clean",
    "protein_forward": "Protein‑Forward"
}
D = len(ATTRS)

def db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS images(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        name TEXT,
        attrs TEXT NOT NULL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id TEXT PRIMARY KEY,
        w TEXT,
        b REAL,
        n INTEGER DEFAULT 0,
        updated_at REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS responses(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        ts REAL NOT NULL,
        left_image_id INTEGER NOT NULL,
        right_image_id INTEGER NOT NULL,
        choice TEXT NOT NULL CHECK(choice IN ('left','right'))
    )""")
    conn.commit(); conn.close()

def seed_images_if_empty():
    conn = db()
    n = conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
    if n == 0:
        # Local URLs pointing to /img/*
        def u(name): return f"/img/{name}"
        seed = [
            (u("bigmac.png"), "Big Mac",[0.15,0.7,0.05,0.05,0.05,0.7,0.6,0.3,0.2,0.85,0.1,0.4]),
            (u("neapolitan_pizza.png"), "Neapolitan Pizza",[0.15,0.55,0.1,0.2,0.05,0.7,0.4,0.4,0.3,0.5,0.3,0.4]),
            (u("spicy_hotpot.png"), "Spicy Hotpot",[0.1,0.55,0.9,0.1,0.1,0.8,0.3,0.2,0.2,0.6,0.2,0.5]),
            (u("fish_and_chips.png"), "Fish & Chips",[0.05,0.75,0.05,0.05,0.05,0.5,0.2,0.9,0.2,0.8,0.1,0.3]),
            (u("chicken_caesar_salad.png"), "Chicken Caesar Salad",[0.05,0.35,0.0,0.05,0.05,0.4,0.3,0.2,0.2,0.1,0.9,0.7]),
            (u("pho.png"), "Pho with Herbs",[0.05,0.35,0.15,0.1,0.05,0.7,0.1,0.1,0.1,0.1,0.9,0.6]),
            (u("matcha_parfait.png"), "Matcha Parfait",[0.8,0.05,0.0,0.1,0.35,0.2,0.9,0.2,0.2,0.2,0.7,0.2]),
            (u("birria_tacos.png"), "Birria Tacos",[0.1,0.6,0.7,0.1,0.05,0.8,0.3,0.5,0.3,0.6,0.2,0.6]),
            (u("mapo_tofu.png"), "Mapo Tofu",[0.05,0.55,0.85,0.1,0.05,0.8,0.2,0.1,0.1,0.4,0.2,0.4]),
            (u("yogurt_granola.png"), "Yogurt & Granola Bowl",[0.55,0.05,0.0,0.1,0.05,0.2,0.8,0.2,0.2,0.1,0.8,0.3]),
            (u("sashimi.png"), "Sashimi Platter",[0.0,0.2,0.0,0.05,0.05,0.6,0.2,0.1,0.3,0.05,0.95,0.9]),
            (u("buffalo_wings.png"), "Buffalo Wings",[0.05,0.6,0.75,0.05,0.05,0.7,0.2,0.6,0.2,0.7,0.15,0.5])
        ]
        conn.executemany("INSERT INTO images(url,name,attrs) VALUES(?,?,?)",
                         [(url, name, json.dumps(v)) for url, name, v in seed])
        conn.commit()
    conn.close()

def load_images() -> List[Dict[str,Any]]:
    conn = db()
    rows = conn.execute("SELECT id,url,name,attrs FROM images").fetchall()
    conn.close()
    return [{"id": r["id"], "url": r["url"], "name": r["name"], "x": np.array(json.loads(r["attrs"]), dtype=float)} for r in rows]

def load_responses(user_id: str):
    conn = db()
    rows = conn.execute("SELECT left_image_id,right_image_id,choice FROM responses WHERE user_id=? ORDER BY id ASC",(user_id,)).fetchall()
    conn.close()
    return [{"left": r["left_image_id"], "right": r["right_image_id"], "choice": r["choice"]} for r in rows]

def sigmoid(z: float): return 1.0/(1.0+math.exp(-z))

def train_user_w(user_id: str, images: List[Dict[str,Any]], lr=0.5, l2=0.01, epochs=250):
    idx = {im["id"]: im for im in images}
    R = load_responses(user_id)
    if len(R) == 0: return np.zeros(len(ATTRS)), 0.0
    X, y = [], []
    for r in R:
        xl, xr = idx.get(r["left"]), idx.get(r["right"])
        if xl is None or xr is None: continue
        X.append(xl["x"] - xr["x"]); y.append(1 if r["choice"] == "left" else 0)
    if not X: return np.zeros(len(ATTRS)), 0.0
    X = np.vstack(X); y = np.array(y, float)
    w = np.zeros(len(ATTRS), float); b = 0.0
    for _ in range(epochs):
        z = X.dot(w) + b; p = 1/(1+np.exp(-z))
        grad_w = X.T.dot(p - y)/len(y) + l2*w
        grad_b = float(np.mean(p - y))
        w -= lr*grad_w; b -= lr*grad_b
    return w, b

def bucketize(attrs, w):
    absw = np.abs(w); mag = float(np.linalg.norm(w) + 1e-9)
    rel = absw / mag if mag > 0 else np.zeros_like(absw)
    hi = rel >= 0.25; lo = rel <= 0.08
    strong, moderate, low = [], [], []
    display = {
        "sweet": "Sweet","salty":"Salty","spicy":"Spicy","sour":"Sour / Tangy","bitter":"Bitter",
        "umami":"Umami / Savory","creamy":"Creamy","crispy":"Crispy","chewy":"Chewy","greasy":"Rich / Greasy",
        "fresh_clean":"Fresh & Clean","protein_forward":"Protein‑Forward"
    }
    for i, a in enumerate(attrs):
        label = display.get(a, a.title())
        entry = {"key": a, "label": label, "strength": float(rel[i]), "weight": float(w[i])}
        if hi[i]: strong.append(entry)
        elif lo[i]: low.append(entry)
        else: moderate.append(entry)
    strong.sort(key=lambda e: -e["strength"]); moderate.sort(key=lambda e: -e["strength"]); low.sort(key=lambda e: e["strength"])
    return strong, moderate, low, rel.tolist()

def persona_from_profile(strong):
    keys = {e["key"] for e in strong}
    title = "Savory & Protein‑Forward" if "umami" in keys or "protein_forward" in keys else "Balanced Explorer"
    blurbs = []
    if "protein_forward" in keys: blurbs.append("loves meals centered on meat, poultry, or seafood")
    if "umami" in keys: blurbs.append("craves deep, savory flavors")
    if "creamy" in keys: blurbs.append("enjoys creamy, rich textures")
    if "crispy" in keys: blurbs.append("likes crispy, crunchy bites")
    if "spicy" in keys: blurbs.append("seeks spicy heat")
    if "sweet" in keys: blurbs.append("has a sweet tooth")
    if not blurbs: blurbs.append("has flexible, context‑dependent taste")
    return title, blurbs

def dish_recs_from_profile(strong, moderate, low):
    strong_keys = {e["key"] for e in strong}
    recs = []
    if "protein_forward" in strong_keys and "umami" in strong_keys:
        recs += ["Pho with beef or chicken", "Ramen (tonkotsu/shoyu)", "Grilled steak or chicken entree", "Teriyaki salmon", "Cheeseburger with cheddar"]
    if "crispy" in strong_keys: recs += ["Crispy chicken sandwich", "Fish & chips", "Tempura appetizer"]
    if "creamy" in strong_keys: recs += ["Creamy pasta (alfredo/carbonara)", "Katsu curry", "Creamy risotto"]
    if "fresh_clean" in strong_keys: recs += ["Sashimi platter", "Mediterranean salad bowls"]
    if "sweet" in strong_keys: recs += ["Gelato, matcha parfait, or crème brûlée"]
    # dedup
    seen=set(); out=[]
    for r in recs:
        if r not in seen: seen.add(r); out.append(r)
    return out[:8] or ["Try a mixed flight of small plates to explore more dimensions."]

class NextPairResponse(BaseModel):
    left_id: int; right_id: int
    left_url: str; right_url: str
    left_name: str; right_name: str

class ClickPayload(BaseModel):
    user_id: str; left_id: int; right_id: int; choice: str

init_db(); seed_images_if_empty()
app = FastAPI(title="Taste Profile Demo", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Serve static frontend and images
app.mount("/img", StaticFiles(directory=BASE_DIR / "static" / "img"), name="img")
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

@app.get("/health")
def health(): return {"ok": True, "attrs": len(ATTRS)}

@app.get("/attrs")
def get_attributes(): return {"attributes": ATTRS, "display": DISPLAY}

@app.get("/images")
def images_list():
    ims = load_images()
    return {"count": len(ims), "images": [{"id":i["id"],"name":i["name"],"url":i["url"]} for i in ims]}

@app.get("/profile")
def get_profile(user_id: str = Query(...)):
    images = load_images()
    w, b = train_user_w(user_id, images, lr=0.5, l2=0.01, epochs=300)
    R = load_responses(user_id)
    n = len(R)
    mag = float(np.linalg.norm(w)+1e-9)
    rel = (np.abs(w) / mag).tolist() if mag > 0 else [0.0]*len(ATTRS)
    return {"user_id": user_id, "attrs": ATTRS, "w": w.tolist(), "bias": b, "n_responses": n, "relative_importance": rel}

@app.get("/profile_readable")
def profile_readable(user_id: str = Query(...)):
    images = load_images()
    w, b = train_user_w(user_id, images, lr=0.5, l2=0.01, epochs=300)
    R = load_responses(user_id)
    strong, moderate, low, rel = bucketize(ATTRS, w)
    title, blurbs = persona_from_profile(strong)
    return {"user_id": user_id, "n_responses": len(R), "persona_title": title, "persona_blurbs": blurbs, "strong": strong, "moderate": moderate, "low": low, "attrs_display": DISPLAY}

@app.get("/recommend")
def recommend(user_id: str = Query(...)):
    images = load_images()
    w, b = train_user_w(user_id, images, lr=0.5, l2=0.01, epochs=300)
    strong, moderate, low, rel = bucketize(ATTRS, w)
    recs = dish_recs_from_profile(strong, moderate, low)
    return {"user_id": user_id, "suggestions": recs}

@app.get("/next", response_model=NextPairResponse)
def next_pair(user_id: str = Query(...)):
    images = load_images()
    if len(images) < 2: raise HTTPException(400, "Need at least 2 images in catalog.")
    # current estimate
    idx = {im["id"]: im for im in images}
    # estimate w,b
    def train():
        X=[]; y=[]
        conn = db()
        rows = conn.execute("SELECT left_image_id,right_image_id,choice FROM responses WHERE user_id=? ORDER BY id ASC",(user_id,)).fetchall()
        conn.close()
        for r in rows:
            xl=idx.get(r["left"]); xr=idx.get(r["right"])
            if xl is None or xr is None: continue
            X.append(xl["x"]-xr["x"]); y.append(1 if r["choice"]=='left' else 0)
        if not X: return np.zeros(D), 0.0
        X=np.vstack(X); y=np.array(y,float)
        w=np.zeros(D,float); b=0.0
        for _ in range(200):
            z=X.dot(w)+b; p=1/(1+np.exp(-z))
            grad_w=X.T.dot(p-y)/len(y)+0.01*w
            grad_b=float(np.mean(p-y))
            w-=0.5*grad_w; b-=0.5*grad_b
        return w,b
    w,b = train()
    # sample uncertain pair
    seen=set(); conn=db()
    for r in conn.execute("SELECT left_image_id,right_image_id FROM responses WHERE user_id=? ORDER BY id DESC LIMIT 200",(user_id,)):
        a,bid=int(r["left_image_id"]), int(r["right_image_id"])
        seen.add((a,bid)); seen.add((bid,a))
    conn.close()
    def sig(z): return 1/(1+math.exp(-z))
    best=None; best_unc=999
    for _ in range(60):
        L,R = random.sample(images,2)
        if (L["id"],R["id"]) in seen: continue
        p = sig(float((L["x"]-R["x"]).dot(w) + b))
        unc = abs(p-0.5)
        if unc < best_unc:
            best_unc=unc; best=(L,R)
    if best is None: L,R = random.sample(images,2); best=(L,R)
    L,R = best
    return NextPairResponse(left_id=L["id"], right_id=R["id"], left_url=L["url"], right_url=R["url"], left_name=L["name"], right_name=R["name"])

@app.post("/click")
def record_click(payload: ClickPayload):
    if payload.choice not in ("left","right"): raise HTTPException(400, "choice must be 'left' or 'right'")
    conn = db(); cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users(id, w, b, n, updated_at) VALUES(?,?,?,?,?)",(payload.user_id, json.dumps([0.0]*D), 0.0, 0, time.time()))
    cur.execute("INSERT INTO responses(user_id, ts, left_image_id, right_image_id, choice) VALUES(?,?,?,?,?)",
                (payload.user_id, time.time(), payload.left_id, payload.right_id, payload.choice))
    cur.execute("UPDATE users SET n=COALESCE(n,0)+1, updated_at=? WHERE id=?", (time.time(), payload.user_id))
    conn.commit(); conn.close()
    return {"ok": True}
