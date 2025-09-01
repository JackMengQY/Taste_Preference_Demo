import os, json, sqlite3, time, math, random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

# Writable DB path for Render
DB_PATH = Path(os.environ.get("DB_PATH", "/tmp/taste.db"))

ATTRS = ["sweet","salty","spicy","sour","bitter","umami","creamy","crispy","chewy","greasy","fresh_clean","protein_forward"]
DISPLAY = {
  "sweet":"Sweet","salty":"Salty","spicy":"Spicy","sour":"Sour / Tangy","bitter":"Bitter",
  "umami":"Umami / Savory","creamy":"Creamy","crispy":"Crispy","chewy":"Chewy",
  "greasy":"Rich / Greasy","fresh_clean":"Fresh & Clean","protein_forward":"Protein-Forward"
}
D = len(ATTRS)

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS images(id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, name TEXT, attrs TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, w TEXT, b REAL, n INTEGER DEFAULT 0, updated_at REAL)")
    c.execute("""CREATE TABLE IF NOT EXISTS responses(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, ts REAL, left_image_id INTEGER, right_image_id INTEGER,
        choice TEXT CHECK(choice IN ('left','right'))
    )""")
    conn.commit(); conn.close()

def seed_images_if_empty():
    conn = db()
    cur = conn.cursor()
    n = cur.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
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
            (u("yogurt_granola.png"), "Yogurt & Granola Bowl", [0.55,0.05,0.0,0.1,0.05,0.2,0.8,0.2,0.2,0.1,0.8,0.3]),
            (u("sashimi.png"), "Sashimi Platter", [0.0,0.2,0.0,0.05,0.05,0.6,0.2,0.1,0.3,0.05,0.95,0.9]),
            (u("buffalo_wings.png"), "Buffalo Wings", [0.05,0.6,0.75,0.05,0.05,0.7,0.2,0.6,0.2,0.7,0.15,0.5])
        ]
        rows = [(url, name, json.dumps(attrs)) for (url, name, attrs) in data]  # JSON text
        cur.executemany("INSERT INTO images(url,name,attrs) VALUES(?,?,?)", rows)
        conn.commit()
    conn.close()

def load_images():
    conn = db()
    rows = conn.execute("SELECT id,url,name,attrs FROM images").fetchall()
    conn.close()
    return [
        {"id": r["id"], "url": r["url"], "name": r["name"], "x": np.array(json.loads(r["attrs"]), dtype=float)}
        for r in rows
    ]

def load_responses(uid: str):
    conn = db()
    rows = conn.execute(
        "SELECT left_image_id,right_image_id,choice FROM responses WHERE user_id=? ORDER BY id ASC", (uid,)
    ).fetchall()
    conn.close()
    return [{"left": r["left_image_id"], "right": r["right_image_id"], "choice": r["choice"]} for r in rows]

# BEFORE
# def sigmoid(z): return 1.0/(1.0+math.exp(-z))

# AFTER
import numpy as np  # (already imported above)
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def train_user_w(uid: str, images: List[Dict[str,Any]], lr=0.5, l2=0.01, epochs=250):
    idx = {im["id"]: im for im in images}
    R = load_responses(uid)
    if not R: return np.zeros(D), 0.0
    X, y = [], []
    for r in R:
        L, Rg = idx.get(r["left"]), idx.get(r["right"])
        if L is None or Rg is None: continue
        X.append(L["x"] - Rg["x"])
        y.append(1 if r["choice"] == "left" else 0)
    if not X: return np.zeros(D), 0.0
    X, y = np.vstack(X), np.array(y, float)
    w, b = np.zeros(D), 0.0
    for _ in range(epochs):
        p = sigmoid(X.dot(w) + b)
        w -= lr * (X.T.dot(p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b

# ---------- App ----------
init_db()
seed_images_if_empty()

app = FastAPI(title="Taste Demo", version="0.4.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- tiny attrs endpoint (optional for UI) ---
@app.get("/attrs")
def get_attrs(): return {"attributes": ATTRS, "display": DISPLAY}

@app.get("/health")
def health(): return {"ok": True, "attrs": len(ATTRS)}

@app.get("/images")
def images_list():
    ims = load_images()
    return {"count": len(ims), "images": [{"id": i["id"], "name": i["name"], "url": i["url"]} for i in ims]}

class NextPairResponse(BaseModel):
    left_id: int; right_id: int
    left_url: str; right_url: str
    left_name: str; right_name: str

@app.get("/next", response_model=NextPairResponse)
def next_pair(user_id: str = Query(...)):
    ims = load_images()
    if len(ims) < 2:
        raise HTTPException(400, "Not enough images.")
    w, b = train_user_w(user_id, ims)
    best = None; best_unc = 999.0
    for _ in range(120):
        L, R = random.sample(ims, 2)
        try:
            p = sigmoid(float((L["x"] - R["x"]).dot(w) + b))
        except Exception:
            p = 0.5
        unc = abs(p - 0.5)
        if unc < best_unc:
            best_unc, best = unc, (L, R)
    if best is None:
        L, R = random.sample(ims, 2); best = (L, R)
    L, R = best
    return NextPairResponse(
        left_id=L["id"], right_id=R["id"], left_url=L["url"], right_url=R["url"], left_name=L["name"], right_name=R["name"]
    )

class ClickPayload(BaseModel):
    user_id: str; left_id: int; right_id: int; choice: str

@app.post("/click")
def record_click(payload: ClickPayload):
    if payload.choice not in ("left","right"):
        raise HTTPException(400, "choice must be 'left' or 'right'")
    conn = db()
    conn.execute("INSERT OR IGNORE INTO users(id, w, b, n, updated_at) VALUES(?,?,?,?,?)",
                 (payload.user_id, json.dumps([0.0]*D), 0.0, 0, time.time()))
    conn.execute("INSERT INTO responses(user_id, ts, left_image_id, right_image_id, choice) VALUES(?,?,?,?,?)",
                 (payload.user_id, time.time(), payload.left_id, payload.right_id, payload.choice))
    conn.execute("UPDATE users SET n=COALESCE(n,0)+1, updated_at=? WHERE id=?", (time.time(), payload.user_id))
    conn.commit(); conn.close()
    return {"ok": True}

# ---------- Profile/report ----------
def bucketize(attrs, w):
    absw = np.abs(w); mag = float(np.linalg.norm(w) + 1e-9)
    rel = absw / mag if mag > 0 else np.zeros_like(absw)
    hi = rel >= 0.25; lo = rel <= 0.08
    strong, moderate, low = [], [], []
    for i, a in enumerate(attrs):
        entry = {"key": a, "label": DISPLAY.get(a, a.title()), "strength": float(rel[i]), "weight": float(w[i])}
        if hi[i]: strong.append(entry)
        elif lo[i]: low.append(entry)
        else: moderate.append(entry)
    strong.sort(key=lambda e: -e["strength"]); moderate.sort(key=lambda e: -e["strength"]); low.sort(key=lambda e: e["strength"])
    return strong, moderate, low, rel.tolist()

def persona_from_profile(strong):
    keys = {e["key"] for e in strong}
    title = "Savory & Protein-Forward" if ("umami" in keys or "protein_forward" in keys) else "Balanced Explorer"
    blurbs = []
    if "protein_forward" in keys: blurbs.append("loves meals centered on meat, poultry, or seafood")
    if "umami" in keys: blurbs.append("craves deep, savory flavors")
    if "creamy" in keys: blurbs.append("enjoys creamy, rich textures")
    if "crispy" in keys: blurbs.append("likes crispy, crunchy bites")
    if "spicy" in keys: blurbs.append("seeks spicy heat")
    if "sweet" in keys: blurbs.append("has a sweet tooth")
    if not blurbs: blurbs.append("has flexible, context-dependent taste")
    return title, blurbs

@app.get("/profile")
def get_profile(user_id: str = Query(...)):
    imgs = load_images()
    w, b = train_user_w(user_id, imgs, lr=0.5, l2=0.01, epochs=300)
    R = load_responses(user_id); n = len(R)
    mag = float(np.linalg.norm(w)+1e-9)
    rel = (np.abs(w) / mag).tolist() if mag > 0 else [0.0]*len(ATTRS)
    return {"user_id": user_id, "attrs": ATTRS, "display": DISPLAY,
            "w": [float(v) for v in w.tolist()], "bias": float(b),
            "n_responses": n, "relative_importance": rel}

@app.get("/profile_readable")
def profile_readable(user_id: str = Query(...)):
    imgs = load_images()
    w, _ = train_user_w(user_id, imgs, lr=0.5, l2=0.01, epochs=300)
    R = load_responses(user_id)
    strong, moderate, low, rel = bucketize(ATTRS, w)
    title, blurbs = persona_from_profile(strong)
    return {"user_id": user_id, "n_responses": len(R), "persona_title": title,
            "persona_blurbs": blurbs, "strong": strong, "moderate": moderate, "low": low,
            "attrs_display": DISPLAY}

# ---------- Static mounts at the very end ----------
BASE_DIR = Path(__file__).parent
app.mount("/img", StaticFiles(directory=BASE_DIR / "static" / "img"), name="img")
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
