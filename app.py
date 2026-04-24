# app.py
import os
import re
import hashlib
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# huggingface client
from huggingface_hub import InferenceClient

# load .env
load_dotenv()

# Config
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_TTL = 3600

if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_HUB_TOKEN not set in .env or environment")

# init client (uses provider default; you can pass provider="nscale" if needed)
client = InferenceClient(api_key=HF_TOKEN)

# simple in-memory cache for embeddings
_embed_cache: Dict[str, Any] = {}

def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def cache_get(key: str):
    entry = _embed_cache.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > CACHE_TTL:
        del _embed_cache[key]
        return None
    return value

def cache_set(key: str, value):
    _embed_cache[key] = (time.time(), value)

def normalize_text(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def get_embedding(text: str) -> np.ndarray:
    key = _cache_key(text)
    cached = cache_get(key)
    if cached is not None:
        return cached

    # appel InferenceClient pour embeddings
    # la méthode 'embeddings' retourne généralement une liste/array de floats
    try:
        resp = client.embeddings(model=MODEL_ID, inputs=text)
    except Exception as e:
        raise RuntimeError(f"Hugging Face InferenceClient error: {e}")

    # resp peut être une liste de floats ou structure imbriquée ; normaliser en numpy array
    if isinstance(resp, dict) and "error" in resp:
        raise RuntimeError(f"HF error: {resp['error']}")
    # si resp est une liste de listes (token vectors), tenter d'extraire un vecteur poolé
    vec = None
    if isinstance(resp, list):
        # si c'est une liste de nombres -> vecteur direct
        if resp and isinstance(resp[0], (int, float)):
            vec = np.array(resp, dtype=float)
        else:
            # chercher récursivement le premier vecteur numérique
            def find_vector(obj):
                if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
                    return obj
                if isinstance(obj, list):
                    for it in obj:
                        v = find_vector(it)
                        if v is not None:
                            return v
                return None
            v = find_vector(resp)
            if v is not None:
                vec = np.array(v, dtype=float)
    if vec is None:
        raise RuntimeError("Impossible d'extraire un embedding depuis la réponse HF")

    cache_set(key, vec)
    return vec

def semantic_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    emb_a = get_embedding(a)
    emb_b = get_embedding(b)
    sim = cosine_similarity([emb_a], [emb_b])[0][0]
    if sim < 0:
        sim = (sim + 1) / 2
    return float(sim)

def keywords_presence_ratio(keywords: List[str], cv_text: str) -> float:
    if not keywords:
        return 0.0
    cv = normalize_text(cv_text).lower()
    found = 0
    for k in keywords:
        if not k:
            continue
        if re.search(r"\b" + re.escape(k.lower()) + r"\b", cv):
            found += 1
    return found / len(keywords)

def combined_score(semantic: float, kw_ratio: float, w_semantic: float = 0.8, w_kw: float = 0.2) -> float:
    s = w_semantic * semantic + w_kw * kw_ratio
    return round(s * 100, 2)

# FastAPI
class ScoreRequest(BaseModel):
    job_title: Optional[str] = None
    job_text: str
    job_keywords: Optional[List[str]] = []
    cv_text: str

app = FastAPI(title="CV Scorer - HF InferenceClient")

@app.post("/score")
async def score(payload: ScoreRequest) -> Dict[str, Any]:
    if not payload.job_text or not payload.cv_text:
        raise HTTPException(status_code=400, detail="job_text and cv_text are required")
    try:
        semantic = semantic_similarity(payload.job_text, payload.cv_text)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    kw_ratio = keywords_presence_ratio(payload.job_keywords or [], payload.cv_text)
    final = combined_score(semantic, kw_ratio)
    return {
        "job_title": payload.job_title,
        "semantic_similarity": round(semantic, 4),
        "keyword_presence_ratio": round(kw_ratio, 4),
        "final_score_percent": final
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "cache_size": len(_embed_cache)}
