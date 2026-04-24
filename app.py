# app.py
import os
import re
import time
import hashlib
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
HF_TOKEN_ENV = "HUGGINGFACE_HUB_TOKEN"
HF_EMBED_API = "https://api-inference.huggingface.co/embeddings"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # modèle d'embeddings
REQUEST_TIMEOUT = 30  # secondes
CACHE_TTL = 3600  # secondes pour cache en mémoire (optionnel)

# --- Simple in-memory cache pour embeddings (clé -> (timestamp, vector)) ---
_embed_cache: Dict[str, Any] = {}

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

# --- Helpers ---
def normalize_text(text: Optional[str]) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def keywords_presence_ratio(keywords: List[str], cv_text: str) -> float:
    if not keywords:
        return 0.0
    cv = normalize_text(cv_text).lower()
    found = 0
    for k in keywords:
        if not k:
            continue
        pattern = r"\b" + re.escape(k.lower()) + r"\b"
        if re.search(pattern, cv):
            found += 1
    return found / len(keywords)

def _cache_key_for_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_embeddings_from_hf(texts: List[str]) -> List[np.ndarray]:
    """
    Appelle l'API HF embeddings. Envoie la liste 'texts' en une seule requête si possible.
    Utilise un cache mémoire simple pour éviter recomputation.
    """
    if not texts:
        return []

    # préparer résultats et liste à demander
    results = []
    to_request = []
    request_indices = []

    for i, t in enumerate(texts):
        key = _cache_key_for_text(t)
        cached = cache_get(key)
        if cached is not None:
            results.append(cached)
        else:
            results.append(None)
            to_request.append(t)
            request_indices.append(i)

    if to_request:
        hf_token = os.environ.get(HF_TOKEN_ENV)
        if not hf_token:
            raise RuntimeError("HUGGINGFACE_HUB_TOKEN not set in environment")

        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"model": MODEL_ID, "input": to_request}
        resp = requests.post(HF_EMBED_API, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"Hugging Face API error {resp.status_code}: {resp.text}")
        emb_json = resp.json()
        # emb_json should be a list of vectors (one per input)
        if not isinstance(emb_json, list):
            raise RuntimeError("Unexpected HF response format for embeddings")

        # convert to numpy arrays and fill results + cache
        for idx, vec in enumerate(emb_json):
            vec_arr = np.array(vec, dtype=float)
            target_pos = request_indices[idx]
            results[target_pos] = vec_arr
            cache_set(_cache_key_for_text(texts[target_pos]), vec_arr)

    return results

def semantic_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    emb_a, emb_b = get_embeddings_from_hf([a, b])
    sim = cosine_similarity([emb_a], [emb_b])[0][0]
    # normaliser si négatif
    if sim < 0:
        sim = (sim + 1) / 2
    return float(sim)

def combined_score(semantic: float, kw_ratio: float, w_semantic: float = 0.8, w_kw: float = 0.2) -> float:
    s = w_semantic * semantic + w_kw * kw_ratio
    return round(s * 100, 2)

# --- FastAPI app ---
class ScoreRequest(BaseModel):
    job_title: Optional[str] = None
    job_text: str
    job_keywords: Optional[List[str]] = []
    cv_text: str

app = FastAPI(title="CV Scorer (HF embeddings)")

@app.post("/score")
async def score(payload: ScoreRequest) -> Dict[str, Any]:
    if not payload.job_text:
        raise HTTPException(status_code=400, detail="job_text is required")
    if not payload.cv_text:
        raise HTTPException(status_code=400, detail="cv_text is required")

    job_text = payload.job_text
    cv_text = payload.cv_text
    keywords = payload.job_keywords or []

    try:
        semantic = semantic_similarity(job_text, cv_text)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    kw_ratio = keywords_presence_ratio(keywords, cv_text)
    final = combined_score(semantic, kw_ratio, w_semantic=0.8, w_kw=0.2)

    return {
        "job_title": payload.job_title,
        "semantic_similarity": round(semantic, 4),
        "keyword_presence_ratio": round(kw_ratio, 4),
        "final_score_percent": final
    }

@app.get("/health")
async def health():
    hf_token_set = bool(os.environ.get(HF_TOKEN_ENV))
    return {
        "status": "ok",
        "hf_token_set": hf_token_set,
        "model": MODEL_ID,
        "cache_size": len(_embed_cache)
    }

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
