# app.py
import os
import re
import time
import hashlib
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ===============================
# LOAD ENV
# ===============================
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_HUB_TOKEN not found")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_TTL = 3600

client = InferenceClient(api_key=HF_TOKEN)

app = FastAPI(title="CV Scorer API")

# ===============================
# CACHE
# ===============================
_embed_cache: Dict[str, Any] = {}


def cache_key(text: str):
    return hashlib.sha256(text.encode()).hexdigest()


def cache_get(key):
    if key not in _embed_cache:
        return None

    ts, val = _embed_cache[key]

    if time.time() - ts > CACHE_TTL:
        del _embed_cache[key]
        return None

    return val


def cache_set(key, value):
    _embed_cache[key] = (time.time(), value)


# ===============================
# HELPERS
# ===============================
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)

    if denom == 0:
        return 0.0

    return float(np.dot(v1, v2) / denom)


# ===============================
# EMBEDDING VIA SENTENCE SIMILARITY
# ===============================
def hf_similarity(text1: str, text2: str) -> float:
    """
    HuggingFace sentence similarity API
    """

    try:
        result = client.post(
            json={
                "inputs": {
                    "source_sentence": text1,
                    "sentences": [text2]
                }
            },
            model=MODEL_ID
        )

        if isinstance(result, bytes):
            import json
            result = json.loads(result.decode())

        return float(result[0])

    except Exception as e:
        raise RuntimeError(str(e))


# ===============================
# KEYWORDS SCORE
# ===============================
def keyword_score(keywords: List[str], cv_text: str) -> float:
    if not keywords:
        return 0.0

    cv = cv_text.lower()

    found = 0

    for kw in keywords:
        if kw.lower() in cv:
            found += 1

    return found / len(keywords)


# ===============================
# FINAL SCORE
# ===============================
def final_score(semantic: float, keywords: float):
    score = semantic * 0.8 + keywords * 0.2
    return round(score * 100, 2)


# ===============================
# REQUEST MODEL
# ===============================
class ScoreRequest(BaseModel):
    job_title: Optional[str] = None
    job_text: str
    job_keywords: Optional[List[str]] = []
    cv_text: str


# ===============================
# ROUTES
# ===============================
@app.post("/score")
async def score(data: ScoreRequest):

    try:
        semantic = hf_similarity(
            clean_text(data.job_text),
            clean_text(data.cv_text)
        )

        kw = keyword_score(
            data.job_keywords,
            data.cv_text
        )

        score = final_score(semantic, kw)

        return {
            "job_title": data.job_title,
            "semantic_similarity": round(semantic, 4),
            "keyword_presence_ratio": round(kw, 4),
            "final_score_percent": score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID
    }
