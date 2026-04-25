import os
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ======================
# LOAD ENV
# ======================
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

MODEL_URL = (
    "https://router.huggingface.co/"
    "hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_HUB_TOKEN not set")

# ======================
# APP
# ======================
app = FastAPI(title="CV Scoring API")

# ======================
# CORS (ALL SITES)
# ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tous les domaines
    allow_credentials=True,
    allow_methods=["*"],      # GET POST PUT DELETE OPTIONS...
    allow_headers=["*"],      # tous les headers
)

# ======================
# REQUEST MODEL
# ======================
class ScoreRequest(BaseModel):
    job_title: Optional[str] = None
    job_text: str
    job_keywords: Optional[List[str]] = []
    cv_text: str


# ======================
# HF SIMILARITY
# ======================
def hf_similarity(
    job_text: str,
    job_keywords: List[str],
    cv_text: str,
) -> float:

    job_full = job_text

    if job_keywords:
        job_full += " Keywords: " + ", ".join(job_keywords)

    payload = {
        "inputs": {
            "source_sentence": job_full,
            "sentences": [cv_text],
        }
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        MODEL_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    result = response.json()

    return float(result[0])


# ======================
# SCORE ROUTE
# ======================
@app.post("/score")
async def score(data: ScoreRequest):
    try:
        semantic = hf_similarity(
            data.job_text,
            data.job_keywords or [],
            data.cv_text,
        )

        final_score = round(semantic * 100, 2)

        return {
            "job_title": data.job_title,
            "semantic_similarity": round(semantic, 4),
            "final_score_percent": final_score,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ======================
# HEALTH CHECK
# ======================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }
