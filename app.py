"""
app.py
------
FastAPI application for the Jewellery Earring Recommender (Production Ready).
Uses Deep Learning for accurate matching and caches features for instant startup.
"""

from __future__ import annotations

import csv
import logging
import pickle
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config import settings
from feature_extractor import extract_features, compute_similarity

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global State Types
# ---------------------------------------------------------------------------
# We will store the dataset and precomputed features in app.state
# during the lifespan event.

class AppState:
    necklaces: List[Dict[str, str]] = []
    earrings: List[Dict[str, str]] = []
    necklace_features: Dict[str, Any] = {}
    earring_features: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_dataset() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Read the candidate_dataset.csv and return necklaces and earrings."""
    products: List[Dict[str, str]] = []
    with open(settings.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip().lstrip("\ufeff"): v.strip() for k, v in row.items()}
            products.append(clean)
            
    necklaces = [p for p in products if p["product_type"] == "Necklace"]
    earrings = [p for p in products if p["product_type"] == "Earrings"]
    return necklaces, earrings

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager runs before the app starts accepting requests.
    We use it to load the dataset, ML model, and cache.
    """
    logger.info(f"Starting {settings.app_name}...")
    
    # Ensure cache directory exists
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = settings.cache_dir / "features_cache.pkl"
    
    # 1. Load Dataset
    necklaces, earrings = load_dataset()
    app.state.necklaces = necklaces
    app.state.earrings = earrings
    logger.info(f"Loaded {len(necklaces)} necklaces and {len(earrings)} earrings.")

    # 2. Load or Compute Features
    cache_loaded = False
    if cache_file.exists():
        logger.info(f"Loading precomputed features from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            app.state.necklace_features = cache_data["necklaces"]
            app.state.earring_features = cache_data["earrings"]
            cache_loaded = True
            logger.info("Cache loaded successfully. Startup instant!")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}. Will recompute.")
            cache_file.unlink(missing_ok=True)

    if not cache_loaded:
        logger.info("No cache found. Precomputing ML embeddings (this will take a moment)...")

        nck_feats = {}
        for nck in necklaces:
            img_path = str(settings.images_dir / nck["image_file"])
            nck_feats[nck["id"]] = extract_features(img_path)

        ear_feats = {}
        for ear in earrings:
            img_path = str(settings.images_dir / ear["image_file"])
            ear_feats[ear["id"]] = extract_features(img_path)

        app.state.necklace_features = nck_feats
        app.state.earring_features = ear_feats

        # Save to cache
        logger.info(f"Saving computed features to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "necklaces": nck_feats,
                "earrings": ear_feats,
            }, f)

    logger.info("Application startup complete. Ready to serve requests.")
    yield
    
    logger.info("Shutting down application...")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    description="Recommends matching earrings using Deep Learning embeddings",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    necklace_id: str
    top_k: int = 5

class EarringMatch(BaseModel):
    id: str
    image_file: str
    total_score: float

class RecommendResponse(BaseModel):
    necklace_id: str
    necklace_image: str
    recommendations: List[EarringMatch]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page HTML frontend."""
    return HTMLResponse(content=settings.template_path.read_text(encoding="utf-8"))

@app.get("/necklaces")
async def list_necklaces(request: Request):
    """Return the list of necklaces available for selection."""
    return [
        {"id": n["id"], "image_file": n["image_file"]}
        for n in request.app.state.necklaces
    ]

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest, request: Request):
    """
    Given a necklace ID, compute similarity against all earrings and return
    the top-k matches ranked by total score.
    """
    necklaces = request.app.state.necklaces
    earrings = request.app.state.earrings
    nck_feats = request.app.state.necklace_features
    ear_feats = request.app.state.earring_features
    
    # Validate necklace ID
    necklace = next((n for n in necklaces if n["id"] == req.necklace_id), None)
    if necklace is None:
        raise HTTPException(status_code=404, detail=f"Necklace '{req.necklace_id}' not found")

    target_emb = nck_feats[req.necklace_id]

    # Score every earring
    scored: List[Dict[str, Any]] = []
    for ear in earrings:
        cand_emb = ear_feats[ear["id"]]
        score = compute_similarity(target_emb, cand_emb)
        scored.append({
            "id": ear["id"],
            "image_file": ear["image_file"],
            "total_score": round(score * 100, 2),  # Store as a clean percentage out of 100
        })

    # Sort descending by total score and take top k
    scored.sort(key=lambda x: x["total_score"], reverse=True)
    top = scored[: req.top_k]

    return RecommendResponse(
        necklace_id=req.necklace_id,
        necklace_image=necklace["image_file"],
        recommendations=[EarringMatch(**m) for m in top],
    )

@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve a jewellery image file."""
    filepath = settings.images_dir / filename
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)
