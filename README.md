# Jewellery Earring Recommender

A web application that recommends matching earrings for a selected necklace from a provided jewellery inventory, using visual similarity powered by Deep Learning.

---

## Approach: How Images Are Compared

### Problem
Given a selected necklace image, find the most visually similar earrings from an inventory of 15 candidates. The matching needs to go beyond simple pixel/colour comparison — it should understand **style, patterns, textures, and structural motifs** (e.g., temple jewellery vs. contemporary diamond pieces).

### Solution: Deep Learning Semantic Embeddings (ResNet50)

I use a **pre-trained ResNet50 convolutional neural network** (trained on ImageNet) as a feature extractor. Instead of using the model to classify images, I remove its final classification layer and extract the **2048-dimensional embedding vector** from the second-to-last layer. This vector acts as a rich, semantic "fingerprint" of the image.

**Why this works for jewellery:**
- ResNet50 has learned to recognize **high-level visual concepts** — shapes, textures, patterns, colour relationships, and structural complexity — from millions of images.
- Two pieces of jewellery that share a similar style (e.g., both are antique gold Jhumkas with temple motifs) will produce embedding vectors that are **close together** in this 2048-dimensional space.
- This is far more accurate than classical approaches (e.g., raw colour histograms or edge detection), which only capture low-level pixel statistics and miss the semantic meaning of the design.

**Matching Pipeline:**
1. Each image is preprocessed (resized to 224×224, normalized) and passed through ResNet50.
2. The resulting 2048-dim vector is **L2-normalized**.
3. **Cosine Similarity** (dot product of normalized vectors) is used to compare embeddings — yielding a score from 0 (no match) to 1 (identical).
4. Earrings are ranked by this score and the top 5 are returned.

**Performance Optimization:**
- All embeddings are precomputed at startup and **cached to disk** (`.cache/features_cache.pkl`). Subsequent server starts load the cache instantly (< 1 second) instead of re-extracting features.

---

## Technologies & Tools

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | FastAPI + Uvicorn | Async web framework and ASGI server |
| **ML Model** | PyTorch + torchvision (ResNet50) | Pre-trained CNN for semantic feature extraction |
| **Image Processing** | Pillow (PIL) | Image loading and format conversion |
| **Similarity** | NumPy | Cosine similarity via dot product |
| **Configuration** | pydantic-settings | Environment-based config management |
| **Data** | pandas / csv | Dataset loading |
| **Frontend** | HTML / CSS / JavaScript | Single-page UI (dark theme, responsive) |
| **Deployment** | Docker | Containerized production deployment |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# Open in browser
# http://127.0.0.1:8000
```

Or with Docker:
```bash
docker build -t jewellery-recommender .
docker run -p 8000:8000 jewellery-recommender
```

---

## Project Structure

```
├── app.py                  # FastAPI backend (endpoints, caching, lifespan)
├── feature_extractor.py    # ResNet50 feature extraction & cosine similarity
├── config.py               # Centralized settings (pydantic-settings)
├── requirements.txt        # Python dependencies
├── candidate_dataset.csv   # Product inventory (5 necklaces, 15 earrings)
├── Dockerfile              # Production container
├── templates/
│   └── index.html          # Frontend UI
└── Jewelry Images/         # Product images (Nck_*.jpg, Ear_*.jpg)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/necklaces` | Lists available necklaces |
| `POST` | `/recommend` | Returns top-k matching earrings for a necklace |
| `GET` | `/images/{filename}` | Serves jewellery images |
