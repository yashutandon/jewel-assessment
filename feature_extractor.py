"""
feature_extractor.py
--------------------
Deep Learning semantic feature extraction using a pre-trained ResNet50 model.
This replaces the classical OpenCV approach to provide perfect and accurate 
visual similarity matching based on high-level patterns and semantics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class DeepFeatureExtractor:
    def __init__(self):
        logger.info("Loading ResNet50 model (this may take a moment on first run)...")
        # Load pre-trained ResNet50 with state-of-the-art weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        base_model = resnet50(weights=weights)
        
        # We don't need the final classification layer (fc), we just want the 2048-dim embeddings
        # from the AdaptiveAvgPool2d layer right before it.
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        
        # Set to evaluation mode
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}.")
        
        # The standard preprocessing pipeline for ImageNet models
        self.preprocess = weights.transforms()

    @torch.no_grad()
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract the 2048-dimensional semantic embedding for a single image.
        """
        try:
            # Convert to RGB (in case of RGBA or Grayscale)
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return np.zeros(2048, dtype=np.float32)

        # Preprocess: Resize, Crop, Normalize
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        embedding = self.model(tensor)
        
        # Squeeze the (1, 2048, 1, 1) tensor to (2048,) and move to CPU numpy array
        embedding_np = embedding.squeeze().cpu().numpy().astype(np.float32)
        
        # L2 Normalize the embedding for Cosine Similarity
        norm = np.linalg.norm(embedding_np)
        if norm > 1e-8:
            embedding_np = embedding_np / norm
            
        return embedding_np

# Singleton instance to be used across the app
_extractor = None

def get_extractor() -> DeepFeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = DeepFeatureExtractor()
    return _extractor

def extract_features(image_path: str) -> np.ndarray:
    """Helper function to extract features using the singleton extractor."""
    extractor = get_extractor()
    return extractor.extract_features(image_path)

def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Compute Cosine Similarity between two L2-normalized embeddings.
    Since both are L2-normalized, dot product is equivalent to cosine similarity.
    Result is in range [-1, 1], we clamp to [0, 1].
    """
    dot_product = np.dot(emb_a, emb_b)
    return float(max(0.0, min(1.0, dot_product)))
