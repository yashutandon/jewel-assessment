"""
feature_extractor.py
--------------------
Deep Learning semantic feature extraction using a pre-trained ResNet50 model.
This replaces the classical OpenCV approach to provide perfect and accurate 
visual similarity matching based on high-level patterns and semantics.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def preprocess_image(img: Image.Image) -> np.ndarray:
    # Resize to 256x256
    img = img.resize((256, 256), Image.Resampling.BILINEAR)
    
    # Center crop 224x224
    width, height = img.size
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array and scale to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # HWC to CHW
    img_np = np.transpose(img_np, (2, 0, 1))
    
    return img_np

class DeepFeatureExtractor:
    def __init__(self):
        logger.info("Loading ONNX ResNet50 model...")
        self.session = ort.InferenceSession("resnet50.onnx")
        self.input_name = self.session.get_inputs()[0].name

    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract the 2048-dimensional semantic embedding for a single image using ONNX.
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return np.zeros(2048, dtype=np.float32)

        tensor_np = preprocess_image(img)
        tensor_np = np.expand_dims(tensor_np, axis=0) # Add batch dimension

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: tensor_np})
        embedding_np = outputs[0].flatten()

        # L2 Normalize
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
