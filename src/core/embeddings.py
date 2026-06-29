"""
financial_rag_eval/src/core/embeddings.py
Manages the generation, verification, and persistence of dense vector representations.
"""

import os
import json
import hashlib
import logging
import time
from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("EmbeddingManager")

class EmbeddingManager:
    """
    Handles BGE-M3 embedding generation with strict integrity and caching mechanisms.
    Ensures dimensions and checksums match before loading cached matrices.
    """
    def __init__(self, workspace: str, model_name: str = "BAAI/bge-m3", batch_size: int = 32):
        self.workspace = os.path.abspath(workspace)
        self.embeddings_dir = os.path.join(self.workspace, "..", "..", "embeddings") # Shared across experiments
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None

    def _load_model(self):
        """Lazy loads the embedding model to preserve VRAM until explicitly needed."""
        if self._model is None:
            logger.info(f"Loading embedding model {self.model_name} onto {self.device}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()

    def _generate_checksum(self, filepath: str) -> str:
        """Generates an MD5 checksum for matrix integrity verification."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_or_create_embeddings(self, corpus_df: pd.DataFrame, text_column: str, corpus_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Retrieves existing verified embeddings or generates new ones if unavailable/corrupted.
        """
        matrix_path = os.path.join(self.embeddings_dir, f"{corpus_id}_embeddings.npy")
        metadata_path = os.path.join(self.embeddings_dir, f"{corpus_id}_metadata.json")

        # 1. Attempt to load and verify existing embeddings
        if os.path.exists(matrix_path) and os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                logger.info(f"Verifying existing embeddings for {corpus_id}...")
                current_checksum = self._generate_checksum(matrix_path)
                
                if current_checksum != metadata["checksum"]:
                    raise ValueError("Checksum mismatch: The embedding file is corrupted.")
                    
                embeddings = np.load(matrix_path)
                
                if embeddings.shape[0] != len(corpus_df):
                    raise ValueError(f"Dimension mismatch: Expected {len(corpus_df)} rows, found {embeddings.shape[0]}.")
                    
                logger.info("Embeddings successfully verified and loaded from cache.")
                return embeddings, metadata
                
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Recomputing...")

        # 2. Generate new embeddings
        self._load_model()
        texts = corpus_df[text_column].tolist()
        
        logger.info(f"Generating embeddings for {len(texts)} documents. This may take a while...")
        start_time = time.perf_counter()
        
        # BGE-M3 outputs normalized embeddings by default, which is required for FAISS Inner Product (Cosine)
        embeddings = self._model.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True
        ).astype(np.float32)
        
        generation_time = time.perf_counter() - start_time
        
        # 3. Store and Verify
        np.save(matrix_path, embeddings)
        checksum = self._generate_checksum(matrix_path)
        
        metadata = {
            "model": self.model_name,
            "corpus_id": corpus_id,
            "dimensions": embeddings.shape[1],
            "num_documents": embeddings.shape[0],
            "normalized": True,
            "generation_time_seconds": generation_time,
            "checksum": checksum
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Embeddings generated and saved securely with checksum {checksum}.")
        return embeddings, metadata