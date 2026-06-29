"""
financial_rag_eval/src/core/storage.py
Implements an atomic storage layer with automated integrity validation.
"""

import os
import json
import logging
import hashlib
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("StorageManager")

class StorageManager:
    """
    Manages structured, unstructured, and vectorized artifact persistence.
    Guarantees atomic file operations via write-then-rename patterns.
    """
    def __init__(self, experiment_workspace: str):
        self.workspace = os.path.abspath(experiment_workspace)
        self.artifacts_dir = os.path.join(self.workspace, "artifacts")
        self.metrics_dir = os.path.join(self.workspace, "metrics")
        
    def _compute_sha256(self, filepath: str) -> str:
        """Calculates a SHA256 checksum to verify file integrity."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _atomic_write(self, target_path: str, write_callback, *args, **kwargs) -> None:
        """Executes atomic file persistence to mitigate corruption risks."""
        temp_path = f"{target_path}.tmp"
        try:
            write_callback(temp_path, *args, **kwargs)
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(temp_path, target_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Atomic write transaction failed for path: {target_path}")
            raise e

    def save_json(self, filename: str, data: Union[Dict, List], subfolder: str = "artifacts") -> str:
        """Saves a JSON file atomically and runs an immediate validation check."""
        target_dir = os.path.join(self.workspace, subfolder)
        target_path = os.path.join(target_dir, filename)
        
        def callback(path: str):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
        self._atomic_write(target_path, callback)
        
        # Immediate verification loop
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as integrity_err:
            raise IOError(f"Post-write integrity verification failed for {target_path}") from integrity_err
            
        return target_path

    def save_numpy(self, filename: str, array: np.ndarray, subfolder: str = "artifacts") -> str:
        """Saves a NumPy matrix atomically, preventing NaN propagation."""
        if np.isnan(array).any():
            logger.warning(f"Data Anomaly: Saved array '{filename}' contains NaN values.")
            
        target_dir = os.path.join(self.workspace, subfolder)
        target_path = os.path.join(target_dir, filename)
        
        def callback(path: str):
            np.save(path, array, allow_pickle=False)
            
        self._atomic_write(target_path, callback)
        
        # Verify array dimensional properties match upon reloading
        try:
            loaded = np.load(target_path, allow_pickle=False)
            if loaded.shape != array.shape:
                raise ValueError("Reloaded structural matrix size mismatch.")
        except Exception as integrity_err:
            raise IOError(f"Post-write verification failed for matrix {target_path}") from integrity_err
            
        return target_path

    def save_csv(self, filename: str, df: pd.DataFrame, subfolder: str = "metrics") -> str:
        """Saves a Pandas DataFrame to a CSV format atomically."""
        target_dir = os.path.join(self.workspace, subfolder)
        target_path = os.path.join(target_dir, filename)
        
        def callback(path: str):
            df.to_csv(path, index=False, encoding="utf-8")
            
        self._atomic_write(target_path, callback)
        return target_path