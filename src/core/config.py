"""
financial_rag_eval/src/core/config.py
Implements metadata capture, hyperparameter management, and JSON-based immutability.
"""

import os
import json
import platform
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional
import torch

class ExperimentConfiguration:
    """
    Manages global configurations and extracts hardware and software execution contexts.
    Ensures that every evaluated pipeline parameter is logged deterministically.
    """
    def __init__(self, experiment_id: str, base_path: str, custom_params: Optional[Dict[str, Any]] = None):
        self.experiment_id: str = experiment_id
        self.base_path: str = os.path.abspath(base_path)
        self.timestamp: str = datetime.utcnow().isoformat() + "Z"
        
        # Core Default Hyperparameters
        self.params: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "random_seed": 42,
            "dataset_version": "MASTER_FINANCIAL_CORPUS_V1.csv",
            "embedding_model": "BAAI/bge-m3",
            "generation_llm": "Qwen/Qwen2.5-7B-Instruct",
            "cross_validation_folds": 5,
            "retrieval_depth_first_stage": 100,
            "top_k_context": 7,
            "mmr_lambda": 0.7,
            "rrf_constant": 60,
            "query_expansion_count": 3
        }

        if custom_params:
            self.params.update(custom_params)

        self._capture_environment_metadata()

    def _get_git_hash(self) -> str:
        """Retrieves the active git commit hash for code tracking."""
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            return "UNKNOWN_OR_NOT_A_GIT_REPO"

    def _capture_environment_metadata(self) -> None:
        """Collects exhaustive hardware, software, OS, and CUDA metadata states."""
        env_meta: Dict[str, Any] = {
            "git_commit_hash": self._get_git_hash(),
            "python_version": platform.python_version(),
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "cpu_architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        self.params["environment_metadata"] = env_meta

    def save(self, destination_directory: str) -> str:
        """
        Serializes configuration metadata to disk as an immutable record.
        """
        target_file = os.path.join(destination_directory, f"config_{self.experiment_id}.json")
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4, sort_keys=True)
        return target_file

    def get(self, key: str) -> Any:
        """Retrieves targeted pipeline parameters dynamically."""
        if key not in self.params:
            raise KeyError(f"Requested configuration field '{key}' does not exist.")
        return self.params[key]