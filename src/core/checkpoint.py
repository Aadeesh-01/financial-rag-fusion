"""
financial_rag_eval/src/core/checkpoint.py
Implements structured pipeline checkpointing and recovery tracking.
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("CheckpointManager")

class CheckpointManager:
    """
    Manages active state tracking across folds and unique target queries.
    Enables zero-overhead resume capabilities for multi-day pipelines.
    """
    def __init__(self, experiment_workspace: str):
        self.workspace = os.path.abspath(experiment_workspace)
        self.checkpoint_dir = os.path.join(self.workspace, "checkpoints")
        self.state_file = os.path.join(self.checkpoint_dir, "pipeline_state.json")
        self.state: Dict[str, Any] = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> Dict[str, Any]:
        """Loads a pre-existing state file or constructs a fresh pipeline context."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    loaded_state = json.load(f)
                logger.info("Found existing checkpoint state. Resuming pipeline execution.")
                return loaded_state
            except Exception as e:
                logger.error(f"Checkpoint state file corrupted: {e}. Generating new timeline tracker.")
                
        return {
            "current_fold": 0,
            "completed_folds": [],
            "processed_queries_by_fold": {},
            "execution_metadata": {}
        }

    def commit_query_checkpoint(self, fold_idx: int, query_id: str, pipeline_key: str) -> None:
        """Marks an individual query block complete within the active fold."""
        fold_str = str(fold_idx)
        if fold_str not in self.state["processed_queries_by_fold"]:
            self.state["processed_queries_by_fold"][fold_str] = []
            
        record_key = f"{query_id}::{pipeline_key}"
        if record_key not in self.state["processed_queries_by_fold"][fold_str]:
            self.state["processed_queries_by_fold"][fold_str].append(record_key)
            
        self._flush_to_disk()

    def commit_fold_checkpoint(self, fold_idx: int) -> None:
        """Marks a fold completely evaluated and appends it to the safety register."""
        if fold_idx not in self.state["completed_folds"]:
            self.state["completed_folds"].append(fold_idx)
        self.state["current_fold"] = fold_idx + 1
        self._flush_to_disk()
        logger.info(f"Fold {fold_idx} fully finalized and saved to checkpoint registry.")

    def is_query_processed(self, fold_idx: int, query_id: str, pipeline_key: str) -> bool:
        """Verifies if an individual query has already been calculated and stored."""
        fold_str = str(fold_idx)
        if fold_str not in self.state["processed_queries_by_fold"]:
            return False
        return f"{query_id}::{pipeline_key}" in self.state["processed_queries_by_fold"][fold_str]

    def is_fold_completed(self, fold_idx: int) -> bool:
        """Verifies if an entire fold's calculations can be safely skipped."""
        return fold_idx in self.state["completed_folds"]

    def _flush_to_disk(self) -> None:
        """Saves current state tracking records to disk using atomic replacement."""
        temp_path = f"{self.state_file}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=4)
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            os.rename(temp_path, self.state_file)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.critical(f"Critical Error: Checkpoint persistence failure: {e}")