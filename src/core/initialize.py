"""
financial_rag_eval/src/core/initialize.py
Provides programmatic directory workspace structure setup with integrity locks.
"""

import os
import sys
import logging
from typing import Dict, List

def setup_workspace(base_dir: str, experiment_id: str) -> Dict[str, str]:
    """
    Initializes a publication-quality directory structure for the evaluation pipeline.
    Prevents accidental overwrites by checking for existing experimental workspaces.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("WorkspaceSetup")

    exp_dir = os.path.join(base_dir, "experiments", f"EXP_{experiment_id}")
    
    if os.path.exists(exp_dir):
        logger.error(f"Safety Violation: Workspace 'EXP_{experiment_id}' already exists at {exp_dir}.")
        raise FileExistsError(f"Aborting execution to prevent overwriting existing experiment artifacts.")

    paths_to_create: List[str] = [
        os.path.join(base_dir, "configs"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "embeddings"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "paper"),
        exp_dir,
        os.path.join(exp_dir, "checkpoints"),
        os.path.join(exp_dir, "artifacts"),
        os.path.join(exp_dir, "metrics"),
        os.path.join(exp_dir, "statistics"),
        os.path.join(exp_dir, "plots"),
        os.path.join(exp_dir, "tables"),
        os.path.join(exp_dir, "logs"),
    ]

    resolved_paths: Dict[str, str] = {}
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)
        key = os.path.basename(path) if "EXP_" not in os.path.basename(path) else f"exp_{experiment_id}"
        resolved_paths[key] = os.path.abspath(path)
        
    logger.info(f"Workspace successfully provisioned for Experiment ID: {experiment_id}")
    return resolved_paths