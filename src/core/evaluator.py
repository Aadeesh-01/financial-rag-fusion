"""
financial_rag_eval/src/evaluation/evaluator.py
Calculates rigorous IR metrics (nDCG, MRR, Recall, Diversity, Drift) for retrieved contexts.
"""

import math
import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("EvaluationManager")

class EvaluationManager:
    """
    Computes strict ranking and semantic metrics for evaluated retrieval pipelines.
    """
    def __init__(self, top_k: int = 7):
        self.top_k = top_k

    def calculate_ndcg(self, retrieved_relevance_scores: List[float]) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain (nDCG@k).
        Expects a list of relevance scores (e.g., 0 to 3) for the retrieved documents.
        """
        k = min(self.top_k, len(retrieved_relevance_scores))
        dcg = sum([rel / math.log2(idx + 2) for idx, rel in enumerate(retrieved_relevance_scores[:k])])
        
        # Ideal DCG (sort relevance scores descending)
        ideal_scores = sorted(retrieved_relevance_scores, reverse=True)
        idcg = sum([rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_scores[:k])])
        
        return dcg / idcg if idcg > 0 else 0.0

    def calculate_mrr(self, retrieved_relevance_scores: List[float], relevance_threshold: float = 1.0) -> float:
        """
        Calculates Mean Reciprocal Rank (MRR). 
        Considers a document relevant if its score >= relevance_threshold.
        """
        for idx, rel in enumerate(retrieved_relevance_scores[:self.top_k]):
            if rel >= relevance_threshold:
                return 1.0 / (idx + 1)
        return 0.0

    def calculate_recall(self, retrieved_relevance_scores: List[float], total_relevant_in_corpus: int, relevance_threshold: float = 1.0) -> float:
        """
        Calculates Recall@k. Requires knowing the total number of relevant documents in the fold.
        """
        if total_relevant_in_corpus == 0:
            return 0.0
            
        relevant_retrieved = sum(1 for rel in retrieved_relevance_scores[:self.top_k] if rel >= relevance_threshold)
        return relevant_retrieved / total_relevant_in_corpus

    def calculate_intra_list_diversity(self, retrieved_embeddings: np.ndarray) -> float:
        """
        Calculates Intra-List Diversity (ILD) using 1 - average pairwise cosine similarity.
        Penalizes redundant context windows.
        """
        k = min(self.top_k, retrieved_embeddings.shape[0])
        if k <= 1:
            return 0.0 # No diversity to measure in a single/zero document list
            
        embeddings_k = retrieved_embeddings[:k]
        sim_matrix = cosine_similarity(embeddings_k)
        
        # Extract upper triangle excluding diagonal
        upper_tri_indices = np.triu_indices(k, k=1)
        mean_sim = np.mean(sim_matrix[upper_tri_indices])
        
        return float(1.0 - mean_sim)

    def calculate_semantic_drift(self, original_query_emb: np.ndarray, expanded_query_embs: np.ndarray) -> float:
        """
        Measures the cosine distance between the original query and the LLM variations.
        Lower similarity = Higher drift (potential hallucination).
        """
        if expanded_query_embs.shape[0] == 0:
            return 0.0
            
        original_vector = original_query_emb.reshape(1, -1)
        sims = cosine_similarity(original_vector, expanded_query_embs)[0]
        return float(np.mean(sims))

    def evaluate_query_run(self, 
                           retrieved_relevance_scores: List[float], 
                           retrieved_embeddings: np.ndarray,
                           total_relevant: int,
                           original_query_emb: np.ndarray = None,
                           expanded_query_embs: np.ndarray = None) -> Dict[str, float]:
        """Runs the full evaluation suite for a single query."""
        metrics = {
            f"ndcg@{self.top_k}": self.calculate_ndcg(retrieved_relevance_scores),
            f"mrr@{self.top_k}": self.calculate_mrr(retrieved_relevance_scores),
            f"recall@{self.top_k}": self.calculate_recall(retrieved_relevance_scores, total_relevant),
            f"ild@{self.top_k}": self.calculate_intra_list_diversity(retrieved_embeddings)
        }
        
        if original_query_emb is not None and expanded_query_embs is not None:
            metrics["semantic_drift"] = self.calculate_semantic_drift(original_query_emb, expanded_query_embs)
            
        return metrics