"""
financial_rag_eval/src/retrieval/retriever.py
Implements the core retrieval pipelines: FAISS, BM25, MMR, and RRF strategies.
"""

import time
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("RetrievalManager")

class RetrievalManager:
    """
    Executes and times multiple retrieval strategies, ensuring deterministic outputs.
    """
    def __init__(self, corpus_texts: List[str], corpus_embeddings: np.ndarray):
        self.corpus_texts = corpus_texts
        self.corpus_embeddings = corpus_embeddings
        self.num_docs = len(corpus_texts)
        
        # Initialize Indices
        logger.info("Initializing Sparse and Dense Indices...")
        self._init_bm25()
        self._init_faiss()

    def _init_bm25(self):
        """Initializes the BM25 lexical index."""
        tokenized_corpus = [doc.lower().split() for doc in self.corpus_texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def _init_faiss(self):
        """Initializes the FAISS exact search index (Inner Product for normalized vectors)."""
        dim = self.corpus_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.corpus_embeddings)

    def retrieve_bm25(self, query: str, top_k: int = 100) -> Dict[str, Any]:
        """Executes standard lexical search."""
        start_time = time.perf_counter()
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        latency = (time.perf_counter() - start_time) * 1000  # ms
        
        return {
            "retrieved_ids": top_indices.tolist(),
            "scores": top_scores.tolist(),
            "retrieval_latency_ms": latency,
            "reranking_latency_ms": 0.0
        }

    def retrieve_faiss(self, query_embedding: np.ndarray, top_k: int = 100) -> Dict[str, Any]:
        """Executes standard dense vector search."""
        start_time = time.perf_counter()
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        scores, indices = self.faiss_index.search(query_vector, top_k)
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            "retrieved_ids": indices[0].tolist(),
            "scores": scores[0].tolist(),
            "retrieval_latency_ms": latency,
            "reranking_latency_ms": 0.0
        }

    def retrieve_mmr(self, query_embedding: np.ndarray, fetch_k: int = 100, top_k: int = 7, lambda_param: float = 0.7) -> Dict[str, Any]:
        """Executes dense retrieval followed by Maximum Marginal Relevance reranking."""
        # First Stage: Dense Retrieval
        first_stage = self.retrieve_faiss(query_embedding, fetch_k)
        candidate_ids = first_stage["retrieved_ids"]
        candidate_embs = self.corpus_embeddings[candidate_ids]
        
        # Second Stage: MMR Reranking
        start_rerank = time.perf_counter()
        query_vector = query_embedding.reshape(1, -1)
        
        selected_indices = [0] # Greedily pick the absolute best match first
        unselected_indices = list(range(1, len(candidate_embs)))
        
        query_sims = cosine_similarity(query_vector, candidate_embs)[0]
        doc_sims = cosine_similarity(candidate_embs)
        
        while len(selected_indices) < top_k and unselected_indices:
            mmr_scores = []
            for i in unselected_indices:
                q_sim = query_sims[i]
                max_d_sim = max([doc_sims[i][j] for j in selected_indices])
                mmr_score = lambda_param * q_sim - (1 - lambda_param) * max_d_sim
                mmr_scores.append((mmr_score, i))
                
            best_idx = max(mmr_scores, key=lambda x: x[0])[1]
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
            
        final_ids = [candidate_ids[i] for i in selected_indices]
        rerank_latency = (time.perf_counter() - start_rerank) * 1000
        
        return {
            "retrieved_ids": final_ids,
            "scores": [float(query_sims[i]) for i in selected_indices],
            "retrieval_latency_ms": first_stage["retrieval_latency_ms"],
            "reranking_latency_ms": rerank_latency
        }

    def retrieve_rag_fusion(self, query_embeddings: List[np.ndarray], top_k: int = 7, rrf_k: int = 60) -> Dict[str, Any]:
        """
        Executes Reciprocal Rank Fusion on multiple query variations (from LLM).
        `query_embeddings` contains the original query + N LLM-generated variations.
        """
        start_retrieval = time.perf_counter()
        all_rankings = []
        
        # Retrieve independently for each query variation
        for q_emb in query_embeddings:
            res = self.retrieve_faiss(q_emb, top_k=20) # Fetch more per query for fusion depth
            all_rankings.append(res["retrieved_ids"])
            
        retrieval_latency = (time.perf_counter() - start_retrieval) * 1000
        
        # RRF Reranking
        start_rerank = time.perf_counter()
        rrf_scores = {}
        for rank_list in all_rankings:
            for rank, doc_id in enumerate(rank_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
                
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_ids = [doc_id for doc_id, score in sorted_docs][:top_k]
        final_scores = [score for doc_id, score in sorted_docs][:top_k]
        rerank_latency = (time.perf_counter() - start_rerank) * 1000
        
        return {
            "retrieved_ids": final_ids,
            "scores": final_scores,
            "retrieval_latency_ms": retrieval_latency,
            "reranking_latency_ms": rerank_latency
        }