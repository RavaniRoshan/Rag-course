"""
RAG Course - Module 1: Reranking Implementation

This file contains the implementation of the Advanced Reranker class
as described in the module. It includes all the functionality from the
theoretical explanation and is designed to be testable and production-ready.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional
import numpy as np
import time
from collections import defaultdict


class AdvancedReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rerank_with_confidence(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Rerank candidates with confidence scoring and thresholding
        """
        # Prepare sentence pairs for cross-encoder
        sentence_pairs = [[query, candidate["text"]] for candidate in candidates]

        # Get similarity scores
        similarity_scores = self.model.predict(sentence_pairs)

        # Enhance candidates with scores and metadata
        for idx, candidate in enumerate(candidates):
            candidate["relevance_score"] = float(similarity_scores[idx])
            candidate["confidence"] = self._calculate_confidence(similarity_scores[idx])
            candidate["reranked_position"] = idx

        # Filter by threshold and sort
        filtered_candidates = [
            candidate for candidate in candidates
            if candidate["relevance_score"] >= score_threshold
        ]

        # Sort by relevance score
        filtered_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

        return filtered_candidates[:top_k]

    def _calculate_confidence(self, score: float) -> str:
        """Calculate confidence level based on score"""
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"

    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Dict]],
        batch_size: int = 32
    ) -> List[List[Dict]]:
        """Batch processing for multiple queries"""
        results = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            batch_candidates = candidates_list[i:i+batch_size]

            batch_results = []
            for query, candidates in zip(batch_queries, batch_candidates):
                reranked = self.rerank_with_confidence(query, candidates)
                batch_results.append(reranked)

            results.extend(batch_results)

        return results


class OptimizedReranker:
    def __init__(self, model_name: str, optimization_level: str = "balanced"):
        self.model = CrossEncoder(model_name)
        self.optimization_level = optimization_level

    def optimized_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Optimized reranking with various latency reduction techniques"""

        if self.optimization_level == "speed":
            return self._fast_rerank(query, candidates)
        elif self.optimization_level == "balanced":
            return self._balanced_rerank(query, candidates)
        else:  # "accuracy"
            return self._accurate_rerank(query, candidates)

    def _fast_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Speed-optimized reranking"""
        # Pre-filter candidates using cheap methods
        pre_filtered = self._pre_filter_candidates(query, candidates)

        # Use smaller batch sizes
        batch_size = 8
        results = []

        for i in range(0, len(pre_filtered), batch_size):
            batch = pre_filtered[i:i+batch_size]
            sentence_pairs = [[query, candidate["text"]] for candidate in batch]
            scores = self.model.predict(sentence_pairs, batch_size=batch_size)

            for j, candidate in enumerate(batch):
                candidate["relevance_score"] = float(scores[j])

            results.extend(batch)

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:5]  # Return fewer results for speed

    def _pre_filter_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Pre-filter using cheap similarity measures"""
        query_terms = set(query.lower().split())

        filtered = []
        for candidate in candidates:
            candidate_terms = set(candidate["text"].lower().split())
            overlap = len(query_terms.intersection(candidate_terms))
            jaccard_similarity = overlap / len(query_terms.union(candidate_terms))

            if jaccard_similarity > 0.1:  # Threshold for pre-filtering
                filtered.append(candidate)

        return filtered


class RerankingEvaluator:
    def __init__(self, test_dataset: List[Dict]):
        self.test_dataset = test_dataset

    def evaluate_reranker(self, reranker: AdvancedReranker) -> Dict:
        """Comprehensive evaluation of reranker performance"""

        metrics = {
            'precision @k': [],
            'recall @k': [],
            'ndcg @k': [],
            'mrr': [],
            'latency': []
        }

        for test_case in self.test_dataset:
            query = test_case['query']
            candidates = test_case['candidates']
            relevant_docs = set(test_case['relevant_docs'])

            start_time = time.time()
            reranked_results = reranker.rerank_with_confidence(query, candidates)
            latency = time.time() - start_time

            # Calculate metrics
            precision = self._calculate_precision(reranked_results, relevant_docs, k=5)
            recall = self._calculate_recall(reranked_results, relevant_docs, k=5)
            ndcg = self._calculate_ndcg(reranked_results, relevant_docs, k=5)
            mrr = self._calculate_mrr(reranked_results, relevant_docs)

            metrics['precision @k'].append(precision)
            metrics['recall @k'].append(recall)
            metrics['ndcg @k'].append(ndcg)
            metrics['mrr'].append(mrr)
            metrics['latency'].append(latency)

        # Aggregate results
        aggregated_metrics = {}
        for metric, values in metrics.items():
            aggregated_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return aggregated_metrics

    def _calculate_precision(self, results: List[Dict], relevant_docs: set, k: int) -> float:
        """Calculate precision at k"""
        retrieved_docs = set([doc.get('id') for doc in results[:k]])
        if len(retrieved_docs) == 0:
            return 0.0
        relevant_retrieved = retrieved_docs.intersection(relevant_docs)
        return len(relevant_retrieved) / len(retrieved_docs)

    def _calculate_recall(self, results: List[Dict], relevant_docs: set, k: int) -> float:
        """Calculate recall at k"""
        retrieved_docs = set([doc.get('id') for doc in results[:k]])
        if len(relevant_docs) == 0:
            return 0.0
        relevant_retrieved = retrieved_docs.intersection(relevant_docs)
        return len(relevant_retrieved) / len(relevant_docs)

    def _calculate_ndcg(self, results: List[Dict], relevant_docs: set, k: int) -> float:
        """Calculate normalized discounted cumulative gain at k"""
        if not results:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(results[:k]):
            doc_id = doc.get('id')
            if doc_id in relevant_docs:
                relevance = 1  # For binary relevance
                dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_docs), k)):
            idcg += 1 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_mrr(self, results: List[Dict], relevant_docs: set) -> float:
        """Calculate mean reciprocal rank"""
        for i, doc in enumerate(results):
            if doc.get('id') in relevant_docs:
                return 1 / (i + 1)  # Return reciprocal rank of first relevant doc
        return 0.0


class CustomDomainReranker(AdvancedReranker):
    def __init__(self, model_name: str, domain_keywords: List[str]):
        super().__init__(model_name)
        self.domain_keywords = set(keyword.lower() for keyword in domain_keywords)
    
    def rerank_with_domain_boost(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10,
        domain_boost_factor: float = 0.2
    ) -> List[Dict]:
        """
        Rerank with additional scoring based on domain keyword matches
        """
        # First get base relevance scores
        base_results = self.rerank_with_confidence(query, candidates, top_k=len(candidates))
        
        # Apply domain-specific boosting
        for candidate in base_results:
            text = candidate["text"].lower()
            keyword_matches = sum(1 for keyword in self.domain_keywords if keyword in text)
            
            # Boost score for domain-relevant content
            if keyword_matches > 0:
                candidate["domain_boost_score"] = keyword_matches * domain_boost_factor
                candidate["enhanced_relevance_score"] = (
                    candidate["relevance_score"] + 
                    candidate["domain_boost_score"]
                )
            else:
                candidate["domain_boost_score"] = 0
                candidate["enhanced_relevance_score"] = candidate["relevance_score"]
        
        # Re-sort by enhanced score
        base_results.sort(key=lambda x: x["enhanced_relevance_score"], reverse=True)
        return base_results[:top_k]


def main():
    """Example usage of the Advanced Reranker"""
    print("RAG Course - Module 1: Reranking Example")
    print("=" * 50)
    
    # Sample documents
    sample_documents = [
        {"id": "doc1", "text": "This document explains how to implement API calls in the new framework"},
        {"id": "doc2", "text": "General information about programming concepts"},
        {"id": "doc3", "text": "Detailed guide to using the SDK with code examples"},
        {"id": "doc4", "text": "Unrelated content about cooking recipes"},
        {"id": "doc5", "text": "Step-by-step tutorial for setting up the development environment"},
        {"id": "doc6", "text": "Advanced debugging techniques for complex applications"},
        {"id": "doc7", "text": "Best practices for code optimization and performance"},
        {"id": "doc8", "text": "Understanding memory management in modern systems"},
        {"id": "doc9", "text": "Overview of machine learning algorithms and applications"},
        {"id": "doc10", "text": "Introduction to cloud computing and deployment strategies"}
    ]
    
    # Initialize reranker
    reranker = AdvancedReranker()
    
    # Test query
    query = "how to implement API"
    
    print(f"Query: {query}")
    print(f"Number of documents to rerank: {len(sample_documents)}")
    print()
    
    # Perform reranking
    results = reranker.rerank_with_confidence(query, sample_documents, top_k=5)
    
    print("Top 5 results after reranking:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Document ID: {result['id']}")
        print(f"   Text: {result['text'][:80]}...")
        print(f"   Relevance Score: {result['relevance_score']:.3f}")
        print(f"   Confidence: {result['confidence']}")
        print()
    
    # Test domain-specific reranker
    print("\nTesting Domain-Specific Reranker:")
    print("-" * 40)
    
    tech_keywords = ["API", "SDK", "framework", "library", "implementation", "code", "function", "class"]
    tech_reranker = CustomDomainReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        domain_keywords=tech_keywords
    )
    
    domain_results = tech_reranker.rerank_with_domain_boost(query, sample_documents, top_k=5)
    
    print("Top 5 results after domain-boosted reranking:")
    for i, result in enumerate(domain_results, 1):
        print(f"{i}. Document ID: {result['id']}")
        print(f"   Text: {result['text'][:80]}...")
        print(f"   Original Score: {result['relevance_score']:.3f}")
        print(f"   Domain Boost: {result['domain_boost_score']:.3f}")
        print(f"   Enhanced Score: {result['enhanced_relevance_score']:.3f}")
        print(f"   Confidence: {result['confidence']}")
        print()


if __name__ == "__main__":
    main()