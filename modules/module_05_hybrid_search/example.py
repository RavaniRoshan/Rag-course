"""
Module 5: Hybrid Search
Implementation Examples

This module demonstrates hybrid search techniques combining keyword search (BM25) 
and semantic search (vector similarity) to improve retrieval accuracy.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import uuid


class HybridSearchBase:
    """Base class for hybrid search implementations"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.doc_ids = []
    
    def add_documents(self, documents: List[str]) -> List[str]:
        """Add documents to the search system"""
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)
        return doc_ids
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        raise NotImplementedError


class BM25Search:
    """BM25 keyword search implementation"""
    
    def __init__(self, documents: List[str]):
        # Preprocess documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if score > 0
                results.append({
                    'doc_id': idx,
                    'document': self.documents[idx],
                    'score': float(scores[idx]),
                    'rank': len(results) + 1
                })
        
        return results


class VectorSearch:
    """Vector similarity search implementation"""
    
    def __init__(self, documents: List[str], embedder: SentenceTransformer):
        self.documents = documents
        self.embedder = embedder
        
        # Generate embeddings
        embeddings = self.embedder.encode(documents)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search using vector similarity"""
        # Generate query embedding
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Perform similarity search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.documents):  # Valid index check
                results.append({
                    'doc_id': int(idx),
                    'document': self.documents[idx],
                    'score': float(scores[0][i]),  # Cosine similarity score
                    'rank': len(results) + 1
                })
        
        return results


class HybridSearch(HybridSearchBase):
    """Hybrid search combining BM25 and vector search"""
    
    def __init__(self, use_reranker: bool = False):
        super().__init__()
        self.bm25_search = None
        self.vector_search = None
        self.use_reranker = use_reranker
        self.reranker = None
        
        if use_reranker:
            # Load a cross-encoder for re-ranking
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def add_documents(self, documents: List[str]) -> List[str]:
        """Add documents and initialize search components"""
        doc_ids = super().add_documents(documents)
        
        # Initialize search components
        self.bm25_search = BM25Search(documents)
        self.vector_search = VectorSearch(documents, self.embedder)
        
        return doc_ids
    
    def search_bm25(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform BM25 search"""
        return self.bm25_search.search(query, k)
    
    def search_vectors(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform vector search"""
        return self.vector_search.search(query, k)
    
    def hybrid_search_rrf(self, query: str, k: int = 10, keyword_weight: float = 0.5) -> List[Dict[str, Any]]:
        """Perform hybrid search using Reciprocal Rank Fusion"""
        # Get results from both search methods
        bm25_results = self.bm25_search.search(query, k=k)
        vector_results = self.vector_search.search(query, k=k)
        
        # Create a combined results dictionary
        all_docs = {}
        
        # Add BM25 results with rank-based scoring
        for idx, result in enumerate(bm25_results):
            doc_id = result['doc_id']
            rank = result['rank']
            # RRF scoring: 1 / (k + rank), where k is smoothing parameter (commonly 60)
            score = 1.0 / (60 + rank)
            all_docs[doc_id] = {
                'document': result['document'],
                'bm25_score': result['score'],
                'vector_score': 0.0,
                'rrf_score': score,
                'bm25_rank': rank,
                'vector_rank': float('inf')
            }
        
        # Add or update with vector results
        for idx, result in enumerate(vector_results):
            doc_id = result['doc_id']
            rank = result['rank']
            
            if doc_id in all_docs:
                # Update existing entry
                all_docs[doc_id]['vector_score'] = result['score']
                all_docs[doc_id]['vector_rank'] = rank
                # Add RRF score from vector search
                all_docs[doc_id]['rrf_score'] += 1.0 / (60 + rank)
            else:
                # Add new entry
                all_docs[doc_id] = {
                    'document': result['document'],
                    'bm25_score': 0.0,
                    'vector_score': result['score'],
                    'rrf_score': 1.0 / (60 + rank),
                    'bm25_rank': float('inf'),
                    'vector_rank': rank
                }
        
        # Sort by RRF score (descending)
        sorted_results = sorted(all_docs.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
        
        # Format results
        final_results = []
        for idx, (doc_id, data) in enumerate(sorted_results[:k]):
            final_results.append({
                'doc_id': doc_id,
                'document': data['document'],
                'bm25_score': data['bm25_score'],
                'vector_score': data['vector_score'],
                'rrf_score': data['rrf_score'],
                'rank': idx + 1
            })
        
        return final_results
    
    def hybrid_search_linear(self, query: str, k: int = 10, keyword_weight: float = 0.3, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Perform hybrid search using linear combination of scores"""
        # Get results from both search methods
        bm25_results = self.bm25_search.search(query, k=len(self.documents))
        vector_results = self.vector_search.search(query, k=len(self.documents))
        
        # Create score dictionaries
        bm25_scores = {r['doc_id']: r['score'] for r in bm25_results}
        vector_scores = {r['doc_id']: r['score'] for r in vector_results}
        
        # Normalize scores to 0-1 range
        if bm25_scores:
            bm25_max = max(bm25_scores.values()) if bm25_scores.values() else 1.0
            bm25_min = min(bm25_scores.values()) if bm25_scores.values() else 0.0
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
            bm25_norm = {doc_id: (score - bm25_min) / bm25_range for doc_id, score in bm25_scores.items()}
        else:
            bm25_norm = {}
        
        if vector_scores:
            vector_max = max(vector_scores.values()) if vector_scores.values() else 1.0
            vector_min = min(vector_scores.values()) if vector_scores.values() else 0.0
            vector_range = vector_max - vector_min if vector_max != vector_min else 1.0
            vector_norm = {doc_id: (score - vector_min) / vector_range for doc_id, score in vector_scores.items()}
        else:
            vector_norm = {}
        
        # Calculate combined scores
        combined_scores = {}
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_norm.get(doc_id, 0.0)
            vector_score = vector_norm.get(doc_id, 0.0)
            combined_score = keyword_weight * bm25_score + semantic_weight * vector_score
            combined_scores[doc_id] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        final_results = []
        for idx, (doc_id, score) in enumerate(sorted_results[:k]):
            final_results.append({
                'doc_id': doc_id,
                'document': self.documents[doc_id],
                'combined_score': score,
                'bm25_score': bm25_scores.get(doc_id, 0.0),
                'vector_score': vector_scores.get(doc_id, 0.0),
                'rank': idx + 1
            })
        
        return final_results
    
    def search(self, query: str, k: int = 5, method: str = 'rrf') -> List[Dict[str, Any]]:
        """Default search using RRF method"""
        if method == 'rrf':
            return self.hybrid_search_rrf(query, k)
        elif method == 'linear':
            return self.hybrid_search_linear(query, k)
        else:
            raise ValueError(f"Unknown method: {method}")


class HybridSearchEvaluator:
    """Evaluator for hybrid search systems"""
    
    def __init__(self):
        self.sample_docs = [
            "Machine learning algorithms can improve with more data",
            "Deep learning is a subset of machine learning",
            "Natural language processing helps computers understand text",
            "Python is a popular programming language for AI",
            "Vector embeddings represent text in high-dimensional space",
            "Semantic search finds meaning rather than keywords",
            "Information retrieval systems use various ranking algorithms",
            "Neural networks learn patterns from data",
            "Data science combines statistics and programming",
            "Artificial intelligence aims to simulate human intelligence"
        ]
        
        self.sample_queries = [
            "AI and machine learning",
            "programming languages for data science",
            "understanding human language with computers",
            "neural networks and pattern recognition"
        ]
    
    def evaluate_method(self, search_system: HybridSearch, method: str, k: int = 5) -> Dict[str, float]:
        """Evaluate a specific hybrid search method"""
        start_time = time.time()
        
        # Perform searches for all sample queries
        all_results = []
        for query in self.sample_queries:
            results = search_system.search(query, k, method=method)
            all_results.append(results)
        
        total_time = time.time() - start_time
        
        return {
            'method': method,
            'avg_query_time': total_time / len(self.sample_queries),
            'total_time': total_time,
            'num_results': len(all_results)
        }
    
    def compare_methods(self, search_system: HybridSearch) -> List[Dict[str, float]]:
        """Compare different hybrid search methods"""
        methods = ['rrf', 'linear']
        results = []
        
        for method in methods:
            eval_result = self.evaluate_method(search_system, method)
            results.append(eval_result)
        
        return results


def advanced_hybrid_techniques():
    """Demonstrate advanced hybrid search techniques"""
    print("=== Advanced Hybrid Search Techniques ===\n")
    
    # Create sample documents
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing helps computers understand human language",
        "Vector databases store high-dimensional embeddings for similarity search",
        "Retrieval augmented generation combines retrieval and generation models",
        "Cosine similarity measures the angle between two vectors",
        "Embeddings represent text in high-dimensional vector space",
        "Chroma is an open-source vector database for AI applications",
        "FAISS is Facebook's library for efficient similarity search",
        "Pinecone is a managed vector database service"
    ]
    
    # Initialize hybrid search
    hybrid_search = HybridSearch()
    hybrid_search.add_documents(docs)
    
    # Example query
    query = "vector databases and similarity search"
    
    print(f"Query: '{query}'\n")
    
    # BM25 only results
    bm25_results = hybrid_search.search_bm25(query, k=5)
    print("BM25 Results:")
    for idx, result in enumerate(bm25_results, 1):
        print(f"  {idx}. Score: {result['score']:.3f} - {result['document']}")
    print()
    
    # Vector only results
    vector_results = hybrid_search.search_vectors(query, k=5)
    print("Vector Search Results:")
    for idx, result in enumerate(vector_results, 1):
        print(f"  {idx}. Score: {result['score']:.3f} - {result['document']}")
    print()
    
    # Hybrid RRF results
    rrf_results = hybrid_search.hybrid_search_rrf(query, k=5)
    print("Hybrid RRF Results:")
    for idx, result in enumerate(rrf_results, 1):
        print(f"  {idx}. RRF Score: {result['rrf_score']:.3f} - {result['document']}")
    print()
    
    # Hybrid Linear results
    linear_results = hybrid_search.hybrid_search_linear(query, k=5)
    print("Hybrid Linear Results:")
    for idx, result in enumerate(linear_results, 1):
        print(f"  {idx}. Combined Score: {result['combined_score']:.3f} - {result['document']}")
    print()


def parameter_tuning_example():
    """Show how to tune parameters for hybrid search"""
    print("=== Parameter Tuning for Hybrid Search ===\n")
    
    # Sample documents
    docs = [
        "Python programming language basics",
        "Machine learning algorithms explained",
        "Data science with Python and pandas",
        "Natural language processing techniques",
        "Deep learning neural networks tutorial",
        "Web development with Python frameworks",
        "Statistical analysis in data science",
        "Computer vision and image processing"
    ]
    
    # Initialize hybrid search
    hybrid_search = HybridSearch()
    hybrid_search.add_documents(docs)
    
    query = "data science and Python"
    
    print(f"Tuning parameters for query: '{query}'\n")
    
    # Try different weights for linear combination
    weights = [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]
    
    for kw_weight, sem_weight in weights:
        results = hybrid_search.hybrid_search_linear(query, k=3, keyword_weight=kw_weight, semantic_weight=sem_weight)
        print(f"Keyword Weight: {kw_weight}, Semantic Weight: {sem_weight}")
        for idx, result in enumerate(results, 1):
            print(f"  {idx}. Score: {result['combined_score']:.3f} - {result['document']}")
        print()


def main():
    """Main function to demonstrate hybrid search implementations"""
    print("Module 5: Hybrid Search")
    print("=" * 50)
    
    # Create sample documents
    sample_docs = [
        "Machine learning algorithms can improve with more data",
        "Deep learning is a subset of machine learning",
        "Natural language processing helps computers understand text",
        "Python is a popular programming language for AI",
        "Vector embeddings represent text in high-dimensional space",
        "Semantic search finds meaning rather than keywords",
        "Information retrieval systems use various ranking algorithms",
        "Neural networks learn patterns from data",
        "Data science combines statistics and programming",
        "Artificial intelligence aims to simulate human intelligence"
    ]
    
    # Initialize hybrid search
    hybrid_search = HybridSearch()
    hybrid_search.add_documents(sample_docs)
    
    # Demonstrate different search methods
    query = "AI and machine learning concepts"
    
    print(f"Query: '{query}'\n")
    
    # Test RRF method
    print("Reciprocal Rank Fusion (RRF) Results:")
    rrf_results = hybrid_search.hybrid_search_rrf(query, k=5)
    for idx, result in enumerate(rrf_results, 1):
        print(f"  {idx}. RRF Score: {result['rrf_score']:.3f} - {result['document']}")
    print()
    
    # Test Linear Combination method
    print("Linear Combination Results:")
    linear_results = hybrid_search.hybrid_search_linear(query, k=5)
    for idx, result in enumerate(linear_results, 1):
        print(f"  {idx}. Combined Score: {result['combined_score']:.3f} - {result['document']}")
    print()
    
    # Performance comparison
    evaluator = HybridSearchEvaluator()
    eval_results = evaluator.compare_methods(hybrid_search)
    
    print("Performance Comparison:")
    print(f"{'Method':<15} {'Avg Query Time':<15} {'Total Time':<12}")
    print("-" * 45)
    for result in eval_results:
        print(f"{result['method']:<15} {result['avg_query_time']:<15.4f} {result['total_time']:<12.4f}")
    print()
    
    # Show advanced techniques
    advanced_hybrid_techniques()
    
    # Show parameter tuning
    parameter_tuning_example()


if __name__ == "__main__":
    main()