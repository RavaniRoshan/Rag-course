"""
Module 12: Optimization Techniques
Implementation Examples

This module demonstrates optimization techniques for RAG systems,
including caching, resource management, and performance improvements.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import time
import uuid
import json
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
import threading
import asyncio
import heapq
from collections import OrderedDict, defaultdict
import pickle
import hashlib
from sklearn.decomposition import PCA
import faiss


class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from the cache"""
        if key in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put an item in the cache"""
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_capacity:
            # Remove least recently used item
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def size(self) -> int:
        """Get current size of cache"""
        return len(self.cache)
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()


class QueryResultCache:
    """Cache for query results to avoid recomputation"""
    
    def __init__(self, max_capacity: int = 500, ttl_seconds: int = 3600):
        self.cache = LRUCache(max_capacity)
        self.ttl_seconds = ttl_seconds
        self.timestamps = {}  # Track creation time of each entry
    
    def _get_key(self, query: str, top_k: int, filters: Dict[str, Any] = None) -> str:
        """Generate a cache key for the query"""
        key_data = {
            'query': query,
            'top_k': top_k,
            'filters': filters or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached query results"""
        key = self._get_key(query, top_k, filters)
        
        # Check if entry exists and hasn't expired
        if key in self.timestamps:
            if datetime.now() - self.timestamps[key] > timedelta(seconds=self.ttl_seconds):
                # Entry has expired, remove it
                del self.timestamps[key]
                self.cache.cache.pop(key, None)
                return None
        
        return self.cache.get(key)
    
    def put(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None, 
            results: List[Dict[str, Any]] = None):
        """Cache query results"""
        key = self._get_key(query, top_k, filters)
        self.cache.put(key, results)
        self.timestamps[key] = datetime.now()


class EmbeddingCache:
    """Cache for computed embeddings to avoid recomputation"""
    
    def __init__(self, max_capacity: int = 10000):
        self.cache = LRUCache(max_capacity)
    
    def _get_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = self._get_key(text)
        return self.cache.get(key)
    
    def put(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        key = self._get_key(text)
        self.cache.put(key, embedding)


class EmbeddingOptimizer:
    """Class to optimize embeddings for performance and memory"""
    
    def __init__(self):
        self.pca_model = None
        self.is_fitted = False
    
    def reduce_dimensions(self, embeddings: np.ndarray, target_dims: int = 128) -> np.ndarray:
        """Reduce embedding dimensions using PCA"""
        if target_dims >= embeddings.shape[1]:
            return embeddings
        
        if not self.is_fitted or self.pca_model.n_components != target_dims:
            self.pca_model = PCA(n_components=target_dims)
            reduced_embeddings = self.pca_model.fit_transform(embeddings)
            self.is_fitted = True
        else:
            reduced_embeddings = self.pca_model.transform(embeddings)
        
        return reduced_embeddings
    
    def quantize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Quantize embeddings to reduce memory footprint"""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        
        # Quantize to 8-bit integers
        quantized = ((normalized + 1) / 2 * 255).astype(np.uint8)
        return quantized
    
    def dequantize_embeddings(self, quantized_embeddings: np.ndarray) -> np.ndarray:
        """Restore quantized embeddings to float"""
        # Convert back to float and denormalize
        embeddings = (quantized_embeddings.astype(np.float32) / 255) * 2 - 1
        
        # Note: We can't fully restore the original norms, so return unit vectors
        return embeddings


class IndexOptimizer:
    """Optimize vector index for performance"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
    
    def create_optimized_index(self, embeddings: np.ndarray, 
                              index_type: str = "HNSW") -> Any:
        """Create an optimized index based on type and data"""
        if index_type == "HNSW":
            # Hierarchical Navigable Small World index
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
        elif index_type == "IVF":
            # Inverted file index with quantizer
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            if nlist < 1:
                nlist = 1
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            # Train the index if there are enough vectors
            if len(embeddings) >= nlist:
                self.index.train(embeddings.astype('float32'))
        elif index_type == "Flat":
            # Exact search index
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        return self.index


class ResourceOptimizer:
    """Manage system resources for optimal performance"""
    
    def __init__(self):
        self.current_batch_size = 32  # Default batch size
        self.max_memory_usage = 0.8  # Use up to 80% of available memory
        self.gpu_available = self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for computation"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def suggest_batch_size(self, input_length: int, available_memory_gb: float = 16.0) -> int:
        """Suggest optimal batch size based on memory constraints"""
        # Rough estimation: assume each item takes ~10MB of memory
        memory_per_item_mb = 10.0
        available_memory_mb = available_memory_gb * 1024
        
        suggested_size = int(available_memory_mb / memory_per_item_mb)
        
        # Cap the batch size to prevent excessive memory usage
        capped_size = min(suggested_size, 128)
        
        # Ensure batch size is at least 1
        final_size = max(1, capped_size)
        
        return final_size
    
    def optimize_for_gpu(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for GPU processing if available"""
        if self.gpu_available:
            try:
                import torch
                return torch.tensor(data).cuda()
            except:
                # If GPU processing fails, return original data
                return data
        else:
            return data


class QueryOptimizer:
    """Optimize query processing for better performance"""
    
    def __init__(self):
        self.query_pattern_cache = {}
        self.query_complexity_estimator = self._create_complexity_estimator()
    
    def _create_complexity_estimator(self) -> Callable[[str], float]:
        """Create a function to estimate query complexity"""
        def estimate_complexity(query: str) -> float:
            # Simple heuristic for query complexity
            complexity = len(query.split()) * 0.1  # Longer queries are more complex
            
            # Add complexity for complex terms
            complex_terms = ['compare', 'difference', 'relationship', 'vs', 'versus', 'analyze']
            for term in complex_terms:
                if term in query.lower():
                    complexity += 0.5
            
            # Cap complexity
            return min(complexity, 5.0)
        
        return estimate_complexity
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query for better performance"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase
        query = query.lower()
        
        # Remove common stop words that don't affect meaning
        # (in a real system, you might use NLTK or spaCy)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
    
    def determine_strategy(self, query: str) -> Dict[str, Any]:
        """Determine the best processing strategy for a query"""
        complexity = self.query_complexity_estimator(query)
        
        strategy = {
            'batch_size': 32,
            'use_cache': True,
            'early_stopping': False,
            'max_steps': 10
        }
        
        if complexity > 3.0:
            # Complex query needs more resources
            strategy['batch_size'] = 16
            strategy['early_stopping'] = True
            strategy['max_steps'] = 20
        elif complexity < 1.0:
            # Simple query can be processed quickly
            strategy['batch_size'] = 64
            strategy['use_cache'] = True
        
        return strategy


class OptimizedRAG:
    """Optimized RAG system with caching and resource management"""
    
    def __init__(self, 
                 query_cache_capacity: int = 500,
                 embedding_cache_capacity: int = 5000,
                 enable_caching: bool = True):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.enable_caching = enable_caching
        
        # Initialize caches
        self.query_result_cache = QueryResultCache(query_cache_capacity) if enable_caching else None
        self.embedding_cache = EmbeddingCache(embedding_cache_capacity) if enable_caching else None
        
        # Initialize optimization components
        self.embedding_optimizer = EmbeddingOptimizer()
        self.resource_optimizer = ResourceOptimizer()
        self.query_optimizer = QueryOptimizer()
        
        # Vector database
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="optimized_rag",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Track performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_queries': 0,
            'total_time': 0.0
        }
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> str:
        """Add a document with optimization"""
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}
        metadata['id'] = doc_id
        
        # Generate embedding with caching
        if self.embedding_cache:
            cached_embedding = self.embedding_cache.get(text)
            if cached_embedding is not None:
                embedding = cached_embedding
            else:
                embedding = self.embedder.encode([text])
                self.embedding_cache.put(text, embedding)
        else:
            embedding = self.embedder.encode([text])
        
        # Add to collection
        self.collection.add(
            embeddings=embedding.tolist(),
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def retrieve_optimized(self, query: str, top_k: int = 5, 
                          filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform optimized retrieval with caching and resource management"""
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Check query cache first
        if self.query_result_cache:
            cached_result = self.query_result_cache.get(query, top_k, filters)
            if cached_result is not None:
                self.metrics['cache_hits'] += 1
                retrieval_time = time.time() - start_time
                self.metrics['total_time'] += retrieval_time
                return cached_result
            else:
                self.metrics['cache_misses'] += 1
        
        # Preprocess query
        processed_query = self.query_optimizer.preprocess_query(query)
        
        # Determine processing strategy
        strategy = self.query_optimizer.determine_strategy(query)
        
        # Perform retrieval
        results = self.collection.query(
            query_texts=[processed_query],
            n_results=top_k,
            where=filters,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1.0 - results['distances'][0][i]
            })
        
        # Cache result if caching is enabled
        if self.query_result_cache:
            self.query_result_cache.put(query, top_k, filters, formatted_results)
        
        retrieval_time = time.time() - start_time
        self.metrics['total_time'] += retrieval_time
        
        return formatted_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enable_caching:
            return {'caching_disabled': True}
        
        cache_hit_rate = (
            self.metrics['cache_hits'] / 
            max(self.metrics['cache_hits'] + self.metrics['cache_misses'], 1)
        )
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'query_cache_size': self.query_result_cache.cache.size() if self.query_result_cache else 0,
            'embedding_cache_size': self.embedding_cache.cache.size() if self.embedding_cache else 0,
            'total_cache_operations': self.metrics['cache_hits'] + self.metrics['cache_misses']
        }
    
    def optimize_batch_retrieval(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """Optimize retrieval for a batch of queries"""
        # Determine the best strategy for batch processing
        batch_size = self.resource_optimizer.suggest_batch_size(len(queries))
        
        results = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = []
            
            for query in batch:
                result = self.retrieve_optimized(query, top_k)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results


class PerformanceMonitor:
    """Monitor and analyze system performance"""
    
    def __init__(self):
        self.query_times = []
        self.cache_stats_history = []
        self.resource_usage = []
    
    def record_query_time(self, query_time: float):
        """Record query processing time"""
        self.query_times.append(query_time)
        
        # Keep only last 1000 measurements
        if len(self.query_times) > 1000:
            self.query_times.pop(0)
    
    def record_cache_stats(self, cache_stats: Dict[str, Any]):
        """Record cache statistics"""
        self.cache_stats_history.append({
            'timestamp': datetime.now(),
            'stats': cache_stats
        })
        
        # Keep only last 100 measurements
        if len(self.cache_stats_history) > 100:
            self.cache_stats_history.pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.query_times:
            return {'message': 'No performance data available'}
        
        return {
            'total_queries_recorded': len(self.query_times),
            'avg_query_time': sum(self.query_times) / len(self.query_times),
            'p95_query_time': np.percentile(self.query_times, 95) if len(self.query_times) > 1 else 0,
            'min_query_time': min(self.query_times),
            'max_query_time': max(self.query_times),
            'query_time_std': np.std(self.query_times) if len(self.query_times) > 1 else 0
        }


def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques"""
    print("=== Optimization Techniques Demonstration ===\n")
    
    # Create optimized RAG system
    optimized_rag = OptimizedRAG(
        query_cache_capacity=100,
        embedding_cache_capacity=1000,
        enable_caching=True
    )
    
    # Add sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms",
        "Deep learning uses neural networks with multiple layers to process information",
        "Natural language processing helps computers understand human language",
        "Data science combines statistics, programming, and domain expertise",
        "Vector databases optimize similarity search for high-dimensional embeddings",
        "Retrieval augmented generation combines retrieval and generation approaches"
    ]
    
    doc_ids = []
    for i, doc in enumerate(documents):
        doc_id = optimized_rag.add_document(doc, metadata={"doc_id": f"doc_{i}", "category": "AI"})
        doc_ids.append(doc_id)
    
    print(f"Added {len(documents)} documents to optimized RAG system\n")
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "neural networks",
        "natural language processing",
        "data science techniques"
    ]
    
    print("Testing retrieval performance:\n")
    
    for query in test_queries:
        # First retrieval (cache miss)
        start_time = time.time()
        results1 = optimized_rag.retrieve_optimized(query, top_k=2)
        time1 = time.time() - start_time
        
        # Second retrieval (should be cached)
        start_time = time.time()
        results2 = optimized_rag.retrieve_optimized(query, top_k=2)
        time2 = time.time() - start_time
        
        print(f"Query: '{query}'")
        print(f"  First retrieval: {time1:.4f}s")
        print(f"  Second retrieval (cached): {time2:.4f}s")
        print(f"  Speedup: {time1/time2:.2f}x" if time2 > 0 else f"  Speedup: N/A (cached result)")
        print(f"  Results: {len(results1)} documents retrieved")
        print()
    
    # Show performance metrics
    metrics = optimized_rag.get_performance_metrics()
    cache_stats = optimized_rag.get_cache_stats()
    
    print("Performance Metrics:")
    print(f"  Total queries: {metrics['total_queries']}")
    print(f"  Cache hits: {metrics['cache_hits']}")
    print(f"  Cache misses: {metrics['cache_misses']}")
    print(f"  Total processing time: {metrics['total_time']:.4f}s")
    print()
    
    print("Cache Statistics:")
    print(f"  Cache hit rate: {cache_stats['cache_hit_rate']:.2%}")
    print(f"  Query cache size: {cache_stats['query_cache_size']}")
    print(f"  Embedding cache size: {cache_stats['embedding_cache_size']}")


def performance_comparison():
    """Compare performance with and without optimizations"""
    print("\n=== Performance Comparison ===\n")
    
    # Create two RAG systems - one with optimizations, one without
    optimized_rag = OptimizedRAG(enable_caching=True)
    basic_rag = OptimizedRAG(enable_caching=False)  # Disabling caching to simulate basic system
    
    # Add the same documents to both
    sample_docs = [
        "Machine learning algorithms are powerful tools for data analysis",
        "Deep neural networks have revolutionized artificial intelligence",
        "Natural language processing enables human-computer interaction",
        "Data science extracts insights from large datasets"
    ] * 5  # Repeat to have more documents
    
    for doc in sample_docs:
        optimized_rag.add_document(doc)
        basic_rag.add_document(doc)
    
    # Test queries (with some repeated queries to test caching)
    test_queries = [
        "machine learning",
        "neural networks", 
        "natural language processing",
        "data science",
        "machine learning",  # Repeated query
        "neural networks",   # Repeated query
    ]
    
    # Time optimized system
    start_time = time.time()
    for query in test_queries:
        optimized_rag.retrieve_optimized(query, top_k=3)
    optimized_time = time.time() - start_time
    
    # Time basic system
    start_time = time.time()
    for query in test_queries:
        basic_rag.retrieve_optimized(query, top_k=3)  # This won't use cache since disabled
    basic_time = time.time() - start_time
    
    print(f"Performance Comparison Results:")
    print(f"  Optimized system: {optimized_time:.4f}s for {len(test_queries)} queries")
    print(f"  Basic system: {basic_time:.4f}s for {len(test_queries)} queries")
    print(f"  Improvement: {(basic_time/optimized_time):.2f}x faster")


def main():
    """Main function to demonstrate optimization techniques"""
    print("Module 12: Optimization Techniques")
    print("=" * 50)
    
    # Demonstrate optimization techniques
    demonstrate_optimization_techniques()
    
    # Show performance comparison
    performance_comparison()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of resource optimization
    print("Resource Optimization Example:")
    resource_optimizer = ResourceOptimizer()
    
    batch_size = resource_optimizer.suggest_batch_size(input_length=100, available_memory_gb=8.0)
    print(f"Suggested batch size for 100 inputs with 8GB memory: {batch_size}")
    
    # Example of query optimization
    print("\nQuery Optimization Example:")
    query_optimizer = QueryOptimizer()
    
    complex_query = "Compare the differences between neural networks and decision trees in machine learning"
    simple_query = "What is AI?"
    
    complex_strategy = query_optimizer.determine_strategy(complex_query)
    simple_strategy = query_optimizer.determine_strategy(simple_query)
    
    print(f"Complex query strategy: {complex_strategy}")
    print(f"Simple query strategy: {simple_strategy}")
    
    print(f"\nModule 12 completed - Optimization techniques implemented and demonstrated")


if __name__ == "__main__":
    main()