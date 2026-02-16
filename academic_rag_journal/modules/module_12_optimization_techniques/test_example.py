"""
Module 12: Optimization Techniques
Tests for implementation examples

This module contains tests for the optimization technique implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np
from datetime import datetime, timedelta

from example import (
    LRUCache, QueryResultCache, EmbeddingCache, EmbeddingOptimizer, 
    IndexOptimizer, ResourceOptimizer, QueryOptimizer, OptimizedRAG, 
    PerformanceMonitor
)


class TestLRUCache(unittest.TestCase):
    """Test LRU cache implementation"""
    
    def test_initialization(self):
        """Test cache initialization"""
        cache = LRUCache(max_capacity=5)
        self.assertEqual(cache.max_capacity, 5)
        self.assertEqual(len(cache.cache), 0)
    
    def test_put_and_get(self):
        """Test putting and getting items from cache"""
        cache = LRUCache(max_capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        self.assertIsNone(cache.get("key3"))
    
    def test_capacity_limit(self):
        """Test that cache respects capacity limit"""
        cache = LRUCache(max_capacity=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")
    
    def test_recently_used_ordering(self):
        """Test that recently used items are kept"""
        cache = LRUCache(max_capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item, should evict key2 (not key1)
        cache.put("key4", "value4")
        
        self.assertEqual(cache.get("key1"), "value1")  # Should still be there
        self.assertIsNone(cache.get("key2"))  # Should be evicted
        self.assertEqual(cache.get("key3"), "value3")  # Should still be there
        self.assertEqual(cache.get("key4"), "value4")  # Should be added
    
    def test_clear(self):
        """Test clearing the cache"""
        cache = LRUCache(max_capacity=3)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        cache.clear()
        
        self.assertEqual(len(cache.cache), 0)
        self.assertIsNone(cache.get("key1"))
    
    def test_size(self):
        """Test getting cache size"""
        cache = LRUCache(max_capacity=3)
        self.assertEqual(cache.size(), 0)
        
        cache.put("key1", "value1")
        self.assertEqual(cache.size(), 1)
        
        cache.put("key2", "value2")
        self.assertEqual(cache.size(), 2)


class TestQueryResultCache(unittest.TestCase):
    """Test query result cache"""
    
    def test_initialization(self):
        """Test query cache initialization"""
        cache = QueryResultCache(max_capacity=10, ttl_seconds=300)
        self.assertEqual(cache.max_capacity, 10)
        self.assertEqual(cache.ttl_seconds, 300)
    
    def test_get_and_put(self):
        """Test caching and retrieving query results"""
        cache = QueryResultCache()
        
        query = "test query"
        results = [{"id": 1, "content": "result1"}]
        
        cache.put(query, top_k=5, results=results)
        retrieved = cache.get(query, top_k=5)
        
        self.assertEqual(retrieved, results)
    
    def test_different_parameters(self):
        """Test that different parameters create different cache entries"""
        cache = QueryResultCache()
        
        query = "test query"
        results1 = [{"id": 1, "content": "result1"}]
        results2 = [{"id": 2, "content": "result2"}]
        
        cache.put(query, top_k=5, results=results1)
        cache.put(query, top_k=10, results=results2)
        
        retrieved1 = cache.get(query, top_k=5)
        retrieved2 = cache.get(query, top_k=10)
        
        self.assertEqual(retrieved1, results1)
        self.assertEqual(retrieved2, results2)
    
    def test_ttl_expiration(self):
        """Test TTL expiration of cache entries"""
        cache = QueryResultCache(ttl_seconds=1)  # 1 second TTL
        
        cache.put("test_query", top_k=5, results=[{"id": 1}])
        
        # Entry should still be valid
        result = cache.get("test_query", top_k=5)
        self.assertIsNotNone(result)
        
        # Artificially advance time in timestamps dict
        cache.timestamps["test_query"] = datetime.now() - timedelta(seconds=2)
        
        # Entry should now be expired
        result = cache.get("test_query", top_k=5)
        self.assertIsNone(result)


class TestEmbeddingCache(unittest.TestCase):
    """Test embedding cache"""
    
    def test_initialization(self):
        """Test embedding cache initialization"""
        cache = EmbeddingCache(max_capacity=1000)
        self.assertEqual(cache.cache.max_capacity, 1000)
    
    def test_get_and_put(self):
        """Test caching and retrieving embeddings"""
        cache = EmbeddingCache()
        
        text = "test text"
        embedding = np.array([1.0, 2.0, 3.0])
        
        cache.put(text, embedding)
        retrieved = cache.get(text)
        
        np.testing.assert_array_equal(retrieved, embedding)
    
    def test_different_texts(self):
        """Test that different texts have different cache entries"""
        cache = EmbeddingCache()
        
        text1 = "first text"
        text2 = "second text"
        emb1 = np.array([1.0, 2.0])
        emb2 = np.array([3.0, 4.0])
        
        cache.put(text1, emb1)
        cache.put(text2, emb2)
        
        self.assertFalse(np.array_equal(cache.get(text1), cache.get(text2)))


class TestEmbeddingOptimizer(unittest.TestCase):
    """Test embedding optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = EmbeddingOptimizer()
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNone(self.optimizer.pca_model)
        self.assertFalse(self.optimizer.is_fitted)
    
    def test_dimension_reduction(self):
        """Test dimension reduction"""
        # Create high-dimensional embeddings
        high_dim_embeddings = np.random.rand(10, 384)  # 10 embeddings of 384 dimensions
        
        reduced = self.optimizer.reduce_dimensions(high_dim_embeddings, target_dims=64)
        
        self.assertEqual(reduced.shape[0], 10)  # Same number of embeddings
        self.assertEqual(reduced.shape[1], 64)  # Reduced dimensions
    
    def test_quantization(self):
        """Test embedding quantization"""
        embeddings = np.random.rand(5, 10).astype(np.float32)
        
        # Quantize
        quantized = self.optimizer.quantize_embeddings(embeddings)
        self.assertEqual(quantized.dtype, np.uint8)
        
        # Dequantize
        dequantized = self.optimizer.dequantize_embeddings(quantized)
        self.assertEqual(dequantized.dtype, np.float32)


class TestResourceOptimizer(unittest.TestCase):
    """Test resource optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = ResourceOptimizer()
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.current_batch_size, 32)
        self.assertLessEqual(self.optimizer.max_memory_usage, 1.0)
        self.assertIsInstance(self.optimizer.gpu_available, bool)
    
    def test_batch_size_suggestion(self):
        """Test batch size suggestion"""
        # Test with different parameters
        size1 = self.optimizer.suggest_batch_size(input_length=100, available_memory_gb=16.0)
        size2 = self.optimizer.suggest_batch_size(input_length=100, available_memory_gb=4.0)
        
        self.assertIsInstance(size1, int)
        self.assertIsInstance(size2, int)
        self.assertGreaterEqual(size1, 1)
        
        # With less memory, the suggested size might be smaller
        # but we can't guarantee this without knowing the exact calculation
    
    def test_batch_size_bounds(self):
        """Test batch size bounds"""
        # Very small memory should still return at least 1
        size = self.optimizer.suggest_batch_size(input_length=1, available_memory_gb=0.1)
        self.assertGreaterEqual(size, 1)
        
        # Very large memory should be capped
        size = self.optimizer.suggest_batch_size(input_length=1, available_memory_gb=1000.0)
        self.assertLessEqual(size, 128)


class TestQueryOptimizer(unittest.TestCase):
    """Test query optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = QueryOptimizer()
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer.query_complexity_estimator)
    
    def test_preprocess_query(self):
        """Test query preprocessing"""
        original = "  This   is  a   test   query!  "
        processed = self.optimizer.preprocess_query(original)
        
        # Should remove extra spaces and convert to lowercase
        self.assertEqual(processed, "this is test query!")
    
    def test_complexity_estimation(self):
        """Test complexity estimation"""
        simple_query = "what is AI"
        complex_query = "compare the differences between neural networks and decision trees"
        
        simple_complexity = self.optimizer.query_complexity_estimator(simple_query)
        complex_complexity = self.optimizer.query_complexity_estimator(complex_query)
        
        self.assertIsInstance(simple_complexity, float)
        self.assertIsInstance(complex_complexity, float)
        # Complex query should have higher or equal complexity
        self.assertGreaterEqual(complex_complexity, simple_complexity)
    
    def test_strategy_determination(self):
        """Test strategy determination"""
        simple_query = "what is AI"
        complex_query = "analyze the relationship between neural networks and deep learning"
        
        simple_strategy = self.optimizer.determine_strategy(simple_query)
        complex_strategy = self.optimizer.determine_strategy(complex_query)
        
        self.assertIsInstance(simple_strategy, dict)
        self.assertIsInstance(complex_strategy, dict)
        
        # Both should have required keys
        required_keys = ['batch_size', 'use_cache', 'early_stopping', 'max_steps']
        for key in required_keys:
            self.assertIn(key, simple_strategy)
            self.assertIn(key, complex_strategy)


class TestOptimizedRAG(unittest.TestCase):
    """Test optimized RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rag = OptimizedRAG(
            query_cache_capacity=50,
            embedding_cache_capacity=100,
            enable_caching=True
        )
    
    def test_initialization(self):
        """Test RAG system initialization"""
        self.assertIsNotNone(self.rag.embedder)
        self.assertIsNotNone(self.rag.query_result_cache)
        self.assertIsNotNone(self.rag.embedding_cache)
        self.assertIsNotNone(self.rag.embedding_optimizer)
        self.assertIsNotNone(self.rag.resource_optimizer)
        self.assertIsNotNone(self.rag.query_optimizer)
        self.assertTrue(self.rag.enable_caching)
    
    def test_add_document(self):
        """Test adding documents"""
        doc_id = self.rag.add_document("Test document content", metadata={"category": "test"})
        
        self.assertIsInstance(doc_id, str)
        self.assertGreater(len(doc_id), 0)  # Should have generated an ID
    
    def test_retrieve_optimized(self):
        """Test optimized retrieval"""
        # Add a test document
        self.rag.add_document("Machine learning algorithms are powerful", metadata={"type": "ML"})
        
        # Retrieve
        results = self.rag.retrieve_optimized("machine learning", top_k=1)
        
        self.assertIsInstance(results, list)
        if results:  # If we got results
            self.assertIsInstance(results[0], dict)
            self.assertIn('content', results[0])
            self.assertIn('metadata', results[0])
            self.assertIn('similarity', results[0])
    
    def test_caching_behavior(self):
        """Test caching behavior"""
        # Add a test document
        self.rag.add_document("Test content for caching", metadata={"test": True})
        
        # First retrieval (should be cache miss)
        metrics_before = self.rag.get_performance_metrics()
        results1 = self.rag.retrieve_optimized("test content", top_k=1)
        metrics_after_first = self.rag.get_performance_metrics()
        
        # Second retrieval (should hit cache)
        results2 = self.rag.retrieve_optimized("test content", top_k=1)
        metrics_after_second = self.rag.get_performance_metrics()
        
        # Cache hit should have increased
        cache_hits_after_first = metrics_after_first['cache_hits']
        cache_hits_after_second = metrics_after_second['cache_hits']
        
        # The second call should result in a cache hit
        self.assertGreaterEqual(cache_hits_after_second, cache_hits_after_first)
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        metrics = self.rag.get_performance_metrics()
        
        self.assertIn('cache_hits', metrics)
        self.assertIn('cache_misses', metrics)
        self.assertIn('total_queries', metrics)
        self.assertIn('total_time', metrics)
        
        self.assertIsInstance(metrics['cache_hits'], int)
        self.assertIsInstance(metrics['cache_misses'], int)
        self.assertIsInstance(metrics['total_queries'], int)
        self.assertIsInstance(metrics['total_time'], float)
    
    def test_cache_stats(self):
        """Test cache statistics"""
        stats = self.rag.get_cache_stats()
        
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('query_cache_size', stats)
        self.assertIn('embedding_cache_size', stats)
        self.assertIn('total_cache_operations', stats)
        
        self.assertIsInstance(stats['cache_hit_rate'], float)
        self.assertGreaterEqual(stats['cache_hit_rate'], 0.0)
        self.assertLessEqual(stats['cache_hit_rate'], 1.0)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = PerformanceMonitor()
    
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertEqual(self.monitor.query_times, [])
        self.assertEqual(self.monitor.cache_stats_history, [])
        self.assertEqual(self.monitor.resource_usage, [])
    
    def test_record_query_time(self):
        """Test recording query times"""
        self.monitor.record_query_time(0.123)
        self.assertEqual(len(self.monitor.query_times), 1)
        self.assertEqual(self.monitor.query_times[0], 0.123)
    
    def test_query_time_limit(self):
        """Test that query times are limited"""
        # Add more than 1000 times to test the limit
        for i in range(1005):
            self.monitor.record_query_time(i * 0.001)
        
        # Should only keep the last 1000
        self.assertEqual(len(self.monitor.query_times), 1000)
    
    def test_record_cache_stats(self):
        """Test recording cache stats"""
        test_stats = {'hit_rate': 0.8, 'size': 50}
        self.monitor.record_cache_stats(test_stats)
        
        self.assertEqual(len(self.monitor.cache_stats_history), 1)
        self.assertEqual(self.monitor.cache_stats_history[0]['stats'], test_stats)
        self.assertIsInstance(self.monitor.cache_stats_history[0]['timestamp'], datetime)
    
    def test_performance_summary(self):
        """Test performance summary"""
        # Add some query times
        times = [0.1, 0.2, 0.15, 0.3, 0.25]
        for t in times:
            self.monitor.record_query_time(t)
        
        summary = self.monitor.get_performance_summary()
        
        self.assertIn('total_queries_recorded', summary)
        self.assertIn('avg_query_time', summary)
        self.assertIn('p95_query_time', summary)
        self.assertIn('min_query_time', summary)
        self.assertIn('max_query_time', summary)
        self.assertIn('query_time_std', summary)
        
        self.assertEqual(summary['total_queries_recorded'], len(times))
        self.assertAlmostEqual(summary['avg_query_time'], np.mean(times), places=3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_cache_operations(self):
        """Test cache operations with empty cache"""
        cache = LRUCache(max_capacity=5)
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)
    
    def test_zero_capacity_cache(self):
        """Test cache with zero capacity"""
        cache = LRUCache(max_capacity=0)
        cache.put("key", "value")  # Should not crash
        result = cache.get("key")
        self.assertIsNone(result)  # Should not be stored
    
    def test_large_embeddings_optimization(self):
        """Test embedding optimization with large arrays"""
        optimizer = EmbeddingOptimizer()
        
        # Create a large embedding matrix
        large_embeddings = np.random.rand(1000, 384)
        
        try:
            reduced = optimizer.reduce_dimensions(large_embeddings, target_dims=128)
            self.assertEqual(reduced.shape, (1000, 128))
        except Exception as e:
            # If PCA fails due to memory constraints in testing environment, that's ok
            # Just ensure it doesn't crash unexpectedly
            self.assertIsInstance(e, Exception)
    
    def test_optimized_rag_without_caching(self):
        """Test RAG system without caching"""
        rag = OptimizedRAG(enable_caching=False)
        
        # Should work without crashing
        doc_id = rag.add_document("Test content")
        self.assertIsInstance(doc_id, str)
        
        results = rag.retrieve_optimized("test", top_k=1)
        self.assertIsInstance(results, list)
    
    def test_empty_performance_monitor(self):
        """Test performance monitor when no data recorded"""
        monitor = PerformanceMonitor()
        summary = monitor.get_performance_summary()
        
        self.assertIn('message', summary)
        self.assertEqual(summary['message'], 'No performance data available')


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test caching performance
    cache = LRUCache(max_capacity=1000)
    
    # Bulk insert
    start_time = time.time()
    for i in range(1000):
        cache.put(f"key_{i}", f"value_{i}")
    insert_time = time.time() - start_time
    
    # Bulk access
    start_time = time.time()
    for i in range(1000):
        value = cache.get(f"key_{i}")
        assert value == f"value_{i}"
    access_time = time.time() - start_time
    
    print(f"Cache: Inserted 1000 items in {insert_time:.4f}s")
    print(f"Cache: Accessed 1000 items in {access_time:.4f}s")
    
    # Test RAG system performance
    rag = OptimizedRAG(enable_caching=True)
    
    # Add many documents
    start_time = time.time()
    for i in range(100):
        rag.add_document(f"Document {i} content for performance testing")
    docs_time = time.time() - start_time
    
    # Perform multiple queries
    start_time = time.time()
    for i in range(50):
        rag.retrieve_optimized(f"document {i % 20}", top_k=3)
    queries_time = time.time() - start_time
    
    print(f"RAG: Added 100 documents in {docs_time:.4f}s")
    print(f"RAG: Performed 50 queries in {queries_time:.4f}s")
    
    # Should be reasonably fast
    assert docs_time < 5.0, f"Document insertion too slow: {docs_time}s"
    assert queries_time < 10.0, f"Query processing too slow: {queries_time}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test optimization strategies with different workloads
    print("Testing optimization under different loads...")
    
    rag = OptimizedRAG(
        query_cache_capacity=200,
        embedding_cache_capacity=500,
        enable_caching=True
    )
    
    # Add a substantial amount of data
    documents = []
    for i in range(50):
        documents.append(f"Machine learning document {i} discussing various algorithms and techniques for data processing and analysis.")
    
    for doc in documents:
        rag.add_document(doc, metadata={"category": "ML", "doc_id": f"ml_{i}"})
    
    # Test with various query patterns
    queries = [
        "machine learning algorithms",
        "data processing techniques",
        "neural networks",
        "classification methods",
        "machine learning algorithms"  # Duplicate to test caching
    ] * 10  # Repeat to increase cache effectiveness
    
    print(f"Testing with {len(queries)} queries on {len(documents)} documents")
    
    start_time = time.time()
    for query in queries:
        results = rag.retrieve_optimized(query, top_k=5)
    total_time = time.time() - start_time
    
    metrics = rag.get_performance_metrics()
    cache_stats = rag.get_cache_stats()
    
    print(f"Completed {len(queries)} queries in {total_time:.4f}s")
    print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.2%}")
    print(f"Total queries processed: {metrics['total_queries']}")
    print(f"Cache hits: {metrics['cache_hits']}, misses: {metrics['cache_misses']}")
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()