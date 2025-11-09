"""
Module 5: Hybrid Search
Tests for implementation examples

This module contains tests for the hybrid search implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np

from example import HybridSearch, BM25Search, VectorSearch, HybridSearchEvaluator


class TestBM25Search(unittest.TestCase):
    """Test BM25 search implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_docs = [
            "Machine learning algorithms can improve with more data",
            "Deep learning is a subset of machine learning",
            "Natural language processing helps computers understand text",
            "Python is a popular programming language for AI"
        ]
        self.bm25_search = BM25Search(self.sample_docs)
    
    def test_initialization(self):
        """Test BM25 search initialization"""
        self.assertIsNotNone(self.bm25_search.bm25)
        self.assertEqual(len(self.bm25_search.documents), 4)
    
    def test_search_basic(self):
        """Test basic BM25 search functionality"""
        results = self.bm25_search.search("machine learning", k=3)
        
        # Should return at least one result
        self.assertGreaterEqual(len(results), 1)
        
        # Check result format
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('score', result)
            self.assertIn('rank', result)
            self.assertIsInstance(result['score'], float)
            self.assertIsInstance(result['rank'], int)
    
    def test_search_no_matches(self):
        """Test search with query that has no matches"""
        results = self.bm25_search.search("xyz123", k=3)
        
        # Should return empty list or results with score 0
        for result in results:
            self.assertEqual(result['score'], 0.0)


class TestVectorSearch(unittest.TestCase):
    """Test vector search implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.sample_docs = [
            "Machine learning algorithms can improve with more data",
            "Deep learning is a subset of machine learning",
            "Natural language processing helps computers understand text",
            "Python is a popular programming language for AI"
        ]
        self.vector_search = VectorSearch(self.sample_docs, self.embedder)
    
    def test_initialization(self):
        """Test vector search initialization"""
        self.assertIsNotNone(self.vector_search.index)
        self.assertEqual(len(self.vector_search.documents), 4)
    
    def test_search_basic(self):
        """Test basic vector search functionality"""
        results = self.vector_search.search("artificial intelligence", k=3)
        
        # Should return results
        self.assertGreaterEqual(len(results), 1)
        
        # Check result format
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('score', result)
            self.assertIn('rank', result)
            self.assertIsInstance(result['score'], float)
            self.assertIsInstance(result['rank'], int)
            # Cosine similarity should be between -1 and 1
            self.assertGreaterEqual(result['score'], -1.0)
            self.assertLessEqual(result['score'], 1.0)


class TestHybridSearch(unittest.TestCase):
    """Test hybrid search implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_docs = [
            "Machine learning algorithms can improve with more data",
            "Deep learning is a subset of machine learning",
            "Natural language processing helps computers understand text",
            "Python is a popular programming language for AI",
            "Vector embeddings represent text in high-dimensional space",
            "Semantic search finds meaning rather than keywords"
        ]
        self.hybrid_search = HybridSearch()
        self.hybrid_search.add_documents(self.sample_docs)
    
    def test_initialization(self):
        """Test hybrid search initialization"""
        self.assertIsNotNone(self.hybrid_search.bm25_search)
        self.assertIsNotNone(self.hybrid_search.vector_search)
        self.assertEqual(len(self.hybrid_search.documents), 6)
    
    def test_add_documents(self):
        """Test adding documents to hybrid search"""
        new_docs = ["New document 1", "New document 2"]
        ids = self.hybrid_search.add_documents(new_docs)
        
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(self.hybrid_search.documents), 8)  # 6 + 2
    
    def test_search_bm25(self):
        """Test BM25 search through hybrid interface"""
        results = self.hybrid_search.search_bm25("machine learning", k=3)
        
        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('score', result)
            self.assertIn('rank', result)
    
    def test_search_vectors(self):
        """Test vector search through hybrid interface"""
        results = self.hybrid_search.search_vectors("artificial intelligence", k=3)
        
        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('score', result)
            self.assertIn('rank', result)
    
    def test_hybrid_search_rrf(self):
        """Test hybrid search using RRF method"""
        results = self.hybrid_search.hybrid_search_rrf("AI and learning", k=4)
        
        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('rrf_score', result)
            self.assertIn('bm25_score', result)
            self.assertIn('vector_score', result)
            self.assertIn('rank', result)
    
    def test_hybrid_search_linear(self):
        """Test hybrid search using linear combination method"""
        results = self.hybrid_search.hybrid_search_linear("AI and learning", k=4)
        
        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('combined_score', result)
            self.assertIn('bm25_score', result)
            self.assertIn('vector_score', result)
            self.assertIn('rank', result)
    
    def test_search_default_method(self):
        """Test default search method (RRF)"""
        results = self.hybrid_search.search("AI", k=3)
        
        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('rrf_score', result)  # RRF is default
    
    def test_search_invalid_method(self):
        """Test search with invalid method"""
        with self.assertRaises(ValueError):
            self.hybrid_search.search("test", method="invalid_method")


class TestHybridSearchEvaluator(unittest.TestCase):
    """Test hybrid search evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = HybridSearchEvaluator()
        self.hybrid_search = HybridSearch()
        self.hybrid_search.add_documents(self.evaluator.sample_docs)
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.sample_docs)
        self.assertIsNotNone(self.evaluator.sample_queries)
        self.assertEqual(len(self.evaluator.sample_docs), 10)
        self.assertEqual(len(self.evaluator.sample_queries), 4)
    
    def test_evaluate_method(self):
        """Test method evaluation"""
        result = self.evaluator.evaluate_method(self.hybrid_search, 'rrf', k=3)
        
        self.assertIn('method', result)
        self.assertIn('avg_query_time', result)
        self.assertIn('total_time', result)
        self.assertIn('num_results', result)
        self.assertEqual(result['method'], 'rrf')
        self.assertGreaterEqual(result['num_results'], 1)
    
    def test_evaluate_different_methods(self):
        """Test evaluation of different methods"""
        results = self.evaluator.evaluate_method(self.hybrid_search, 'linear', k=3)
        
        self.assertIn('method', results)
        self.assertIn('avg_query_time', results)
        self.assertIn('total_time', results)
        self.assertIn('num_results', results)
        self.assertEqual(results['method'], 'linear')
    
    def test_compare_methods(self):
        """Test comparison of different methods"""
        results = self.evaluator.compare_methods(self.hybrid_search)
        
        # Should compare at least RRF and linear methods
        self.assertGreaterEqual(len(results), 2)
        
        methods_tested = [r['method'] for r in results]
        self.assertIn('rrf', methods_tested)
        self.assertIn('linear', methods_tested)
        
        for result in results:
            self.assertGreaterEqual(result['num_results'], 1)
            self.assertGreaterEqual(result['avg_query_time'], 0)
            self.assertGreaterEqual(result['total_time'], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.empty_search = HybridSearch()
    
    def test_search_empty_index(self):
        """Test searching with empty index"""
        # Adding empty search should still work
        results = self.empty_search.search("test query", k=5)
        # This should not fail, but might return empty results
        # depending on how add_documents was called (if at all)
        
    def test_single_document(self):
        """Test with single document"""
        single_doc_search = HybridSearch()
        single_doc_search.add_documents(["Single document for testing"])
        
        results = single_doc_search.search("test", k=5)
        # Should return at least one result
        
    def test_long_documents(self):
        """Test with very long documents"""
        long_doc = "This is a very long document. " * 100
        long_search = HybridSearch()
        long_search.add_documents([long_doc])
        
        results = long_search.search("long document", k=1)
        self.assertEqual(len(results), 1)
    
    def test_special_characters(self):
        """Test with special characters"""
        special_search = HybridSearch()
        special_search.add_documents(["Document with special chars: !@#$%^&*()"])
        
        results = special_search.search("special chars", k=1)
        # Should not fail with special characters
        

def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    # Create larger dataset
    large_docs = [
        "Machine learning is a subset of artificial intelligence. " * 5,
        "Deep learning uses neural networks with multiple layers. " * 5,
        "Natural language processing enables computers to understand text. " * 5,
        "Computer vision allows machines to interpret visual information. " * 5,
        "Data science combines statistics, programming, and domain expertise. " * 5,
        "Reinforcement learning uses rewards to train agents. " * 5,
        "Supervised learning uses labeled training data. " * 5,
        "Unsupervised learning finds patterns in unlabeled data. " * 5,
    ] * 5  # Multiply to get 40 documents
    
    hybrid_search = HybridSearch()
    start_time = __import__('time').time()
    hybrid_search.add_documents(large_docs)
    add_time = __import__('time').time() - start_time
    
    # Test search performance
    query_times = []
    for _ in range(5):
        start = __import__('time').time()
        hybrid_search.search("machine learning", k=3)
        query_times.append(__import__('time').time() - start)
    
    avg_query_time = sum(query_times) / len(query_times)
    
    print(f"Performance results:")
    print(f"  Added {len(large_docs)} docs in {add_time:.3f}s")
    print(f"  Avg query time: {avg_query_time:.3f}s")
    
    # Verify results make sense
    assert len(hybrid_search.documents) == len(large_docs), f"Expected {len(large_docs)} docs, got {len(hybrid_search.documents)}"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test different weight combinations
    print("Testing different weight combinations...")
    
    docs = [
        "Python programming and data science",
        "Machine learning with Python",
        "Web development using Python",
        "Artificial intelligence and neural networks"
    ]
    
    hybrid = HybridSearch()
    hybrid.add_documents(docs)
    
    query = "Python for data science"
    
    # Test different combinations
    combinations = [
        (0.1, 0.9),  # Mostly semantic
        (0.5, 0.5),  # Equal weight
        (0.9, 0.1),  # Mostly keyword
    ]
    
    for kw_weight, sem_weight in combinations:
        results = hybrid.hybrid_search_linear(
            query, k=3, 
            keyword_weight=kw_weight, 
            semantic_weight=sem_weight
        )
        print(f"Keyword weight: {kw_weight}, Semantic weight: {sem_weight}")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['combined_score']:.3f}")
    
    print("Weight combinations tested successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()