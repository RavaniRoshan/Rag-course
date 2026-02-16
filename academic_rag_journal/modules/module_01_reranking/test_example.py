"""
Test file for Module 1: Reranking Implementation
This file contains unit tests to validate the reranking functionality.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the module directory to the path so we can import the example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from example import AdvancedReranker, OptimizedReranker, RerankingEvaluator, CustomDomainReranker


class TestRerankingModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.reranker = AdvancedReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.sample_candidates = [
            {"id": "1", "text": "This is a relevant document", "original_rank": 0},
            {"id": "2", "text": "This is another relevant document", "original_rank": 1},
            {"id": "3", "text": "This document is not relevant", "original_rank": 2}
        ]
    
    def test_rerank_with_confidence(self):
        """Test the reranking with confidence scoring functionality"""
        query = "relevant documents"
        results = self.reranker.rerank_with_confidence(query, self.sample_candidates, top_k=3)
        
        # Check that we get results
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)
        
        # Check that results have confidence scores
        for result in results:
            self.assertIn("relevance_score", result)
            self.assertIn("confidence", result)
            self.assertIn("reranked_position", result)
    
    def test_batch_rerank(self):
        """Test batch reranking functionality"""
        queries = ["query 1", "query 2"]
        candidates_list = [self.sample_candidates, self.sample_candidates]
        
        results = self.reranker.batch_rerank(queries, candidates_list, batch_size=2)
        
        # Check that we have results for both queries
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, list)
    
    def test_score_threshold_filtering(self):
        """Test that score threshold filtering works correctly"""
        query = "test query"
        results = self.reranker.rerank_with_confidence(
            query, 
            self.sample_candidates, 
            top_k=10, 
            score_threshold=0.0  # Low threshold to ensure we get results
        )
        
        # All results should meet the threshold
        for result in results:
            self.assertGreaterEqual(result["relevance_score"], 0.0)
    
    def test_optimized_reranker(self):
        """Test optimized reranker functionality"""
        opt_reranker = OptimizedReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        query = "test query"
        results = opt_reranker.optimized_rerank(query, self.sample_candidates)
        
        # Should return a list of results
        self.assertIsInstance(results, list)
    
    def test_custom_domain_reranker(self):
        """Test custom domain reranker functionality"""
        domain_keywords = ["relevant", "document"]
        domain_reranker = CustomDomainReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            domain_keywords=domain_keywords
        )
        
        query = "relevant documents"
        results = domain_reranker.rerank_with_domain_boost(query, self.sample_candidates)
        
        # Should return a list of results
        self.assertIsInstance(results, list)
        
        # Check for domain boost scores
        for result in results:
            self.assertIn("domain_boost_score", result)
            self.assertIn("enhanced_relevance_score", result)


class TestRerankingEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for evaluator tests."""
        self.test_dataset = [
            {
                'query': 'test query',
                'candidates': [
                    {'id': '1', 'text': 'relevant text'},
                    {'id': '2', 'text': 'irrelevant text'}
                ],
                'relevant_docs': ['1']
            }
        ]
        self.evaluator = RerankingEvaluator(self.test_dataset)
        self.reranker = AdvancedReranker()
    
    def test_evaluate_reranker(self):
        """Test the evaluator functionality"""
        results = self.evaluator.evaluate_reranker(self.reranker)
        
        # Check that we get the expected metrics
        expected_metrics = ['precision @k', 'recall @k', 'ndcg @k', 'mrr', 'latency']
        for metric in expected_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], dict)
            self.assertIn('mean', results[metric])
    
    def test_metric_calculations(self):
        """Test individual metric calculations"""
        results = [
            {'id': '1'},
            {'id': '2'},
            {'id': '3'}
        ]
        relevant_docs = {'1'}
        
        # Test precision calculation
        precision = self.evaluator._calculate_precision(results, relevant_docs, k=1)
        self.assertEqual(precision, 1.0)  # First result is relevant
        
        # Test recall calculation
        recall = self.evaluator._calculate_recall(results, relevant_docs, k=3)
        self.assertEqual(recall, 1.0)  # One relevant doc exists and is retrieved
        
        # Test MRR calculation
        mrr = self.evaluator._calculate_mrr(results, relevant_docs)
        self.assertEqual(mrr, 1.0)  # First result is relevant


def run_tests():
    """Run all tests in the module"""
    print("Running tests for Module 1: Reranking")
    print("=" * 40)
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAll tests passed! ✅")
    else:
        print("\nSome tests failed! ❌")
        sys.exit(1)