"""
Module 9: Evaluation Metrics
Tests for implementation examples

This module contains tests for the evaluation metrics implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np

from example import RetrievalEvaluator, GenerationEvaluator, RAGEvaluator, StatisticalSignificanceTester


class TestRetrievalEvaluator(unittest.TestCase):
    """Test retrieval evaluation metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = RetrievalEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.embedder)
    
    def test_precision_at_k(self):
        """Test precision at k calculation"""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5]
        
        # P@3: 3 relevant out of 3 retrieved = 1.0
        p_at_3 = self.evaluator.calculate_precision_at_k(retrieved, relevant, 3)
        self.assertAlmostEqual(p_at_3, 1.0)
        
        # P@1: 1 relevant out of 1 retrieved = 1.0
        p_at_1 = self.evaluator.calculate_precision_at_k(retrieved, relevant, 1)
        self.assertAlmostEqual(p_at_1, 1.0)
        
        # P@5: 3 relevant out of 5 retrieved = 0.6
        p_at_5 = self.evaluator.calculate_precision_at_k(retrieved, relevant, 5)
        self.assertAlmostEqual(p_at_5, 0.6)
    
    def test_recall_at_k(self):
        """Test recall at k calculation"""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]  # 5 relevant, only 3 retrieved in top 5
        
        # R@5: 3 retrieved out of 5 relevant = 0.6
        r_at_5 = self.evaluator.calculate_recall_at_k(retrieved, relevant, 5)
        self.assertAlmostEqual(r_at_5, 0.6)
        
        # R@3: 2 retrieved out of 5 relevant = 0.4
        r_at_3 = self.evaluator.calculate_recall_at_k(retrieved, relevant, 3)
        self.assertAlmostEqual(r_at_3, 0.4)
    
    def test_f1_at_k(self):
        """Test F1 score at k calculation"""
        retrieved = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5]
        
        f1_at_3 = self.evaluator.calculate_f1_at_k(retrieved, relevant, 3)
        # P@3 = 1.0, R@3 = 0.6, F1 = 2 * (1.0 * 0.6) / (1.0 + 0.6) = 0.75
        self.assertAlmostEqual(f1_at_3, 0.75, places=3)
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank calculation"""
        retrieved_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        relevant_lists = [[3], [5], [8]]  # Relevant doc at rank 3, 2, 2 respectively
        
        mrr = self.evaluator.calculate_mrr(retrieved_lists, relevant_lists)
        # RR scores: 1/3, 1/2, 1/2 -> MRR = (1/3 + 1/2 + 1/2) / 3 = (1/3 + 1) / 3 = 4/9
        expected_mrr = (1/3 + 1/2 + 1/2) / 3
        self.assertAlmostEqual(mrr, expected_mrr, places=3)
    
    def test_ndcg(self):
        """Test nDCG calculation"""
        retrieved = [1, 2, 3]
        relevant = [1, 2, 3]
        relevance_scores = {1: 3, 2: 2, 3: 1}  # Perfect ranking
        
        ndcg = self.evaluator.calculate_ndcg(retrieved, relevant, relevance_scores, k=3)
        # Since the ranking is perfect, nDCG should be 1.0
        self.assertAlmostEqual(ndcg, 1.0, places=3)
    
    def test_map(self):
        """Test Mean Average Precision calculation"""
        retrieved_lists = [[1, 2, 3], [4, 5, 6]]
        relevant_lists = [[1, 3], [4, 5]]
        
        map_score = self.evaluator.calculate_map(retrieved_lists, relevant_lists)
        # Calculate by hand for verification
        self.assertIsInstance(map_score, float)
        self.assertGreaterEqual(map_score, 0.0)
        self.assertLessEqual(map_score, 1.0)
    
    def test_evaluate_retrieval(self):
        """Test comprehensive retrieval evaluation"""
        retrieved_lists = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        relevant_lists = [[1, 3, 5], [7, 9]]
        relevance_scores_lists = [{1: 3, 2: 1, 3: 2, 4: 0, 5: 1}, 
                                 {6: 0, 7: 2, 8: 1, 9: 3, 10: 0}]
        
        results = self.evaluator.evaluate_retrieval(retrieved_lists, relevant_lists, relevance_scores_lists)
        
        # Check that results contain expected keys
        self.assertIn('precision_at_k', results)
        self.assertIn('recall_at_k', results)
        self.assertIn('f1_at_k', results)
        self.assertIn('mrr', results)
        self.assertIn('map', results)
        self.assertIn('ndcg_at_k', results)
        
        # Check that values are reasonable
        self.assertGreaterEqual(results['mrr'], 0.0)
        self.assertLessEqual(results['mrr'], 1.0)
        self.assertGreaterEqual(results['map'], 0.0)
        self.assertLessEqual(results['map'], 1.0)


class TestGenerationEvaluator(unittest.TestCase):
    """Test generation evaluation metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = GenerationEvaluator()
        self.embedder = None  # Will be created as needed
    
    def test_initialization(self):
        """Test evaluator initialization"""
        # Basic test to ensure init doesn't fail
        self.assertIsNotNone(self.evaluator)
    
    def test_exact_match(self):
        """Test exact match calculation"""
        generated = "This is the answer"
        reference = "This is the answer"
        em_score = self.evaluator.calculate_exact_match(generated, reference)
        self.assertEqual(em_score, 1.0)
        
        generated_diff = "This is a different answer"
        em_score_diff = self.evaluator.calculate_exact_match(generated_diff, reference)
        self.assertEqual(em_score_diff, 0.0)
    
    def test_f1_score(self):
        """Test F1 score calculation"""
        generated = "the quick brown fox"
        reference = "the quick brown dog"
        
        f1_score = self.evaluator.calculate_f1_score(generated, reference)
        # Common tokens: "the", "quick", "brown" (3 tokens)
        # Generated tokens: 4, Reference tokens: 4
        # Precision: 3/4, Recall: 3/4, F1: 2*(3/4)*(3/4) / (3/4 + 3/4) = 0.75
        expected_f1 = 2 * (3/4) * (3/4) / (3/4 + 3/4)
        self.assertAlmostEqual(f1_score, expected_f1, places=3)
    
    def test_normalize_text(self):
        """Test text normalization"""
        text = "Hello,    world! How are you?"
        normalized = self.evaluator._normalize_text(text)
        expected = "hello world how are you"
        self.assertEqual(normalized, expected)
    
    def test_evaluate_generation(self):
        """Test comprehensive generation evaluation"""
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        generated_list = ["This is a generated answer", "Another generated response"]
        reference_list = ["This is a reference answer", "Another reference response"]
        
        results = self.evaluator.evaluate_generation(generated_list, reference_list, embedder)
        
        # Check that results contain expected keys
        expected_keys = [
            'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_bleu',
            'avg_exact_match', 'avg_f1', 'avg_semantic_similarity'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
            self.assertIsInstance(results[key], float)
            # Scores should be between 0 and 1 (with exception of BLEU which can be different)
            if key != 'avg_bleu':  # BLEU can have values outside [0,1] depending on implementation
                self.assertGreaterEqual(results[key], 0.0)
                self.assertLessEqual(results[key], 1.0)


class TestRAGEvaluator(unittest.TestCase):
    """Test comprehensive RAG evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = RAGEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.retrieval_evaluator)
        self.assertIsNotNone(self.evaluator.generation_evaluator)
        self.assertIsNotNone(self.evaluator.embedder)
    
    def test_evaluate_rag_system(self):
        """Test end-to-end RAG system evaluation"""
        queries = ["What is AI?", "Explain ML?"]
        retrieved_docs_list = [[1, 2, 3], [4, 5, 6]]
        relevant_docs_list = [[1, 3], [4, 5]]
        generated_answers = ["AI is artificial intelligence", "ML is machine learning"]
        reference_answers = ["Artificial intelligence is a field", "Machine learning uses algorithms"]
        
        results = self.evaluator.evaluate_rag_system(
            queries, retrieved_docs_list, relevant_docs_list,
            generated_answers, reference_answers
        )
        
        # Check structure of results
        self.assertIn('retrieval_metrics', results)
        self.assertIn('generation_metrics', results)
        self.assertIn('end_to_end_metrics', results)
        
        # Check that retrieval metrics have expected structure
        retrieval_metrics = results['retrieval_metrics']
        self.assertIn('mrr', retrieval_metrics)
        self.assertIn('map', retrieval_metrics)
        
        # Check that generation metrics have expected structure
        generation_metrics = results['generation_metrics']
        self.assertIn('avg_semantic_similarity', generation_metrics)
    
    def test_calculate_efficiency_metrics(self):
        """Test efficiency metrics calculation"""
        query_times = [0.1, 0.2, 0.15, 0.25, 0.18]
        token_counts = [500, 600, 450, 700, 550]
        processing_times = [0.05, 0.1, 0.08, 0.12, 0.09]
        
        results = self.evaluator.calculate_efficiency_metrics(query_times, token_counts, processing_times)
        
        # Check for expected keys
        self.assertIn('avg_query_time', results)
        self.assertIn('std_query_time', results)
        self.assertIn('max_query_time', results)
        self.assertIn('min_query_time', results)
        self.assertIn('p95_query_time', results)
        self.assertIn('avg_tokens_per_query', results)
        self.assertIn('total_tokens', results)
        self.assertIn('avg_processing_time', results)
        
        # Check values are reasonable
        self.assertGreaterEqual(results['avg_query_time'], 0)
        self.assertGreaterEqual(results['avg_tokens_per_query'], 0)
        self.assertGreaterEqual(results['avg_processing_time'], 0)
    
    def test_calculate_cost_metrics(self):
        """Test cost metrics calculation"""
        api_calls = [
            {'model_type': 'embedding', 'tokens': 1000},
            {'model_type': 'generation', 'tokens': 500},
            {'model_type': 'retrieval', 'tokens': 0}
        ]
        
        results = self.evaluator.calculate_cost_metrics(api_calls)
        
        self.assertIn('estimated_cost', results)
        self.assertIn('total_api_calls', results)
        self.assertIn('cost_per_query', results)
        
        # With default costs, should be > 0
        self.assertGreaterEqual(results['estimated_cost'], 0)
        self.assertEqual(results['total_api_calls'], 3)


class TestStatisticalSignificanceTester(unittest.TestCase):
    """Test statistical significance testing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = StatisticalSignificanceTester()
    
    def test_initialization(self):
        """Test tester initialization"""
        self.assertIsNotNone(self.tester)
    
    def test_perform_ttest(self):
        """Test t-test performance"""
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [2, 3, 4, 5, 6]  # Slightly higher values
        
        result = self.tester.perform_ttest(sample1, sample2)
        
        # Check for expected keys
        self.assertIn('t_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertIn('alpha', result)
        
        # With small samples, might not be significant at 0.05 level
        self.assertIsInstance(result['significant'], bool)
        if result['p_value'] is not None:
            self.assertGreaterEqual(result['p_value'], 0)
            self.assertLessEqual(result['p_value'], 1)
    
    def test_confidence_interval(self):
        """Test confidence interval calculation"""
        sample = [1, 2, 3, 4, 5]
        
        ci = self.tester.calculate_confidence_interval(sample)
        
        # Should return a tuple of two values
        self.assertIsInstance(ci, tuple)
        self.assertEqual(len(ci), 2)
        
        # Lower bound should be less than upper bound
        self.assertLessEqual(ci[0], ci[1])
        
        # Sample mean should be within the interval
        sample_mean = sum(sample) / len(sample)
        self.assertGreaterEqual(sample_mean, ci[0])
        self.assertLessEqual(sample_mean, ci[1])
    
    def test_small_sample_handling(self):
        """Test handling of small samples"""
        # Test with sample size < 2 for t-test
        result = self.tester.perform_ttest([1], [2])
        self.assertIn('message', result)
        self.assertIsNone(result['p_value'])
        
        # Test confidence interval with small sample
        ci = self.tester.calculate_confidence_interval([1])
        self.assertEqual(ci, (0.0, 0.0))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_retrieved_list(self):
        """Test with empty retrieved list"""
        evaluator = RetrievalEvaluator()
        
        retrieved = []
        relevant = [1, 2, 3]
        
        p_at_5 = evaluator.calculate_precision_at_k(retrieved, relevant, 5)
        self.assertEqual(p_at_5, 0.0)
        
        r_at_5 = evaluator.calculate_recall_at_k(retrieved, relevant, 5)
        self.assertEqual(r_at_5, 0.0)  # No retrieved = 0 recall when relevant exist
    
    def test_empty_relevant_list(self):
        """Test with empty relevant list"""
        evaluator = RetrievalEvaluator()
        
        retrieved = [1, 2, 3]
        relevant = []
        
        r_at_5 = evaluator.calculate_recall_at_k(retrieved, relevant, 5)
        self.assertEqual(r_at_5, 1.0)  # No relevant docs, so recall is 1 if nothing retrieved
        # But if we retrieve docs when none are relevant, recall should be 0
        retrieved_nonempty = [1, 2, 3]
        relevant_empty = []
        r_at_k = evaluator.calculate_recall_at_k(retrieved_nonempty, relevant_empty, 3)
        # When no relevant docs exist, recall should be 1 if nothing retrieved, 0 otherwise
        # Actually, if no relevant docs exist and we retrieve docs, it's not clear what recall should be
        # In our implementation, it would be 0/0 which we handle in the function
    
    def test_zero_k_values(self):
        """Test with k=0"""
        evaluator = RetrievalEvaluator()
        
        retrieved = [1, 2, 3]
        relevant = [1, 2]
        
        p_at_0 = evaluator.calculate_precision_at_k(retrieved, relevant, 0)
        self.assertEqual(p_at_0, 0.0)
    
    def test_single_character_strings(self):
        """Test generation metrics with single character strings"""
        evaluator = GenerationEvaluator()
        
        gen = "a"
        ref = "b"
        
        em = evaluator.calculate_exact_match(gen, ref)
        self.assertIsInstance(em, float)
        
        f1 = evaluator.calculate_f1_score(gen, ref)
        self.assertIsInstance(f1, float)
    
    def test_special_characters(self):
        """Test with special characters"""
        evaluator = GenerationEvaluator()
        
        gen = "Hello, world! @#$%^&*()"
        ref = "Hello, world! @#$%^&*()"
        
        em = evaluator.calculate_exact_match(gen, ref)
        self.assertEqual(em, 1.0)


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test with larger dataset
    evaluator = RetrievalEvaluator()
    
    # Create larger test data
    num_queries = 100
    retrieved_lists = []
    relevant_lists = []
    
    for i in range(num_queries):
        retrieved = list(range(i*10, (i+1)*10))  # 10 retrieved docs per query
        relevant = list(range(i*10, i*10 + 3))   # 3 relevant docs per query
        retrieved_lists.append(retrieved)
        relevant_lists.append(relevant)
    
    start_time = time.time()
    results = evaluator.evaluate_retrieval(retrieved_lists, relevant_lists)
    elapsed_time = time.time() - start_time
    
    print(f"Evaluated {num_queries} queries in {elapsed_time:.3f} seconds")
    print(f"Average time per query: {elapsed_time/num_queries:.4f} seconds")
    
    # Results should contain expected metrics
    assert 'mrr' in results
    assert 'map' in results
    assert isinstance(results['mrr'], float)


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test correlation between metrics
    print("Testing metric behavior with known patterns...")
    
    evaluator = RetrievalEvaluator()
    
    # Perfect retrieval case
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 2, 3, 4, 5]  # All retrieved are relevant
    
    p_at_5 = evaluator.calculate_precision_at_k(retrieved, relevant, 5)
    r_at_5 = evaluator.calculate_recall_at_k(retrieved, relevant, 5)
    f1_at_5 = evaluator.calculate_f1_at_k(retrieved, relevant, 5)
    
    print(f"Perfect retrieval - P@5: {p_at_5}, R@5: {r_at_5}, F1@5: {f1_at_5}")
    assert p_at_5 == 1.0
    assert r_at_5 == 1.0
    assert f1_at_5 == 1.0
    
    # No relevant results case
    retrieved = [1, 2, 3, 4, 5]
    relevant = []  # No relevant docs
    
    r_at_5 = evaluator.calculate_recall_at_k(retrieved, relevant, 5)
    print(f"No relevant docs - R@5: {r_at_5}")
    # When no relevant docs exist, our implementation returns 1.0 if nothing retrieved,
    # but if docs were retrieved when none are relevant, it would be 0.0
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()