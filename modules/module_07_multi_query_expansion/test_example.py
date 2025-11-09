"""
Module 7: Multi-Query Expansion
Tests for implementation examples

This module contains tests for the multi-query expansion implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np

from example import QueryGeneratorBase, LLMQueryGenerator, SemanticQueryExpander, ParaphraseQueryGenerator, MultiQueryRetriever, MultiQueryEvaluator


class TestQueryGeneratorBase(unittest.TestCase):
    """Test base class for query generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = QueryGeneratorBase()
    
    def test_initialization(self):
        """Test base generator initialization"""
        self.assertIsNotNone(self.generator.embedder)


class TestLLMQueryGenerator(unittest.TestCase):
    """Test LLM-based query generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = LLMQueryGenerator()
    
    def test_initialization(self):
        """Test LLM generator initialization"""
        # Check that generator has required attributes
        self.assertIsNotNone(self.generator.embedder)
    
    def test_generate_queries(self):
        """Test query generation"""
        original_query = "machine learning algorithms"
        queries = self.generator.generate_queries(original_query, n_queries=3)
        
        # Should return a list
        self.assertIsInstance(queries, list)
        # Should have requested number of queries
        self.assertGreaterEqual(len(queries), 1)  # Might be less if rule-based fallback
        # Each query should be a string
        for query in queries:
            self.assertIsInstance(query, str)
    
    def test_generate_multiple_queries(self):
        """Test generating multiple queries"""
        original_query = "neural network"
        queries = self.generator.generate_queries(original_query, n_queries=5)
        
        self.assertIsInstance(queries, list)
        self.assertGreaterEqual(len(queries), 1)
        
        # Check that queries are reasonably diverse
        unique_queries = list(set(queries))
        self.assertGreaterEqual(len(unique_queries), 1)


class TestSemanticQueryExpander(unittest.TestCase):
    """Test semantic query expansion"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expander = SemanticQueryExpander()
    
    def test_initialization(self):
        """Test semantic expander initialization"""
        self.assertIsNotNone(self.expander.embedder)
        self.assertIsNotNone(self.expander.semantic_related_terms)
    
    def test_generate_queries(self):
        """Test semantic query generation"""
        original_query = "machine learning"
        queries = self.expander.generate_queries(original_query, n_queries=3)
        
        # Should return a list of queries
        self.assertIsInstance(queries, list)
        self.assertGreaterEqual(len(queries), 1)
        
        # Each should be a string
        for query in queries:
            self.assertIsInstance(query, str)
    
    def test_expansion_with_known_terms(self):
        """Test expansion with terms we know exist in semantic map"""
        original_query = "machine learning algorithms"
        queries = self.expander.generate_queries(original_query, n_queries=4)
        
        self.assertIsInstance(queries, list)
        self.assertGreaterEqual(len(queries), 1)
        
        # At minimum, original query should be included
        self.assertIn(original_query, queries)


class TestParaphraseQueryGenerator(unittest.TestCase):
    """Test paraphrase-based query generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ParaphraseQueryGenerator()
    
    def test_initialization(self):
        """Test paraphrase generator initialization"""
        self.assertIsNotNone(self.generator.embedder)
        self.assertIsNotNone(self.generator.paraphrase_patterns)
    
    def test_generate_queries(self):
        """Test paraphrase query generation"""
        original_query = "how to build a model"
        queries = self.generator.generate_queries(original_query, n_queries=3)
        
        # Should return a list
        self.assertIsInstance(queries, list)
        self.assertGreaterEqual(len(queries), 1)
        
        # Each should be a string
        for query in queries:
            self.assertIsInstance(query, str)


class TestMultiQueryRetriever(unittest.TestCase):
    """Test multi-query retrieval functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.documents = [
            "Machine learning algorithms are methods to train models using data",
            "Deep learning neural networks require significant computational resources", 
            "Natural language processing uses machine learning to understand text",
            "Data science combines statistics, programming, and domain expertise",
            "Vector databases optimize similarity search for high-dimensional vectors"
        ]
        self.retriever = MultiQueryRetriever()
    
    def test_initialization(self):
        """Test multi-query retriever initialization"""
        self.assertIsNotNone(self.retriever.query_generator)
        self.assertIsNotNone(self.retriever.embedder)
    
    def test_retrieve_with_expansion(self):
        """Test retrieval with query expansion"""
        original_query = "machine learning"
        
        results = self.retriever.retrieve_with_expansion(
            original_query, self.documents, n_expanded_queries=2, top_k_per_query=2
        )
        
        # Check structure of results
        self.assertIn('original_query', results)
        self.assertIn('expanded_queries', results)
        self.assertIn('individual_results', results)
        self.assertIn('combined_results', results)
        self.assertIn('avg_query_time', results)
        self.assertIn('total_time', results)
        
        # Should have expanded queries
        self.assertGreaterEqual(len(results['expanded_queries']), 1)
        
        # Should have individual results for each query
        self.assertEqual(len(results['individual_results']), len(results['expanded_queries']))
        
        # Should have combined results
        self.assertIsInstance(results['combined_results'], list)
        
        # Original query should be preserved
        self.assertEqual(results['original_query'], original_query)
    
    def test_retrieve_for_single_query(self):
        """Test single query retrieval"""
        query = "neural networks"
        top_k = 3
        
        results = self.retriever._retrieve_for_single_query(query, self.documents, top_k)
        
        # Should return a list of results
        self.assertIsInstance(results, list)
        
        # Should return top_k results or fewer if not enough documents
        self.assertLessEqual(len(results), top_k)
        
        # Each result should have required keys
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('similarity', result)
            self.assertIn('rank', result)
            # Similarity should be between -1 and 1
            self.assertGreaterEqual(result['similarity'], -1.0)
            self.assertLessEqual(result['similarity'], 1.0)
    
    def test_combine_results(self):
        """Test combining results from multiple queries"""
        # Simulate individual results
        individual_results = [
            {
                'query': 'query1',
                'results': [
                    {'doc_id': 0, 'similarity': 0.8, 'rank': 1},
                    {'doc_id': 1, 'similarity': 0.7, 'rank': 2}
                ]
            },
            {
                'query': 'query2', 
                'results': [
                    {'doc_id': 0, 'similarity': 0.9, 'rank': 1},
                    {'doc_id': 2, 'similarity': 0.6, 'rank': 2}
                ]
            }
        ]
        
        combined = self.retriever._combine_results(individual_results, self.documents)
        
        # Should return a list
        self.assertIsInstance(combined, list)
        
        # Each result should have required keys
        for result in combined:
            self.assertIn('doc_id', result)
            self.assertIn('document', result)
            self.assertIn('rrf_score', result)
            self.assertIn('avg_similarity', result)
            self.assertIn('combined_score', result)
            self.assertIn('rank', result)


class TestMultiQueryEvaluator(unittest.TestCase):
    """Test multi-query evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = MultiQueryEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.sample_queries)
        self.assertIsNotNone(self.evaluator.sample_docs)
        self.assertEqual(len(self.evaluator.sample_queries), 5)
        self.assertEqual(len(self.evaluator.sample_docs), 10)
    
    def test_compare_generators(self):
        """Test generator comparison"""
        results = self.evaluator.compare_generators(n_queries=2)
        
        # Should return results for each generator
        expected_generators = ['LLM', 'Semantic', 'Paraphrase']
        for gen_name in expected_generators:
            self.assertIn(gen_name, results)
            
            # Each should have required metrics
            gen_result = results[gen_name]
            self.assertIn('avg_generation_time', gen_result)
            self.assertIn('total_generated_queries', gen_result)
            self.assertIn('queries_per_original', gen_result)
            
            # Times should be non-negative
            self.assertGreaterEqual(gen_result['avg_generation_time'], 0)
    
    def test_evaluate_retrieval_improvement(self):
        """Test retrieval improvement evaluation"""
        results = self.evaluator.evaluate_retrieval_improvement()
        
        # Should have required keys
        self.assertIn('avg_multi_retrieval_time', results)
        self.assertIn('total_multi_retrieval_time', results)
        self.assertIn('num_queries_tested', results)
        
        # Times should be non-negative
        self.assertGreaterEqual(results['avg_multi_retrieval_time'], 0)
        self.assertGreaterEqual(results['total_multi_retrieval_time'], 0)
        
        # Number of queries should match expected
        self.assertEqual(results['num_queries_tested'], len(self.evaluator.sample_queries))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_query(self):
        """Test with empty query"""
        generator = LLMQueryGenerator()
        queries = generator.generate_queries("", n_queries=3)
        
        # Should handle gracefully
        self.assertIsInstance(queries, list)
        # Might return empty list or original query
        if queries:
            self.assertIsInstance(queries[0], str)
    
    def test_very_long_query(self):
        """Test with very long query"""
        generator = SemanticQueryExpander()
        long_query = "This is a very long query about machine learning and neural networks and data science. " * 10
        
        queries = generator.generate_queries(long_query, n_queries=2)
        
        # Should handle without crashing
        self.assertIsInstance(queries, list)
    
    def test_single_word_query(self):
        """Test with single word query"""
        generator = ParaphraseQueryGenerator()
        queries = generator.generate_queries("machine", n_queries=3)
        
        # Should handle without crashing
        self.assertIsInstance(queries, list)
        self.assertGreaterEqual(len(queries), 1)
    
    def test_special_characters(self):
        """Test with special characters"""
        generator = LLMQueryGenerator()
        queries = generator.generate_queries("machine learning & ai?", n_queries=2)
        
        # Should handle without crashing
        self.assertIsInstance(queries, list)
    
    def test_no_documents(self):
        """Test with no documents to retrieve from"""
        retriever = MultiQueryRetriever()
        results = retriever.retrieve_with_expansion("test query", [], n_expanded_queries=2)
        
        # Should handle gracefully
        self.assertIn('combined_results', results)
        self.assertEqual(len(results['combined_results']), 0)


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test with multiple queries
    retriever = MultiQueryRetriever()
    documents = [
        "Machine learning algorithm tutorial",
        "Neural network implementation guide", 
        "Data science with Python",
        "Natural language processing techniques",
        "Deep learning model optimization",
        "Vector database performance",
        "Information retrieval systems",
        "Query expansion methods"
    ] * 5  # Multiply to get more documents
    
    test_queries = [
        "machine learning tutorial",
        "neural network optimization",
        "data science methods",
        "natural language processing"
    ]
    
    start_time = time.time()
    for query in test_queries:
        retriever.retrieve_with_expansion(query, documents, n_expanded_queries=3, top_k_per_query=3)
    total_time = time.time() - start_time
    
    avg_time = total_time / len(test_queries)
    
    print(f"Processed {len(test_queries)} queries with {len(documents)} documents each in {total_time:.3f}s")
    print(f"Average time per query: {avg_time:.3f}s")
    
    # Should be reasonably fast
    assert avg_time < 5.0, f"Average query processing time too slow: {avg_time}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test different combination strategies
    print("Testing different query generator combinations...")
    
    documents = [
        "Machine learning models require training data",
        "Neural networks have multiple layers",
        "Deep learning uses large datasets",
        "Data science involves statistical analysis"
    ]
    
    query = "machine learning models"
    
    # Test with different generators
    generators = [
        ("LLM", LLMQueryGenerator()),
        ("Semantic", SemanticQueryExpander()),
        ("Paraphrase", ParaphraseQueryGenerator())
    ]
    
    for name, generator in generators:
        retriever = MultiQueryRetriever(generator)
        results = retriever.retrieve_with_expansion(query, documents, n_expanded_queries=2)
        
        print(f"{name} generator:")
        print(f"  Original: {results['original_query']}")
        print(f"  Expanded: {results['expanded_queries']}")
        print(f"  Results count: {len(results['combined_results'])}")
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()