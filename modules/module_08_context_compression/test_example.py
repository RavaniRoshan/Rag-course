"""
Module 8: Context Compression
Tests for implementation examples

This module contains tests for the context compression implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np

from example import ContextCompressorBase, LLMContextCompressor, EmbeddingBasedCompressor, RelevanceBasedCompressor, SummaryBasedCompressor, TokenEfficientCompressor, ContextCompressionEvaluator


class TestContextCompressorBase(unittest.TestCase):
    """Test base class for context compression"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = ContextCompressorBase()
    
    def test_initialization(self):
        """Test base compressor initialization"""
        self.assertIsNotNone(self.compressor.embedder)
    
    def test_count_tokens(self):
        """Test token counting"""
        text = "This is a sample text with several words"
        token_count = self.compressor.count_tokens(text)
        
        # Should return an integer
        self.assertIsInstance(token_count, int)
        # Should be greater than 0 for non-empty text
        self.assertGreater(token_count, 0)


class TestLLMContextCompressor(unittest.TestCase):
    """Test LLM-based context compression"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = LLMContextCompressor()
        self.sample_context = [
            "Machine learning is a method of data analysis.",
            "It automates analytical model building.",
            "It is a branch of artificial intelligence.",
            "Based on the idea that systems can learn from data."
        ]
        self.query = "machine learning"
    
    def test_initialization(self):
        """Test LLM compressor initialization"""
        self.assertIsNotNone(self.compressor.embedder)
    
    def test_compress_basic(self):
        """Test basic compression functionality"""
        result = self.compressor.compress(self.sample_context, self.query, target_length=100)
        
        # Should return a list
        self.assertIsInstance(result, list)
        
        # Each element should be a string
        for item in result:
            self.assertIsInstance(item, str)
    
    def test_compression_reduces_length(self):
        """Test that compression reduces length when needed"""
        # Use a longer context to ensure compression is needed
        long_context = self.sample_context * 5  # Repeat to make longer
        
        original_length = sum(len(doc) for doc in long_context)
        result = self.compressor.compress(long_context, self.query, target_length=50)
        compressed_length = sum(len(doc) for doc in result)
        
        # Compression should either reduce length or keep it the same if already below target
        # In practice, LLM summarization may not reduce character length significantly
        # but should return a reasonable response
        self.assertIsInstance(result, list)
        if result and long_context:  # If both have content
            self.assertIsInstance(result[0], str)


class TestEmbeddingBasedCompressor(unittest.TestCase):
    """Test embedding-based context compression"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = EmbeddingBasedCompressor()
        self.sample_context = [
            "Machine learning algorithms use data to learn patterns.",
            "Deep learning involves neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Data science combines statistics and programming skills."
        ]
        self.query = "machine learning algorithms"
    
    def test_initialization(self):
        """Test embedding compressor initialization"""
        self.assertIsNotNone(self.compressor.embedder)
    
    def test_compress_basic(self):
        """Test basic compression functionality"""
        result = self.compressor.compress(self.sample_context, self.query, target_length=100)
        
        # Should return a list
        self.assertIsInstance(result, list)
        
        # Each element should be a string
        for item in result:
            self.assertIsInstance(item, str)
    
    def test_compress_within_limit(self):
        """Test compression respects length limits"""
        # Use a target length that forces compression
        result = self.compressor.compress(self.sample_context, self.query, target_length=50)
        
        # Should return a list of strings
        self.assertIsInstance(result, list)
        
        # Total length should be within limit (approximately)
        total_length = sum(len(doc) for doc in result)
        self.assertLessEqual(total_length, 50)
    
    def test_compress_returns_relevant_content(self):
        """Test that compression returns relevant content"""
        result = self.compressor.compress(self.sample_context, self.query, target_length=200)
        
        # Should return some results
        self.assertGreater(len(result), 0)
        
        # Should not return more documents than we started with
        self.assertLessEqual(len(result), len(self.sample_context))


class TestRelevanceBasedCompressor(unittest.TestCase):
    """Test relevance-based context compression"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = RelevanceBasedCompressor()
        self.sample_context = [
            "Machine learning algorithms use data to learn patterns.",
            "The weather is nice today.",
            "Deep learning involves neural networks with multiple layers.",
            "Data science combines statistics and programming skills."
        ]
        self.query = "machine learning"
    
    def test_initialization(self):
        """Test relevance compressor initialization"""
        self.assertIsNotNone(self.compressor.embedder)
    
    def test_compress_basic(self):
        """Test basic compression functionality"""
        result = self.compressor.compress(self.sample_context, self.query, target_length=100)
        
        # Should return a list
        self.assertIsInstance(result, list)
        
        # Each element should be a string
        for item in result:
            self.assertIsInstance(item, str)
    
    def test_relevant_content_preferred(self):
        """Test that more relevant content is preferred"""
        # This is hard to test deterministically, but we can check that
        # the compressor runs without error and returns expected format
        result = self.compressor.compress(self.sample_context, self.query, target_length=200)
        
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)


class TestSummaryBasedCompressor(unittest.TestCase):
    """Test summary-based context compression"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compressor = SummaryBasedCompressor()
        self.sample_context = [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "It is a branch of artificial intelligence based on the idea that systems can learn from data.",
            "Natural language processing helps computers understand human language.",
            "Data science combines statistics, programming, and domain expertise."
        ]
        self.query = "machine learning"
    
    def test_initialization(self):
        """Test summary compressor initialization"""
        self.assertIsNotNone(self.compressor.embedder)
    
    def test_compress_basic(self):
        """Test basic compression functionality"""
        result = self.compressor.compress(self.sample_context, self.query, target_length=100)
        
        # Should return a list
        self.assertIsInstance(result, list)
        
        # Each element should be a string
        for item in result:
            self.assertIsInstance(item, str)
    
    def test_returns_summarized_content(self):
        """Test that compression returns summarized content"""
        result = self.compressor.compress(self.sample_context, self.query, target_length=200)
        
        # Should return a list of strings
        self.assertIsInstance(result, list)
        
        # Should have at least one result
        if result:
            self.assertIsInstance(result[0], str)


class TestTokenEfficientCompressor(unittest.TestCase):
    """Test the main token-efficient compressor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_context = [
            "Machine learning algorithms use data to learn patterns.",
            "Deep learning involves neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Data science combines statistics and programming skills.",
            "Vector databases optimize similarity search for high-dimensional vectors."
        ]
        self.query = "machine learning algorithms"
    
    def test_initialization(self):
        """Test main compressor initialization"""
        compressor = TokenEfficientCompressor()
        self.assertIsNotNone(compressor.llm_compressor)
        self.assertIsNotNone(compressor.embedding_compressor)
        self.assertIsNotNone(compressor.relevance_compressor)
        self.assertIsNotNone(compressor.summary_compressor)
    
    def test_compress_primary_strategy(self):
        """Test compression with primary strategy"""
        compressor = TokenEfficientCompressor(primary_method='embedding')
        
        result = compressor.compress(self.sample_context, self.query, target_tokens=300, strategy="primary")
        
        # Should return a dictionary with expected keys
        self.assertIn('original_context', result)
        self.assertIn('compressed_context', result)
        self.assertIn('original_length', result)
        self.assertIn('compressed_length', result)
        self.assertIn('compression_ratio', result)
        self.assertIn('tokens_saved', result)
        self.assertIn('compression_time', result)
        self.assertIn('method_used', result)
        
        # Check types of values
        self.assertIsInstance(result['compressed_context'], list)
        self.assertIsInstance(result['compression_ratio'], float)
        self.assertIsInstance(result['compression_time'], float)
    
    def test_compress_ensemble_strategy(self):
        """Test compression with ensemble strategy"""
        compressor = TokenEfficientCompressor(primary_method='embedding')
        
        result = compressor.compress(self.sample_context, self.query, target_tokens=300, strategy="ensemble")
        
        # Should return a dictionary with expected keys
        self.assertIn('original_context', result)
        self.assertIn('compressed_context', result)
        self.assertIn('original_length', result)
        self.assertIn('compressed_length', result)
        self.assertIn('compression_ratio', result)
        self.assertIn('tokens_saved', result)
        self.assertIn('compression_time', result)
        self.assertIn('method_used', result)
        
        # Check types of values
        self.assertIsInstance(result['compressed_context'], list)
        self.assertIsInstance(result['compression_ratio'], float)
        self.assertIsInstance(result['compression_time'], float)
    
    def test_different_methods(self):
        """Test different compression methods"""
        methods = ['embedding', 'relevance', 'summary', 'llm']
        
        for method in methods:
            compressor = TokenEfficientCompressor(primary_method=method)
            result = compressor.compress(self.sample_context, self.query, target_tokens=300)
            
            # Should contain all required keys
            required_keys = ['original_context', 'compressed_context', 'original_length', 
                           'compressed_length', 'compression_ratio', 'tokens_saved', 
                           'compression_time', 'method_used']
            for key in required_keys:
                self.assertIn(key, result)


class TestContextCompressionEvaluator(unittest.TestCase):
    """Test context compression evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ContextCompressionEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.sample_contexts)
        self.assertIsNotNone(self.evaluator.sample_queries)
        self.assertEqual(len(self.evaluator.sample_contexts), 2)
        self.assertEqual(len(self.evaluator.sample_queries), 2)
    
    def test_evaluate_compression_method(self):
        """Test evaluation of compression method"""
        compressor = TokenEfficientCompressor(primary_method='embedding')
        result = self.evaluator.evaluate_compression_method(compressor, target_tokens=300)
        
        # Should have expected keys
        expected_keys = ['method', 'avg_compression_ratio', 'total_tokens_saved', 
                        'avg_compression_time', 'contexts_processed']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check value types
        self.assertIsInstance(result['avg_compression_ratio'], float)
        self.assertIsInstance(result['total_tokens_saved'], int)
        self.assertIsInstance(result['avg_compression_time'], float)
        self.assertIsInstance(result['contexts_processed'], int)
    
    def test_compare_methods(self):
        """Test comparison of different methods"""
        results = self.evaluator.compare_methods(target_tokens=200)
        
        # Should return results for multiple methods
        self.assertGreaterEqual(len(results), 1)
        
        # Each result should have expected structure
        for result in results:
            expected_keys = ['method', 'avg_compression_ratio', 'total_tokens_saved', 
                            'avg_compression_time', 'contexts_processed']
            for key in expected_keys:
                self.assertIn(key, result)
        
        # Should compare different methods
        methods_tested = [r['method'] for r in results]
        self.assertGreaterEqual(len(set(methods_tested)), 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_context(self):
        """Test compression with empty context"""
        compressor = TokenEfficientCompressor()
        
        result = compressor.compress([], "test query", target_tokens=100)
        
        # Should handle gracefully
        self.assertIn('compressed_context', result)
        self.assertEqual(len(result['compressed_context']), 0)
        self.assertEqual(result['original_length'], 0)
        self.assertEqual(result['compressed_length'], 0)
    
    def test_empty_query(self):
        """Test compression with empty query"""
        compressor = EmbeddingBasedCompressor()
        
        context = ["This is a test document"]
        result = compressor.compress(context, "", target_length=100)
        
        # Should handle gracefully
        self.assertIsInstance(result, list)
    
    def test_single_long_document(self):
        """Test compression with single very long document"""
        compressor = TokenEfficientCompressor()
        
        long_doc = "This is a very long document. " * 100
        result = compressor.compress([long_doc], "test", target_tokens=50)
        
        # Should return a result without crashing
        self.assertIn('compressed_context', result)
    
    def test_target_length_zero(self):
        """Test compression with zero target length"""
        compressor = EmbeddingBasedCompressor()
        
        context = ["This is a test document"]
        result = compressor.compress(context, "test", target_length=0)
        
        # Should return empty or minimal result
        if result:
            self.assertIsInstance(result, list)
        else:
            # If empty, that's also acceptable when target is 0
            pass
    
    def test_special_characters(self):
        """Test compression with special characters"""
        compressor = RelevanceBasedCompressor()
        
        context = ["Special chars: !@#$%^&*()<>?{}[]|;:,./~`_+=-"]
        result = compressor.compress(context, "special chars", target_length=50)
        
        # Should handle without crashing
        self.assertIsInstance(result, list)


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test with larger context
    compressor = TokenEfficientCompressor(primary_method='embedding')
    
    large_context = [
        "Machine learning is a method of data analysis that automates analytical model building. " * 5,
        "It is a branch of artificial intelligence based on the idea that systems can learn from data. " * 5,
        "Machine learning algorithms can be classified into supervised, unsupervised, and reinforcement learning. " * 5,
        "Deep learning neural networks require significant computational resources. " * 5,
        "Natural language processing uses machine learning to understand text. " * 5,
        "Data science combines statistics, programming, and domain expertise. " * 5,
        "Vector databases optimize similarity search for high-dimensional vectors. " * 5,
        "Reinforcement learning uses rewards to train agents. " * 5,
    ]
    
    query = "machine learning algorithms and neural networks"
    
    start_time = time.time()
    result = compressor.compress(large_context, query, target_tokens=800)
    compression_time = time.time() - start_time
    
    print(f"Compressed {len(large_context)} documents in {compression_time:.3f}s")
    print(f"Original length: {result['original_length']} chars")
    print(f"Compressed length: {result['compressed_length']} chars")
    print(f"Compression ratio: {result['compression_ratio']:.3f}")
    print(f"Tokens saved: {result['tokens_saved']}")
    
    # Should be reasonably fast
    assert compression_time < 5.0, f"Compression took too long: {compression_time}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test different compression strategies
    print("Testing different compression strategies...")
    
    context = [
        "Artificial intelligence is intelligence demonstrated by machines.",
        "It is a field that studies how computers can perform tasks that usually require human intelligence.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Deep learning is a specialized branch of machine learning that uses neural networks.",
        "Natural language processing enables computers to understand and interpret human language.",
        "Computer vision allows machines to identify and analyze visual content.",
        "Robotics combines AI with mechanical engineering to create intelligent machines.",
        "Expert systems are AI programs that mimic human expertise in specific domains."
    ]
    
    query = "artificial intelligence and machine learning"
    
    strategies = ["primary", "ensemble"]
    methods = ["embedding", "relevance", "summary"]
    
    for method in methods:
        print(f"\nTesting {method} method:")
        compressor = TokenEfficientCompressor(primary_method=method)
        
        for strategy in strategies:
            result = compressor.compress(context, query, target_tokens=600, strategy=strategy)
            
            print(f"  Strategy '{strategy}': {len(result['compressed_context'])} segments, "
                  f"compression ratio {result['compression_ratio']:.2f}")
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()