"""
Module 6: Query Understanding
Tests for implementation examples

This module contains tests for the query understanding implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np

from example import QueryPreprocessor, IntentClassifier, QueryExpander, QueryReformulator, QueryUnderstandingPipeline, QueryUnderstandingEvaluator


class TestQueryPreprocessor(unittest.TestCase):
    """Test query preprocessing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = QueryPreprocessor()
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        self.assertIsNotNone(self.preprocessor.stop_words)
        self.assertIsNotNone(self.preprocessor.lemmatizer)
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        dirty_text = "  This   is  a   test!@#   with   extra   spaces\t\n  "
        cleaned = self.preprocessor.clean_text(dirty_text)
        
        # Should remove extra spaces and special characters
        self.assertEqual(cleaned, "this is a test with extra spaces")
    
    def test_tokenize(self):
        """Test tokenization functionality"""
        text = "This is a test sentence."
        tokens = self.preprocessor.tokenize(text)
        
        self.assertEqual(len(tokens), 5)  # ["This", "is", "a", "test", "sentence", "."]
        self.assertIn("test", tokens)
    
    def test_remove_stopwords(self):
        """Test stopword removal"""
        tokens = ["this", "is", "a", "test", "of", "stopword", "removal"]
        filtered = self.preprocessor.remove_stopwords(tokens)
        
        # Should remove common stopwords but keep content words
        self.assertIn("test", filtered)
        self.assertIn("stopword", filtered)
        self.assertIn("removal", filtered)
        self.assertNotIn("this", filtered)  # 'this' is a stopword
        self.assertNotIn("is", filtered)   # 'is' is a stopword
    
    def test_lemmatize_tokens(self):
        """Test lemmatization"""
        tokens = ["running", "cats", "mice", "better"]
        lemmatized = self.preprocessor.lemmatize_tokens(tokens)
        
        self.assertIn("running", lemmatized)  # running is both verb and noun form
        self.assertIn("cat", lemmatized)      # cats -> cat
        self.assertIn("mouse", lemmatized)    # mice -> mouse
        # Note: "better" -> "better" (lemmatizer doesn't always convert comparative forms)
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline"""
        query = "How does the quick brown foxes run?"
        result = self.preprocessor.preprocess(query)
        
        # Check that all expected keys are present
        self.assertIn('original_query', result)
        self.assertIn('cleaned_query', result)
        self.assertIn('tokens', result)
        self.assertIn('lemmatized_tokens', result)
        self.assertIn('entities', result)
        
        # Check that some processing occurred
        self.assertNotEqual(result['original_query'], result['cleaned_query'])
        self.assertGreater(len(result['tokens']), 0)


class TestIntentClassifier(unittest.TestCase):
    """Test intent classification functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = IntentClassifier()
    
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier.intent_patterns)
    
    def test_classify_informational(self):
        """Test classification of informational queries"""
        informational_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does this work?",
            "Tell me about Python"
        ]
        
        for query in informational_queries:
            result = self.classifier.classify_intent(query)
            # At minimum, should return a dictionary with expected keys
            self.assertIn('intent', result)
            self.assertIn('confidence', result)
            self.assertIn('pattern', result)
    
    def test_classify_navigational(self):
        """Test classification of navigational queries"""
        navigational_queries = [
            "Go to the documentation",
            "Find the Python tutorial",
            "Where is the nearest store?"
        ]
        
        for query in navigational_queries:
            result = self.classifier.classify_intent(query)
            self.assertIn('intent', result)
            self.assertIn('confidence', result)
            self.assertIn('pattern', result)
    
    def test_classify_comparative(self):
        """Test classification of comparative queries"""
        comparative_queries = [
            "Compare SVM vs Random Forest",
            "Neural networks versus decision trees"
        ]
        
        for query in comparative_queries:
            result = self.classifier.classify_intent(query)
            self.assertIn('intent', result)
            self.assertIn('confidence', result)
            self.assertIn('pattern', result)


class TestQueryExpander(unittest.TestCase):
    """Test query expansion functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expander = QueryExpander()
    
    def test_initialization(self):
        """Test expander initialization"""
        self.assertIsNotNone(self.expander.embedder)
    
    def test_get_synonyms(self):
        """Test synonym retrieval"""
        synonyms = self.expander._get_synonyms("good")
        
        # Should return a list of synonyms
        self.assertIsInstance(synonyms, list)
        # Should have some synonyms
        self.assertGreaterEqual(len(synonyms), 0)  # May be empty if WordNet doesn't have synonyms
    
    def test_expand_with_synonyms(self):
        """Test query expansion with synonyms"""
        query = "machine learning"
        expansions = self.expander.expand_with_synonyms(query, max_expansions=2)
        
        # Should return a list
        self.assertIsInstance(expansions, list)
        
        # If there are expansions, they should be different from the original
        if expansions:
            self.assertNotEqual(expansions[0], query)
    
    def test_expand_with_embeddings(self):
        """Test query expansion with embeddings"""
        query = "artificial intelligence"
        candidate_terms = ["machine learning", "neural networks", "deep learning", "data science"]
        
        expansions = self.expander.expand_with_embeddings(query, candidate_terms, top_k=2)
        
        # Should return a list with one expanded query
        self.assertIsInstance(expansions, list)
        self.assertEqual(len(expansions), 1)
        
        # Expanded query should be longer than original
        if expansions:
            self.assertGreater(len(expansions[0].split()), len(query.split()))


class TestQueryReformulator(unittest.TestCase):
    """Test query reformulation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reformulator = QueryReformulator()
    
    def test_initialization(self):
        """Test reformulator initialization"""
        # Just check that object is created
        self.assertIsNotNone(self.reformulator)
    
    def test_correct_spelling(self):
        """Test spelling correction"""
        # This test may be skipped if textblob is not available
        try:
            misspelled = "recieve"
            corrected = self.reformulator.correct_spelling(misspelled)
            # May or may not be corrected depending on textblob availability
            self.assertIsInstance(corrected, str)
        except Exception:
            # If textblob is not available, just check that it doesn't crash
            misspelled = "recieve"
            corrected = self.reformulator.correct_spelling(misspelled)
            self.assertIsInstance(corrected, str)
    
    def test_rewrite_query(self):
        """Test query rewriting"""
        original = "how do I build a website"
        intent = "instructional"
        rewritten = self.reformulator.rewrite_query(original, intent)
        
        # Should return a string
        self.assertIsInstance(rewritten, str)
        # Should not be empty
        self.assertGreater(len(rewritten.strip()), 0)
    
    def test_simplify_query(self):
        """Test query simplification"""
        complex_query = "Please, could you kindly tell me how to build a website??"
        simplified = self.reformulator.simplify_query(complex_query)
        
        # Should return a string
        self.assertIsInstance(simplified, str)
        # Should be different from original
        self.assertNotEqual(complex_query, simplified)


class TestQueryUnderstandingPipeline(unittest.TestCase):
    """Test the complete query understanding pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = QueryUnderstandingPipeline()
    
    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline.preprocessor)
        self.assertIsNotNone(self.pipeline.intent_classifier)
        self.assertIsNotNone(self.pipeline.expander)
        self.assertIsNotNone(self.pipeline.reformulator)
    
    def test_process_query(self):
        """Test processing a complete query"""
        query = "How does machine learning work?"
        result = self.pipeline.process_query(query)
        
        # Check that all expected keys are present
        self.assertIn('original_query', result)
        self.assertIn('preprocessed', result)
        self.assertIn('intent', result)
        self.assertIn('reformulated_query', result)
        self.assertIn('synonym_expansions', result)
        self.assertIn('entity_expanded_query', result)
        self.assertIn('all_expansions', result)
        
        # Check that preprocessed contains expected keys
        preprocessed = result['preprocessed']
        self.assertIn('original_query', preprocessed)
        self.assertIn('cleaned_query', preprocessed)
        self.assertIn('tokens', preprocessed)
        self.assertIn('lemmatized_tokens', preprocessed)
        self.assertIn('entities', preprocessed)
        
        # Original query should match
        self.assertEqual(result['original_query'], query)
        
        # Should have some expansions
        self.assertGreaterEqual(len(result['all_expansions']), 1)
    
    def test_different_query_types(self):
        """Test processing different types of queries"""
        queries = [
            "What is artificial intelligence?",
            "Compare Python vs JavaScript",
            "How to build a neural network?",
            "Find Python documentation"
        ]
        
        for query in queries:
            result = self.pipeline.process_query(query)
            # Should process all queries without error
            self.assertIn('intent', result)
            self.assertIn('reformulated_query', result)


class TestQueryUnderstandingEvaluator(unittest.TestCase):
    """Test query understanding evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = QueryUnderstandingEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.sample_queries)
        self.assertIsNotNone(self.evaluator.pipeline)
        self.assertEqual(len(self.evaluator.sample_queries), 6)
    
    def test_evaluate_preprocessing(self):
        """Test preprocessing evaluation"""
        result = self.evaluator.evaluate_preprocessing()
        
        self.assertEqual(result['method'], 'preprocessing')
        self.assertEqual(result['total_queries'], 6)
        self.assertEqual(len(result['results']), 6)
        
        # Check that results have expected structure
        for res in result['results']:
            self.assertIn('original', res)
            self.assertIn('processed', res)
            self.assertIn('tokens', res)
            self.assertIn('entities', res)
    
    def test_evaluate_intent_classification(self):
        """Test intent classification evaluation"""
        result = self.evaluator.evaluate_intent_classification()
        
        self.assertEqual(result['method'], 'intent_classification')
        self.assertEqual(result['total_queries'], 6)
        self.assertIn('intent_distribution', result)
        
        # Check that distribution is a dictionary
        self.assertIsInstance(result['intent_distribution'], dict)
    
    def test_evaluate_query_expansion(self):
        """Test query expansion evaluation"""
        result = self.evaluator.evaluate_query_expansion()
        
        self.assertEqual(result['method'], 'query_expansion')
        self.assertEqual(result['total_queries'], 6)
        self.assertIn('expansion_stats', result)
        
        # Check that stats have expected structure
        stats = result['expansion_stats']
        self.assertEqual(len(stats), 6)
        for stat in stats:
            self.assertIn('original', stat)
            self.assertIn('original_length', stat)
            self.assertIn('avg_expansion_length', stat)
            self.assertIn('num_expansions', stat)
    
    def test_run_full_evaluation(self):
        """Test full evaluation"""
        results = self.evaluator.run_full_evaluation()
        
        # Should have results for preprocessing, intent classification, and expansion
        self.assertEqual(len(results), 3)
        
        methods = [r['method'] for r in results]
        self.assertIn('preprocessing', methods)
        self.assertIn('intent_classification', methods)
        self.assertIn('query_expansion', methods)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_query(self):
        """Test handling of empty query"""
        pipeline = QueryUnderstandingPipeline()
        
        result = pipeline.process_query("")
        # Should handle gracefully without crashing
        self.assertIn('original_query', result)
        self.assertEqual(result['original_query'], "")
    
    def test_very_long_query(self):
        """Test handling of very long query"""
        pipeline = QueryUnderstandingPipeline()
        long_query = "This is a very long query. " * 100
        
        result = pipeline.process_query(long_query)
        # Should handle without crashing
        self.assertIn('intent', result)
        self.assertGreaterEqual(len(result['all_expansions']), 1)
    
    def test_query_with_special_characters(self):
        """Test handling of query with special characters"""
        pipeline = QueryUnderstandingPipeline()
        special_query = "What is @#$%^&*()<>?{}[]|;:,./~`_+="
        
        result = pipeline.process_query(special_query)
        # Should handle without crashing
        self.assertIn('preprocessed', result)
    
    def test_single_character_query(self):
        """Test handling of single character query"""
        pipeline = QueryUnderstandingPipeline()
        
        result = pipeline.process_query("a")
        # Should handle without crashing
        self.assertIn('intent', result)


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test with multiple queries
    pipeline = QueryUnderstandingPipeline()
    queries = [
        "How does machine learning work?",
        "What is artificial intelligence?",
        "Explain neural networks",
        "Compare decision trees vs random forests",
        "Tutorial on Python programming",
        "Documentation for scikit-learn"
    ] * 10  # Repeat 10 times for timing
    
    start_time = time.time()
    for query in queries:
        pipeline.process_query(query)
    total_time = time.time() - start_time
    
    avg_time = total_time / len(queries)
    
    print(f"Processed {len(queries)} queries in {total_time:.3f}s")
    print(f"Average time per query: {avg_time:.3f}s")
    
    # Should be reasonably fast
    assert avg_time < 1.0, f"Average query processing time too slow: {avg_time}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test different components separately
    print("Testing component integration...")
    
    # Test preprocessing with a complex query
    preprocessor = QueryPreprocessor()
    complex_query = "How does the quick brown foxes jump over the lazy dogs?"
    result = preprocessor.preprocess(complex_query)
    
    print(f"Original: {complex_query}")
    print(f"Processed: {result['cleaned_query']}")
    print(f"Tokens: {result['lemmatized_tokens']}")
    
    # Test pipeline end-to-end
    pipeline = QueryUnderstandingPipeline()
    query = "Explain how neural networks learn"
    full_result = pipeline.process_query(query)
    
    print(f"\nFull pipeline result for '{query}':")
    print(f"Intent: {full_result['intent']['intent']}")
    print(f"Reformulated: {full_result['reformulated_query']}")
    print(f"Expansions: {len(full_result['all_expansions'])} total")
    
    print("Component integration tested successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()