"""
Test file for Module 3: Embedding Models Implementation
This file contains unit tests to validate the embedding models functionality.
"""

import unittest
import sys
import os

# Add the module directory to the path so we can import the example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from example import AdvancedEmbeddingModel, MultiModelEmbeddingEnsemble, DomainSpecificEmbedder, HealthcareEmbedder, IntelligentEmbeddingSelector, EmbeddingResult


class TestEmbeddingModelsModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Using a lightweight model for testing
        self.model = AdvancedEmbeddingModel("all-MiniLM-L6-v2")
        self.sample_texts = [
            "This is the first sentence.",
            "This is another sentence for testing.",
            "A completely different topic altogether."
        ]
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.tokenizer)
        self.assertEqual(self.model.model_name, "all-MiniLM-L6-v2")
    
    def test_single_text_encoding(self):
        """Test encoding of a single text"""
        result = self.model.encode("Test sentence")
        
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0].vector, np.ndarray)
        self.assertGreater(len(result[0].vector), 0)
        self.assertEqual(result[0].text, "Test sentence")
    
    def test_multiple_text_encoding(self):
        """Test encoding of multiple texts"""
        results = self.model.encode(self.sample_texts)
        
        self.assertEqual(len(results), len(self.sample_texts))
        for result in results:
            self.assertIsInstance(result.vector, np.ndarray)
            self.assertGreater(len(result.vector), 0)
    
    def test_embedding_result_structure(self):
        """Test the EmbeddingResult dataclass"""
        result = self.model.encode("Test")[0]
        
        self.assertIsInstance(result, EmbeddingResult)
        self.assertIsInstance(result.vector, (list, tuple, type(...)))
        self.assertIsInstance(result.tokens_used, int)
        self.assertIsInstance(result.processing_time, float)
        self.assertIsInstance(result.model_name, str)
        self.assertIsInstance(result.text, str)


class TestMultiModelEnsemble(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for ensemble tests."""
        self.ensemble = MultiModelEmbeddingEnsemble([
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L3-v2"
        ])
    
    def test_ensemble_encoding(self):
        """Test multi-model ensemble encoding"""
        texts = ["Test sentence 1", "Test sentence 2"]
        results = self.ensemble.encode(texts, aggregation_method="weighted_average")
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result.vector, (list, tuple, type(...)))
            self.assertGreater(len(result.vector), 0)
    
    def test_concatenation_aggregation(self):
        """Test concatenation aggregation method"""
        texts = ["Test sentence"]
        results = self.ensemble.encode(texts, aggregation_method="concatenation")
        
        self.assertEqual(len(results), 1)
        # Concatenated embeddings should be longer than individual ones
        self.assertGreater(len(results[0].vector), 384)  # Individual models typically have 384 dims


class TestDomainSpecificEmbedders(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for domain-specific embedders."""
        base_model = AdvancedEmbeddingModel("all-MiniLM-L6-v2")
        self.health_embedder = HealthcareEmbedder(base_model)
    
    def test_healthcare_preprocessing(self):
        """Test healthcare-specific preprocessing"""
        original_text = "Pt. has HTN and DM."
        processed = self.health_embedder._medical_preprocessor(original_text)
        
        # Should expand abbreviations
        self.assertIn("patient", processed.lower())
        self.assertIn("hypertension", processed.lower())
        self.assertIn("diabetes", processed.lower())
    
    def test_medical_abbreviations_expansion(self):
        """Test expansion of medical abbreviations"""
        text = "Patient scheduled for EKG and MRI"
        expanded = self.health_embedder._expand_medical_abbreviations(text)
        
        self.assertIn("electrocardiogram", expanded.lower())
        self.assertIn("magnetic resonance imaging", expanded.lower())
    
    def test_healthcare_encoding(self):
        """Test healthcare-specific encoding"""
        result = self.health_embedder.encode("Patient has hypertension and diabetes")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0].vector, (list, tuple, type(...)))


class TestIntelligentSelector(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for intelligent selector."""
        self.selector = IntelligentEmbeddingSelector()
    
    def test_text_analysis(self):
        """Test text characteristic analysis"""
        text = "This is a test sentence."
        characteristics = self.selector.analyze_text(text)
        
        self.assertIn("length", characteristics)
        self.assertIn("word_count", characteristics)
        self.assertIn("avg_word_length", characteristics)
        self.assertIsInstance(characteristics["length"], int)
        self.assertIsInstance(characteristics["word_count"], int)
        self.assertIsInstance(characteristics["avg_word_length"], float)
    
    def test_model_selection(self):
        """Test intelligent model selection"""
        text = "This is a simple English sentence."
        best_model = self.selector.select_best_model(text)
        
        self.assertIsInstance(best_model, str)
        self.assertIn(best_model, self.selector.model_performance.keys())
    
    def test_multilingual_text_selection(self):
        """Test model selection for multilingual text"""
        text = "Le Lorem ipsum dolor sit amet."  # French-looking text
        best_model = self.selector.select_best_model(
            text, 
            requirements={"multilingual_importance": 0.9}
        )
        
        # Should prefer multilingual model when multilingual importance is high
        self.assertIsInstance(best_model, str)


def run_tests():
    """Run all tests in the module"""
    print("Running tests for Module 3: Embedding Models")
    print("=" * 50)
    
    # Create test suites
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestEmbeddingModelsModule)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestMultiModelEnsemble)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestDomainSpecificEmbedders)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestIntelligentSelector)
    
    # Combine suites
    full_suite = unittest.TestSuite([suite1, suite2, suite3, suite4])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(full_suite)
    
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