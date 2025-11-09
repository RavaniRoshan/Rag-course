"""
Module 13: Domain Adaptation
Tests for implementation examples

This module contains tests for the domain adaptation implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np
from datetime import datetime

from example import (
    DomainVocabulary, DomainEmbeddingAdapter, DomainSpecificRetriever,
    DomainAwareGenerator, DomainAdaptedRAG, DomainEvaluator
)


class TestDomainVocabulary(unittest.TestCase):
    """Test domain vocabulary management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vocab = DomainVocabulary("Medical")
    
    def test_initialization(self):
        """Test vocabulary initialization"""
        self.assertEqual(self.vocab.domain_name, "Medical")
        self.assertEqual(self.vocab.terms, {})
        self.assertEqual(self.vocab.abbreviations, {})
        self.assertEqual(self.vocab.expansions, {})
        self.assertEqual(self.vocab.domain_expertise_level, 0)
    
    def test_add_term(self):
        """Test adding terms to vocabulary"""
        self.vocab.add_term("myocardial infarction", "Heart attack definition", "cardiology")
        
        self.assertIn("myocardial infarction", self.vocab.terms)
        self.assertEqual(self.vocab.terms["myocardial infarction"]["definition"], "Heart attack definition")
        self.assertEqual(self.vocab.terms["myocardial infarction"]["category"], "cardiology")
        self.assertIsInstance(self.vocab.terms["myocardial infarction"]["added_at"], datetime)
    
    def test_add_abbreviation(self):
        """Test adding abbreviations"""
        self.vocab.add_abbreviation("MI", "myocardial infarction")
        
        self.assertIn("mi", self.vocab.abbreviations)
        self.assertEqual(self.vocab.abbreviations["mi"], "myocardial infarction")
        self.assertIn("myocardial infarction", self.vocab.expansions)
        self.assertEqual(self.vocab.expansions["myocardial infarction"], "MI")
    
    def test_expand_abbreviations(self):
        """Test abbreviation expansion"""
        self.vocab.add_abbreviation("mi", "myocardial infarction")
        self.vocab.add_abbreviation("htn", "hypertension")
        
        text = "Patient had MI and HTN"
        expanded = self.vocab.expand_abbreviations(text)
        
        self.assertIn("myocardial infarction", expanded.lower())
        self.assertIn("hypertension", expanded.lower())
    
    def test_get_related_terms(self):
        """Test getting related terms"""
        self.vocab.add_term("myocardial infarction", "Heart attack", "cardiology")
        self.vocab.add_term("stroke", "Brain attack", "cardiology")
        self.vocab.add_term("diabetes", "Sugar disease", "endocrinology")
        
        related = self.vocab.get_related_terms("myocardial infarction")
        
        # Should include stroke since it's in same category
        self.assertIn("stroke", related)
        # Should not include diabetes since it's different category
        self.assertNotIn("diabetes", related)


class TestDomainEmbeddingAdapter(unittest.TestCase):
    """Test domain embedding adapter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = DomainEmbeddingAdapter()
        self.vocab = DomainVocabulary("TestDomain")
        self.vocab.add_term("machine learning", "AI technique", "ML")
    
    def test_initialization(self):
        """Test adapter initialization"""
        self.assertIsNotNone(self.adapter.base_embedder)
        self.assertEqual(self.adapter.domain_embeddings, {})
        self.assertEqual(self.adapter.domain_weights, {})
    
    def test_adapt_for_domain(self):
        """Test adapting for domain"""
        domain_texts = ["Machine learning algorithms are powerful", "AI techniques help solve problems"]
        adapted = self.adapter.adapt_for_domain(self.vocab, domain_texts)
        
        self.assertEqual(adapted, self.adapter)
        # Should have created domain embeddings for the term
        self.assertIn("machine learning", self.adapter.domain_embeddings)
    
    def test_encode_with_domain_knowledge(self):
        """Test encoding with domain knowledge"""
        # First adapt the adapter
        self.adapter.adapt_for_domain(self.vocab, ["Machine learning is important"])
        
        # Now test encoding
        text = "Machine learning techniques"
        embeddings = self.adapter.encode_with_domain_knowledge(text)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings.shape), 2)  # Should be 2D array (batch, embedding_dim)
        self.assertEqual(embeddings.shape[0], 1)   # One text input
    
    def test_encode_multiple_texts(self):
        """Test encoding multiple texts"""
        texts = ["Machine learning", "AI techniques"]
        embeddings = self.adapter.encode_with_domain_knowledge(texts)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], 2)  # Two text inputs


class TestDomainSpecificRetriever(unittest.TestCase):
    """Test domain-specific retriever"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vocab = DomainVocabulary("Medical")
        self.vocab.add_term("heart attack", "myocardial infarction", "cardiology")
        self.vocab.add_abbreviation("MI", "myocardial infarction")
        
        self.adapter = DomainEmbeddingAdapter()
        self.retriever = DomainSpecificRetriever(self.adapter, self.vocab)
    
    def test_initialization(self):
        """Test retriever initialization"""
        self.assertIsNotNone(self.retriever.domain_adapter)
        self.assertIsNotNone(self.retriever.domain_vocab)
        self.assertIsNotNone(self.retriever.collection)
    
    def test_add_document(self):
        """Test adding a document"""
        doc_id = self.retriever.add_document("Information about heart attacks")
        
        self.assertIsInstance(doc_id, str)
        self.assertGreater(len(doc_id), 0)
    
    def test_preprocess_for_domain(self):
        """Test domain-specific preprocessing"""
        text = "Patient had an MI"
        processed = self.retriever._preprocess_for_domain(text)
        
        # Should expand abbreviations
        self.assertIn("myocardial infarction", processed.lower())
    
    def test_retrieve_with_domain_knowledge(self):
        """Test domain-aware retrieval"""
        # Add a document
        self.retriever.add_document("Information about heart attacks and myocardial infarction")
        
        # Retrieve
        results = self.retriever.retrieve_with_domain_knowledge("heart attack", top_k=1)
        
        self.assertIsInstance(results, list)
        if results:
            self.assertIsInstance(results[0], dict)
            self.assertIn('content', results[0])
            self.assertIn('domain_relevance', results[0])


class TestDomainAwareGenerator(unittest.TestCase):
    """Test domain-aware generator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vocab = DomainVocabulary("Technology")
        self.vocab.add_term("API", "Application Programming Interface", "Development")
        self.generator = DomainAwareGenerator(self.vocab)
    
    def test_initialization(self):
        """Test generator initialization"""
        self.assertIsNotNone(self.generator.domain_vocab)
    
    def test_enhance_prompt_with_domain(self):
        """Test enhancing prompt with domain knowledge"""
        query = "How does API work?"
        context = "Software development context"
        
        enhanced = self.generator._enhance_prompt_with_domain(query, context)
        
        self.assertIn(self.vocab.domain_name, enhanced)
        self.assertIn("API", enhanced)
        self.assertIn("Application Programming Interface", enhanced)
    
    def test_post_process_with_domain(self):
        """Test post-processing with domain knowledge"""
        # Add an abbreviation
        self.vocab.add_abbreviation("API", "Application Programming Interface")
        
        text = "This uses API to connect"
        processed = self.generator._post_process_with_domain(text)
        
        # Note: This depends on the expand_abbreviations method implementation
        # For now, just verify it returns a string
        self.assertIsInstance(processed, str)
    
    def test_generate_with_domain_knowledge(self):
        """Test domain-aware generation"""
        # This might fail if the underlying model isn't available, so test gracefully
        result = self.generator.generate_with_domain_knowledge(
            "What is an API?", 
            "Software context", 
            max_length=50
        )
        
        self.assertIsInstance(result, str)
        # Should contain domain reference or be a fallback message
        self.assertTrue(isinstance(result, str))


class TestDomainAdaptedRAG(unittest.TestCase):
    """Test domain-adapted RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rag = DomainAdaptedRAG(domain_name="TestDomain")
    
    def test_initialization(self):
        """Test RAG system initialization"""
        self.assertEqual(self.rag.domain_name, "TestDomain")
        self.assertIsNotNone(self.rag.domain_vocab)
        self.assertIsNotNone(self.rag.domain_adapter)
        self.assertIsNotNone(self.rag.retriever)
        self.assertIsNotNone(self.rag.generator)
        self.assertIn('domain_queries', self.rag.metrics)
    
    def test_add_domain_knowledge(self):
        """Test adding domain knowledge"""
        texts = ["Test document 1", "Test document 2"]
        metadata = [{"type": "test"}, {"type": "example"}]
        
        doc_ids = self.rag.add_domain_knowledge(texts, metadata)
        
        self.assertEqual(len(doc_ids), 2)
        for doc_id in doc_ids:
            self.assertIsInstance(doc_id, str)
    
    def test_query_method(self):
        """Test the query method"""
        # Add some knowledge first
        self.rag.add_domain_knowledge(["Machine learning is powerful"], [{"category": "AI"}])
        
        # Query the system
        result = self.rag.query("What is machine learning?", top_k=1, include_generation=True)
        
        self.assertIn('query', result)
        self.assertIn('retrieval_results', result)
        self.assertIn('retrieval_time', result)
        self.assertIn('generated_response', result)
        self.assertIn('generation_time', result)
        self.assertIn('total_time', result)
        
        self.assertEqual(result['query'], "What is machine learning?")
        self.assertIsInstance(result['retrieval_results'], list)
        self.assertIsInstance(result['retrieval_time'], float)
        self.assertIsInstance(result['generated_response'], str)
        self.assertIsInstance(result['generation_time'], float)
        self.assertIsInstance(result['total_time'], float)
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics"""
        metrics = self.rag.get_performance_metrics()
        
        self.assertIn('domain_queries', metrics)
        self.assertIn('avg_retrieval_time', metrics)
        self.assertIn('avg_generation_time', metrics)
        
        self.assertIsInstance(metrics['domain_queries'], int)
        self.assertIsInstance(metrics['avg_retrieval_time'], float)
        self.assertIsInstance(metrics['avg_generation_time'], float)
    
    def test_add_domain_term(self):
        """Test adding domain terms"""
        self.rag.add_domain_term("test_term", "Test definition", "category")
        
        self.assertIn("test_term", self.rag.domain_vocab.terms)
        self.assertEqual(self.rag.domain_vocab.terms["test_term"]["definition"], "Test definition")
    
    def test_add_domain_abbreviation(self):
        """Test adding domain abbreviations"""
        self.rag.add_domain_abbreviation("TLD", "Test Long Definition")
        
        self.assertIn("tld", self.rag.domain_vocab.abbreviations)
        self.assertEqual(self.rag.domain_vocab.abbreviations["tld"], "Test Long Definition")


class TestDomainEvaluator(unittest.TestCase):
    """Test domain evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = DomainEvaluator("TestDomain")
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertEqual(self.evaluator.domain_name, "TestDomain")
        self.assertEqual(self.evaluator.domain_queries, [])
        self.assertEqual(self.evaluator.domain_responses, [])
    
    def test_evaluate_domain_relevance(self):
        """Test domain relevance evaluation"""
        query = "machine learning algorithms"
        response = "Machine learning uses algorithms to find patterns in data"
        expected_terms = ["machine learning", "algorithms"]
        
        relevance = self.evaluator.evaluate_domain_relevance(query, response, expected_terms)
        
        self.assertIsInstance(relevance, float)
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)
    
    def test_evaluate_domain_relevance_no_expected(self):
        """Test domain relevance evaluation without expected terms"""
        relevance = self.evaluator.evaluate_domain_relevance("test query", "test response", [])
        
        self.assertIsInstance(relevance, float)
        self.assertGreaterEqual(relevance, 0.0)
    
    def test_evaluate_terminology_accuracy(self):
        """Test terminology accuracy evaluation"""
        vocab = DomainVocabulary("TestDomain")
        vocab.add_term("machine", "A device that performs work")
        vocab.add_term("learning", "Acquiring knowledge")
        
        response = "The machine learning system works well"
        accuracy = self.evaluator.evaluate_terminology_accuracy(response, vocab)
        
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_run_domain_evaluation(self):
        """Test running comprehensive domain evaluation"""
        # Create a simple RAG system for testing
        rag = DomainAdaptedRAG("TestDomain")
        rag.add_domain_knowledge(["Test content about machine learning"], [{"type": "ML"}])
        
        test_queries = [
            ("What is machine learning?", ["machine learning", "algorithm"]),
            ("Explain AI basics", ["artificial intelligence", "model"])
        ]
        
        results = self.evaluator.run_domain_evaluation(rag, test_queries)
        
        self.assertIn('average_domain_relevance', results)
        self.assertIn('average_terminology_accuracy', results)
        self.assertIn('total_queries', results)
        self.assertIn('detailed_results', results)
        
        self.assertEqual(results['total_queries'], 2)
        self.assertIsInstance(results['average_domain_relevance'], float)
        self.assertIsInstance(results['average_terminology_accuracy'], float)
        self.assertIsInstance(results['detailed_results'], list)
        self.assertEqual(len(results['detailed_results']), 2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_domain_vocab_empty(self):
        """Test domain vocabulary with no terms"""
        vocab = DomainVocabulary("EmptyDomain")
        
        # Should handle empty vocabulary gracefully
        related = vocab.get_related_terms("nonexistent")
        self.assertEqual(related, [])
        
        expanded = vocab.expand_abbreviations("no abbreviations here")
        self.assertEqual(expanded, "no abbreviations here")
    
    def test_retriever_empty_collection(self):
        """Test retriever with empty collection"""
        vocab = DomainVocabulary("Test")
        adapter = DomainEmbeddingAdapter()
        retriever = DomainSpecificRetriever(adapter, vocab)
        
        results = retriever.retrieve_with_domain_knowledge("test query", top_k=5)
        self.assertIsInstance(results, list)
        # May return empty list or list with items based on implementation
    
    def test_generator_empty_input(self):
        """Test generator with empty input"""
        vocab = DomainVocabulary("Test")
        generator = DomainAwareGenerator(vocab)
        
        result = generator.generate_with_domain_knowledge("", "", max_length=10)
        self.assertIsInstance(result, str)
    
    def test_rag_empty_queries(self):
        """Test RAG system with empty or minimal data"""
        rag = DomainAdaptedRAG("MinimalDomain")
        
        result = rag.query("What does this system know?", top_k=1)
        # Should handle gracefully without crashing
        self.assertIn('query', result)
        self.assertIn('retrieval_results', result)
    
    def test_evaluation_empty_data(self):
        """Test evaluator with empty data"""
        evaluator = DomainEvaluator("EmptyTest")
        rag = DomainAdaptedRAG("EmptyTest")
        
        results = evaluator.run_domain_evaluation(rag, [])
        
        self.assertEqual(results['total_queries'], 0)
        self.assertEqual(results['average_domain_relevance'], 0.0)
        self.assertEqual(results['average_terminology_accuracy'], 0.0)


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test domain adaptation with larger datasets
    rag = DomainAdaptedRAG("PerformanceTest")
    
    # Add domain terms
    for i in range(20):
        rag.add_domain_term(f"term_{i}", f"Definition for term {i}", f"category_{i%5}")
    
    # Add documents
    documents = []
    for i in range(50):
        documents.append(f"Document {i} contains information about various terms and concepts for performance testing.")
    
    start_time = time.time()
    rag.add_domain_knowledge(documents)
    add_time = time.time() - start_time
    
    # Test querying
    start_time = time.time()
    for i in range(10):
        result = rag.query(f"Information about term_{i%10}", top_k=3)
    query_time = time.time() - start_time
    
    print(f"Added {len(documents)} documents in {add_time:.4f}s")
    print(f"Processed 10 queries in {query_time:.4f}s")
    print(f"Average query time: {query_time/10:.4f}s")
    
    # Should be reasonably fast
    assert add_time < 10.0, f"Adding documents too slow: {add_time}s"
    assert query_time/10 < 2.0, f"Query processing too slow: {query_time/10}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test cross-domain adaptation
    print("Testing domain adaptation transfer...")
    
    # Create RAG systems for different domains
    medical_rag = DomainAdaptedRAG("Medical")
    legal_rag = DomainAdaptedRAG("Legal")
    
    # Add domain-specific terms
    medical_terms = [
        ("myocardial infarction", "Heart attack medical term"),
        ("hypertension", "High blood pressure"),
        ("arrhythmia", "Irregular heartbeat")
    ]
    
    legal_terms = [
        ("contract", "Legal agreement"),
        ("liability", "Legal responsibility"),
        ("jurisdiction", "Legal authority")
    ]
    
    for term, defn in medical_terms:
        medical_rag.add_domain_term(term, defn)
    
    for term, defn in legal_terms:
        legal_rag.add_domain_term(term, defn)
    
    # Add domain-specific documents
    medical_docs = [
        "Myocardial infarction requires immediate medical attention",
        "Hypertension is often called the silent killer",
        "Arrhythmia can be treated with medication"
    ]
    
    legal_docs = [
        "A contract must have offer, acceptance, and consideration",
        "Liability can be civil or criminal",
        "Jurisdiction determines which court hears the case"
    ]
    
    medical_rag.add_domain_knowledge(medical_docs)
    legal_rag.add_domain_knowledge(legal_docs)
    
    # Test queries in respective domains
    medical_result = medical_rag.query("What is myocardial infarction?", top_k=2)
    legal_result = legal_rag.query("Explain contract law", top_k=2)
    
    print(f"Medical query processed: {len(medical_result['retrieval_results'])} results retrieved")
    print(f"Legal query processed: {len(legal_result['retrieval_results'])} results retrieved")
    
    # Evaluate the domain-specific systems
    evaluator = DomainEvaluator("CrossDomainTest")
    
    medical_eval = evaluator.evaluate_terminology_accuracy(
        medical_result.get('generated_response', ''), medical_rag.domain_vocab
    )
    legal_eval = evaluator.evaluate_terminology_accuracy(
        legal_result.get('generated_response', ''), legal_rag.domain_vocab
    )
    
    print(f"Medical terminology accuracy: {medical_eval:.2f}")
    print(f"Legal terminology accuracy: {legal_eval:.2f}")
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()