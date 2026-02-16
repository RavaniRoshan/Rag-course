"""
Module 4: Vector Databases
Tests for implementation examples

This module contains tests for the vector database implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch
import numpy as np

from example import ChromaDBWrapper, FAISSWrapper, VectorDBComparison


class TestChromaDBWrapper(unittest.TestCase):
    """Test ChromaDB wrapper implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.chroma_db = ChromaDBWrapper(
            collection_name="test_collection",
            persist_directory=self.test_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove test directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ChromaDB initialization"""
        self.assertIsNotNone(self.chroma_db.client)
        self.assertIsNotNone(self.chroma_db.collection)
        self.assertEqual(self.chroma_db.dimension, 384)  # all-MiniLM-L6-v2 dimension
    
    def test_add_embeddings(self):
        """Test adding embeddings to ChromaDB"""
        texts = [
            "This is a test document",
            "Another test document",
            "A third test document"
        ]
        
        ids = self.chroma_db.add_embeddings(texts)
        
        # Check that IDs were returned
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)  # All IDs should be unique
        
        # Check that the collection has the correct count
        self.assertEqual(self.chroma_db.get_count(), 3)
    
    def test_add_embeddings_with_metadata(self):
        """Test adding embeddings with metadata"""
        texts = ["Test document 1", "Test document 2"]
        metadata = [
            {"source": "web", "category": "news"},
            {"source": "book", "category": "fiction"}
        ]
        
        ids = self.chroma_db.add_embeddings(texts, metadata)
        
        # Check that IDs were returned
        self.assertEqual(len(ids), 2)
        
        # Check that the collection has the correct count
        self.assertEqual(self.chroma_db.get_count(), 5)  # Including previous docs
    
    def test_search(self):
        """Test similarity search in ChromaDB"""
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Natural language processing deals with text analysis"
        ]
        
        self.chroma_db.add_embeddings(texts)
        
        # Perform search
        results = self.chroma_db.search("artificial intelligence", k=2)
        
        # Check results format
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('id', result)
            self.assertIn('text', result)
            self.assertIn('metadata', result)
            self.assertIn('distance', result)
    
    def test_empty_search(self):
        """Test search on empty collection"""
        results = self.chroma_db.search("test query", k=5)
        self.assertEqual(len(results), 0)


class TestFAISSWrapper(unittest.TestCase):
    """Test FAISS wrapper implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.faiss_flat = FAISSWrapper(dimension=384, index_type="Flat")
        self.faiss_hnsw = FAISSWrapper(dimension=384, index_type="HNSW")
    
    def test_initialization(self):
        """Test FAISS initialization"""
        self.assertIsNotNone(self.faiss_flat.index)
        self.assertIsNotNone(self.faiss_hnsw.index)
        self.assertEqual(self.faiss_flat.dimension, 384)
        self.assertEqual(self.faiss_hnsw.dimension, 384)
    
    def test_add_embeddings_flat(self):
        """Test adding embeddings to FAISS Flat"""
        texts = [
            "This is a test document",
            "Another test document",
            "A third test document"
        ]
        
        ids = self.faiss_flat.add_embeddings(texts)
        
        # Check that IDs were returned
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)  # All IDs should be unique
        self.assertEqual(len(self.faiss_flat.texts), 3)
        self.assertEqual(len(self.faiss_flat.ids), 3)
        
        # Check that the index has vectors
        self.assertEqual(self.faiss_flat.index.ntotal, 3)
    
    def test_add_embeddings_hnsw(self):
        """Test adding embeddings to FAISS HNSW"""
        texts = [
            "This is a test document",
            "Another test document"
        ]
        
        ids = self.faiss_hnsw.add_embeddings(texts)
        
        # Check that IDs were returned
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)  # All IDs should be unique
        self.assertEqual(len(self.faiss_hnsw.texts), 2)
        self.assertEqual(len(self.faiss_hnsw.ids), 2)
        
        # Check that the index has vectors
        self.assertEqual(self.faiss_hnsw.index.ntotal, 2)
    
    def test_add_embeddings_with_metadata(self):
        """Test adding embeddings with metadata to FAISS"""
        texts = ["Test document 1", "Test document 2"]
        metadata = [
            {"source": "web", "category": "news"},
            {"source": "book", "category": "fiction"}
        ]
        
        ids = self.faiss_flat.add_embeddings(texts, metadata)
        
        # Check that IDs were returned
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(self.faiss_flat.texts), 2)
        self.assertEqual(len(self.faiss_flat.metadata), 2)
        
        # Check metadata was stored correctly
        self.assertEqual(self.faiss_flat.metadata[0]["source"], "web")
        self.assertEqual(self.faiss_flat.metadata[1]["category"], "fiction")
    
    def test_search_flat(self):
        """Test similarity search in FAISS Flat"""
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Natural language processing deals with text analysis"
        ]
        
        self.faiss_flat.add_embeddings(texts)
        
        # Perform search
        results = self.faiss_flat.search("artificial intelligence", k=2)
        
        # Check results format
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('id', result)
            self.assertIn('text', result)
            self.assertIn('metadata', result)
            self.assertIn('score', result)
    
    def test_search_hnsw(self):
        """Test similarity search in FAISS HNSW"""
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
            "Natural language processing deals with text analysis"
        ]
        
        self.faiss_hnsw.add_embeddings(texts)
        
        # Perform search
        results = self.faiss_hnsw.search("artificial intelligence", k=2)
        
        # Check results format
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('id', result)
            self.assertIn('text', result)
            self.assertIn('metadata', result)
            self.assertIn('score', result)
    
    def test_save_and_load(self):
        """Test saving and loading FAISS index"""
        texts = [
            "This is a test document",
            "Another test document"
        ]
        
        self.faiss_flat.add_embeddings(texts)
        
        # Save index
        temp_file = tempfile.mktemp()
        self.faiss_flat.save_index(temp_file)
        
        # Create new instance and load
        new_faiss = FAISSWrapper(dimension=384, index_type="Flat")
        new_faiss.load_index(temp_file)
        
        # Check that data was loaded correctly
        self.assertEqual(len(new_faiss.texts), 2)
        self.assertEqual(new_faiss.index.ntotal, 2)
        
        # Perform search on loaded index
        results = new_faiss.search("test", k=1)
        self.assertEqual(len(results), 1)
        
        # Clean up temp file
        os.remove(f"{temp_file}.faiss")
        os.remove(f"{temp_file}.pkl")


class TestVectorDBComparison(unittest.TestCase):
    """Test vector database comparison functionality"""
    
    def test_initialization(self):
        """Test VectorDBComparison initialization"""
        comparison = VectorDBComparison()
        self.assertIsNotNone(comparison.sample_texts)
        self.assertEqual(len(comparison.sample_texts), 12)
    
    def test_benchmark_db(self):
        """Test database benchmarking"""
        comparison = VectorDBComparison()
        chroma_db = ChromaDBWrapper(
            collection_name="benchmark_test",
            persist_directory=tempfile.mkdtemp()
        )
        
        results = comparison.benchmark_db(chroma_db, "ChromaDB")
        
        # Check results format
        self.assertIn('insertion_time', results)
        self.assertIn('search_time', results)
        self.assertIn('batch_search_time', results)
        self.assertIn('results_count', results)
        self.assertIn('name', results)
        self.assertEqual(results['name'], 'ChromaDB')
    
    def test_compare_all_dbs(self):
        """Test comparison of all databases"""
        comparison = VectorDBComparison()
        results = comparison.compare_all_dbs()
        
        # Should have results for ChromaDB, FAISS Flat, FAISS IVF, and FAISS HNSW
        self.assertEqual(len(results), 4)
        
        # Check that each result has the required keys
        for result in results:
            self.assertIn('insertion_time', result)
            self.assertIn('search_time', result)
            self.assertIn('batch_search_time', result)
            self.assertIn('results_count', result)
            self.assertIn('name', result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_texts(self):
        """Test behavior with empty text lists"""
        chroma_db = ChromaDBWrapper(collection_name="edge_test")
        
        # Adding empty list should not fail
        ids = chroma_db.add_embeddings([])
        self.assertEqual(len(ids), 0)
        
        # Searching empty collection should return empty results
        results = chroma_db.search("test query", k=5)
        self.assertEqual(len(results), 0)
    
    def test_single_text(self):
        """Test behavior with single text"""
        chroma_db = ChromaDBWrapper(collection_name="single_test")
        
        ids = chroma_db.add_embeddings(["Single text document"])
        self.assertEqual(len(ids), 1)
        
        results = chroma_db.search("single", k=5)
        self.assertEqual(len(results), 1)
    
    def test_long_text(self):
        """Test behavior with long text"""
        chroma_db = ChromaDBWrapper(collection_name="long_text_test")
        
        long_text = "This is a very long text document. " * 100
        ids = chroma_db.add_embeddings([long_text])
        self.assertEqual(len(ids), 1)
        
        results = chroma_db.search("long text", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("This is a very long text document", results[0]['text'])
    
    def test_special_characters(self):
        """Test behavior with special characters"""
        chroma_db = ChromaDBWrapper(collection_name="special_chars_test")
        
        texts = ["Text with special chars: !@#$%^&*()"]
        ids = chroma_db.add_embeddings(texts)
        self.assertEqual(len(ids), 1)
        
        results = chroma_db.search("special chars", k=1)
        self.assertEqual(len(results), 1)


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    # Test with larger dataset
    comparison = VectorDBComparison()
    large_texts = comparison.sample_texts * 10  # Repeat 10 times for larger dataset
    
    chroma_db = ChromaDBWrapper(collection_name="perf_test")
    
    # Time the insertion
    import time
    start = time.time()
    chroma_db.add_embeddings(large_texts)
    insertion_time = time.time() - start
    
    # Time a few searches
    search_times = []
    for _ in range(5):
        start = time.time()
        chroma_db.search("artificial intelligence", k=5)
        search_times.append(time.time() - start)
    
    avg_search_time = sum(search_times) / len(search_times)
    
    print(f"Performance results:")
    print(f"  Inserted {len(large_texts)} documents in {insertion_time:.3f}s")
    print(f"  Average search time: {avg_search_time:.3f}s")
    
    # Verify insertion worked
    assert chroma_db.get_count() == len(large_texts), f"Expected {len(large_texts)} documents, got {chroma_db.get_count()}"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test different index types in FAISS
    print("Testing different FAISS index types...")
    
    texts = [
        "Machine learning is a fascinating field",
        "Deep learning is a subset of machine learning",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret visual information"
    ]
    
    # Test Flat index
    flat_db = FAISSWrapper(index_type="Flat")
    flat_db.add_embeddings(texts)
    flat_results = flat_db.search("AI and machine learning", k=3)
    print(f"FAISS Flat results: {len(flat_results)}")
    
    # Test HNSW index
    hnsw_db = FAISSWrapper(index_type="HNSW")
    hnsw_db.add_embeddings(texts)
    hnsw_results = hnsw_db.search("AI and machine learning", k=3)
    print(f"FAISS HNSW results: {len(hnsw_results)}")
    
    # Results should be similar but not necessarily identical due to algorithm differences
    print("Index types tested successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()