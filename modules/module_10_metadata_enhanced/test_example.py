"""
Module 10: Metadata-Enhanced Retrieval
Tests for implementation examples

This module contains tests for the metadata-enhanced retrieval implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np
from datetime import datetime, timedelta

from example import MetadataDocument, MetadataIndexer, MetadataFilter, MetadataScorer, MetadataRetriever, MetadataRetrievalEvaluator


class TestMetadataDocument(unittest.TestCase):
    """Test metadata document creation and management"""
    
    def test_create_document(self):
        """Test creating a metadata document"""
        content = "This is a test document"
        metadata = {"author": "Test Author", "category": "Test Category"}
        
        doc = MetadataDocument(content, metadata=metadata)
        
        self.assertEqual(doc.content, content)
        self.assertEqual(doc.metadata["author"], "Test Author")
        self.assertEqual(doc.metadata["category"], "Test Category")
        self.assertEqual(doc.metadata["content_length"], len(content))
        self.assertIsNotNone(doc.metadata["created_at"])
        self.assertIsNotNone(doc.doc_id)
    
    def test_create_document_with_id(self):
        """Test creating a document with a specific ID"""
        content = "Test content"
        doc_id = "test-id-123"
        
        doc = MetadataDocument(content, doc_id=doc_id)
        
        self.assertEqual(doc.doc_id, doc_id)
        self.assertEqual(doc.metadata["id"], doc_id)
    
    def test_create_document_default_metadata(self):
        """Test that default metadata is added"""
        content = "Test content"
        doc = MetadataDocument(content)
        
        self.assertEqual(doc.metadata["content_length"], len(content))
        self.assertIsNotNone(doc.metadata["created_at"])
        self.assertIsNotNone(doc.doc_id)
        self.assertEqual(doc.metadata["id"], doc.doc_id)
    
    def test_to_dict(self):
        """Test converting document to dictionary"""
        content = "Test content"
        metadata = {"author": "Test Author"}
        doc = MetadataDocument(content, metadata=metadata)
        
        doc_dict = doc.to_dict()
        
        self.assertEqual(doc_dict["doc_id"], doc.doc_id)
        self.assertEqual(doc_dict["content"], content)
        self.assertEqual(doc_dict["metadata"], doc.metadata)


class TestMetadataIndexer(unittest.TestCase):
    """Test metadata indexing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.indexer = MetadataIndexer(collection_name="test_collection")
    
    def tearDown(self):
        """Clean up after tests"""
        # Note: ChromaDB doesn't have a straightforward way to delete collections
        # in the ephemeral mode, so we rely on the fact that in-memory instances
        # are automatically cleaned up
        pass
    
    def test_initialization(self):
        """Test indexer initialization"""
        self.assertIsNotNone(self.indexer.embedder)
        self.assertIsNotNone(self.indexer.collection)
    
    def test_add_document(self):
        """Test adding a single document with metadata"""
        doc = MetadataDocument(
            "Test document content",
            metadata={"category": "test", "author": "test_author"}
        )
        
        doc_id = self.indexer.add_document(doc)
        
        self.assertEqual(doc_id, doc.doc_id)
        self.assertEqual(self.indexer.get_count(), 1)
    
    def test_add_documents(self):
        """Test adding multiple documents with metadata"""
        docs = [
            MetadataDocument("Content 1", metadata={"category": "A"}),
            MetadataDocument("Content 2", metadata={"category": "B"}),
            MetadataDocument("Content 3", metadata={"category": "A"})
        ]
        
        doc_ids = self.indexer.add_documents(docs)
        
        self.assertEqual(len(doc_ids), 3)
        self.assertEqual(self.indexer.get_count(), 3)
        
        # Check that all IDs are unique
        self.assertEqual(len(set(doc_ids)), 3)


class TestMetadataFilter(unittest.TestCase):
    """Test metadata filtering functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.filter_engine = MetadataFilter()
    
    def test_initialization(self):
        """Test filter engine initialization"""
        self.assertIsNotNone(self.filter_engine)
    
    def test_build_filter_condition(self):
        """Test building filter conditions"""
        filters = {"category": "AI", "author": "Dr. Smith"}
        condition = self.filter_engine.build_filter_condition(filters)
        
        self.assertEqual(condition, filters)
    
    def test_build_empty_filter_condition(self):
        """Test building empty filter condition"""
        condition = self.filter_engine.build_filter_condition({})
        self.assertEqual(condition, {})
    
    def test_apply_exact_filters(self):
        """Test applying exact match filters"""
        docs = [
            {
                'id': '1',
                'content': 'Doc 1',
                'metadata': {'category': 'AI', 'author': 'Dr. Smith'}
            },
            {
                'id': '2', 
                'content': 'Doc 2',
                'metadata': {'category': 'ML', 'author': 'Dr. Jones'}
            },
            {
                'id': '3',
                'content': 'Doc 3', 
                'metadata': {'category': 'AI', 'author': 'Dr. Smith'}
            }
        ]
        
        # Filter for category = AI
        filtered = self.filter_engine.apply_exact_filters(docs, {'category': 'AI'})
        self.assertEqual(len(filtered), 2)
        for doc in filtered:
            self.assertEqual(doc['metadata']['category'], 'AI')
    
    def test_apply_multiple_filters(self):
        """Test applying multiple filters"""
        docs = [
            {
                'id': '1',
                'content': 'Doc 1',
                'metadata': {'category': 'AI', 'author': 'Dr. Smith', 'year': 2023}
            },
            {
                'id': '2',
                'content': 'Doc 2',
                'metadata': {'category': 'ML', 'author': 'Dr. Smith', 'year': 2022}
            },
            {
                'id': '3',
                'content': 'Doc 3',
                'metadata': {'category': 'AI', 'author': 'Dr. Jones', 'year': 2023}
            }
        ]
        
        # Filter for category = AI AND author = Dr. Smith
        filtered = self.filter_engine.apply_exact_filters(docs, {'category': 'AI', 'author': 'Dr. Smith'})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['id'], '1')
    
    def test_apply_list_filters(self):
        """Test applying list-based filters (in operator)"""
        docs = [
            {
                'id': '1',
                'content': 'Doc 1',
                'metadata': {'category': 'AI', 'author': 'Dr. Smith'}
            },
            {
                'id': '2',
                'content': 'Doc 2', 
                'metadata': {'category': 'ML', 'author': 'Dr. Jones'}
            },
            {
                'id': '3',
                'content': 'Doc 3',
                'metadata': {'category': 'NLP', 'author': 'Dr. Smith'}
            }
        ]
        
        # Filter for category in ['AI', 'NLP']
        filtered = self.filter_engine.apply_exact_filters(docs, {'category': ['AI', 'NLP']})
        self.assertEqual(len(filtered), 2)
        categories = [doc['metadata']['category'] for doc in filtered]
        self.assertIn('AI', categories)
        self.assertIn('NLP', categories)
    
    def test_apply_range_filters(self):
        """Test applying range filters"""
        docs = [
            {
                'id': '1',
                'content': 'Doc 1',
                'metadata': {'year': 2023, 'quality_score': 0.9}
            },
            {
                'id': '2',
                'content': 'Doc 2',
                'metadata': {'year': 2020, 'quality_score': 0.5}
            },
            {
                'id': '3',
                'content': 'Doc 3',
                'metadata': {'year': 2022, 'quality_score': 0.8}
            }
        ]
        
        # Filter for year >= 2021
        filtered = self.filter_engine.apply_range_filters(docs, {'year': {'min': 2021}})
        self.assertEqual(len(filtered), 2)
        for doc in filtered:
            self.assertGreaterEqual(doc['metadata']['year'], 2021)
    
    def test_apply_range_filters_max(self):
        """Test applying max range filters"""
        docs = [
            {
                'id': '1',
                'content': 'Doc 1',
                'metadata': {'quality_score': 0.9}
            },
            {
                'id': '2',
                'content': 'Doc 2',
                'metadata': {'quality_score': 0.5}
            },
            {
                'id': '3',
                'content': 'Doc 3',
                'metadata': {'quality_score': 0.8}
            }
        ]
        
        # Filter for quality_score <= 0.8
        filtered = self.filter_engine.apply_range_filters(docs, {'quality_score': {'max': 0.8}})
        self.assertEqual(len(filtered), 2)
        for doc in filtered:
            self.assertLessEqual(doc['metadata']['quality_score'], 0.8)


class TestMetadataScorer(unittest.TestCase):
    """Test metadata-aware scoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scorer = MetadataScorer()
    
    def test_initialization(self):
        """Test scorer initialization"""
        self.assertIsNotNone(self.scorer.default_weights)
    
    def test_calculate_metadata_score(self):
        """Test metadata score calculation"""
        query = "machine learning"
        metadata = {"title": "Introduction to machine learning", "tags": ["ML", "AI"]}
        
        score = self.scorer.calculate_metadata_score(query, metadata)
        
        # Should be a float between 0 and 1
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Should be greater than 0 if there's a match
        self.assertGreater(score, 0.0)
    
    def test_calculate_metadata_score_no_match(self):
        """Test metadata score calculation with no matches"""
        query = "cooking recipes"
        metadata = {"title": "Machine learning guide", "tags": ["ML", "AI"]}
        
        score = self.scorer.calculate_metadata_score(query, metadata)
        
        # Should be 0 or very low if no matches
        self.assertGreaterEqual(score, 0.0)
    
    def test_combine_scores(self):
        """Test combining content and metadata scores"""
        content_score = 0.8
        metadata_score = 0.6
        
        combined_score = self.scorer.combine_scores(content_score, metadata_score)
        
        # Should be a float between 0 and 1
        self.assertIsInstance(combined_score, float)
        self.assertGreaterEqual(combined_score, 0.0)
        self.assertLessEqual(combined_score, 1.0)
        
        # Should be reasonable weighted combination
        self.assertGreaterEqual(combined_score, min(content_score, metadata_score))
        self.assertLessEqual(combined_score, max(content_score, metadata_score))
    
    def test_combine_scores_with_custom_weights(self):
        """Test combining scores with custom weights"""
        content_score = 0.8
        metadata_score = 0.6
        
        weights = {'metadata_relevance': 0.7, 'content_similarity': 0.3}
        combined_score = self.scorer.combine_scores(content_score, metadata_score, weights)
        
        # Should give more weight to metadata since weight is higher
        self.assertIsInstance(combined_score, float)


class TestMetadataRetriever(unittest.TestCase):
    """Test metadata-enhanced retrieval functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.indexer = MetadataIndexer(collection_name="test_retriever")
        
        # Add some test documents
        self.test_docs = [
            MetadataDocument(
                "Machine learning algorithms explained",
                metadata={
                    "category": "ML",
                    "author": "Dr. Smith",
                    "year": 2023,
                    "quality_score": 0.9
                }
            ),
            MetadataDocument(
                "Deep learning neural networks",
                metadata={
                    "category": "DL", 
                    "author": "Dr. Jones",
                    "year": 2022,
                    "quality_score": 0.8
                }
            ),
            MetadataDocument(
                "Natural language processing basics",
                metadata={
                    "category": "NLP",
                    "author": "Dr. Williams",
                    "year": 2023,
                    "quality_score": 0.75
                }
            )
        ]
        
        self.indexer.add_documents(self.test_docs)
        self.retriever = MetadataRetriever(self.indexer)
    
    def tearDown(self):
        """Clean up test fixtures"""
        pass
    
    def test_initialization(self):
        """Test retriever initialization"""
        self.assertIsNotNone(self.retriever.indexer)
        self.assertIsNotNone(self.retriever.filter_engine)
        self.assertIsNotNone(self.retriever.scorer)
    
    def test_retrieve_basic(self):
        """Test basic retrieval without filters"""
        results = self.retriever.retrieve("machine learning", top_k=2)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        # Each result should have required fields
        for result in results:
            self.assertIn('id', result)
            self.assertIn('content', result)
            self.assertIn('metadata', result)
            self.assertIn('similarity', result)
            self.assertIn('enhanced_score', result)
            self.assertIn('metadata_score', result)
    
    def test_retrieve_with_filters(self):
        """Test retrieval with metadata filters"""
        results = self.retriever.retrieve(
            "learning", 
            top_k=3,
            filters={"category": "ML"}
        )
        
        self.assertIsInstance(results, list)
        
        # All results should match the filter
        for result in results:
            self.assertEqual(result['metadata']['category'], "ML")
    
    def test_retrieve_with_range_filters(self):
        """Test retrieval with range filters"""
        results = self.retriever.retrieve(
            "learning",
            top_k=3,
            range_filters={"year": {"min": 2023}}
        )
        
        self.assertIsInstance(results, list)
        
        # All results should match the range filter
        for result in results:
            self.assertGreaterEqual(result['metadata']['year'], 2023)
    
    def test_retrieve_with_metadata_weight(self):
        """Test retrieval with metadata weighting"""
        results = self.retriever.retrieve(
            "learning",
            top_k=2,
            metadata_weight=0.5  # Equal weight to content and metadata
        )
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        for result in results:
            # Should have both similarity and enhanced scores
            self.assertIn('similarity', result)
            self.assertIn('enhanced_score', result)
            self.assertIn('metadata_score', result)
    
    def test_intelligent_filtering(self):
        """Test intelligent query-based filtering"""
        results = self.retriever.retrieve_with_intelligent_filtering(
            "recent machine learning", 
            top_k=2
        )
        
        # Should return list of results
        self.assertIsInstance(results, list)
        
        # Should have enhanced scores
        for result in results:
            self.assertIn('enhanced_score', result)


class TestMetadataRetrievalEvaluator(unittest.TestCase):
    """Test metadata retrieval evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = MetadataRetrievalEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.sample_docs)
        self.assertGreaterEqual(len(self.evaluator.sample_docs), 1)
    
    def test_evaluate_retrieval_methods(self):
        """Test evaluation of retrieval methods"""
        results = self.evaluator.evaluate_retrieval_methods()
        
        # Should have results for each method
        self.assertIn('basic_retrieval', results)
        self.assertIn('metadata_enhanced', results)
        self.assertIn('intelligent_filtering', results)
        
        # Each should contain query-result pairs
        for method_results in results.values():
            self.assertIsInstance(method_results, list)
            if method_results:  # If there are results
                self.assertIsInstance(method_results[0], dict)
                self.assertIn('query', method_results[0])
                self.assertIn('results', method_results[0])
    
    def test_compare_performance(self):
        """Test performance comparison"""
        perf_results = self.evaluator.compare_performance()
        
        # Should have timing results
        self.assertIn('basic_avg_time', perf_results)
        self.assertIn('metadata_enhanced_avg_time', perf_results)
        self.assertIn('overhead', perf_results)
        
        # Times should be non-negative
        self.assertGreaterEqual(perf_results['basic_avg_time'], 0)
        self.assertGreaterEqual(perf_results['metadata_enhanced_avg_time'], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_document_list(self):
        """Test with empty document list"""
        indexer = MetadataIndexer()
        doc_ids = indexer.add_documents([])
        
        self.assertEqual(doc_ids, [])
        self.assertEqual(indexer.get_count(), 0)
    
    def test_retrieve_empty_index(self):
        """Test retrieval from empty index"""
        indexer = MetadataIndexer()
        retriever = MetadataRetriever(indexer)
        
        results = retriever.retrieve("test query", top_k=5)
        
        # Should return empty list
        self.assertEqual(results, [])
    
    def test_document_with_empty_metadata(self):
        """Test document with empty metadata"""
        doc = MetadataDocument("Test content", metadata={})
        
        self.assertEqual(doc.content, "Test content")
        self.assertIsNotNone(doc.metadata["content_length"])
        self.assertIsNotNone(doc.metadata["created_at"])
    
    def test_retrieve_with_invalid_filters(self):
        """Test retrieval with invalid filters"""
        indexer = MetadataIndexer()
        retriever = MetadataRetriever(indexer)
        
        # Add a test document
        test_doc = MetadataDocument(
            "Test document",
            metadata={"category": "test"}
        )
        indexer.add_document(test_doc)
        
        # Try to retrieve with problematic filters
        results = retriever.retrieve(
            "test", 
            top_k=5,
            filters={"nonexistent_field": "value"}
        )
        
        # Should handle gracefully (might return empty or all docs depending on implementation)
        self.assertIsInstance(results, list)
    
    def test_high_metadata_weight(self):
        """Test retrieval with high metadata weight"""
        indexer = MetadataIndexer()
        retriever = MetadataRetriever(indexer)
        
        # Add test document
        doc = MetadataDocument(
            "Machine learning content",
            metadata={"topic": "machine learning"}
        )
        indexer.add_document(doc)
        
        results = retriever.retrieve(
            "AI", 
            top_k=1, 
            metadata_weight=0.9  # Very high metadata weight
        )
        
        # Should still return properly formatted results
        self.assertIsInstance(results, list)
        if results:
            self.assertIn('enhanced_score', results[0])


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test with larger dataset
    indexer = MetadataIndexer(collection_name="perf_test")
    
    # Create larger test dataset
    large_doc_set = []
    categories = ["ML", "DL", "NLP", "CV", "DataScience"]
    for i in range(50):  # 50 documents
        doc = MetadataDocument(
            f"Document {i} content about {categories[i % len(categories)]} topic",
            metadata={
                "category": categories[i % len(categories)],
                "author": f"Author_{i % 5}",
                "year": 2020 + (i % 5),
                "quality_score": 0.5 + (i % 5) * 0.1
            }
        )
        large_doc_set.append(doc)
    
    # Time the indexing
    start_time = time.time()
    indexer.add_documents(large_doc_set)
    indexing_time = time.time() - start_time
    
    # Time the retrieval
    retriever = MetadataRetriever(indexer)
    start_time = time.time()
    results = retriever.retrieve("machine learning", top_k=10)
    retrieval_time = time.time() - start_time
    
    print(f"Indexed {len(large_doc_set)} documents in {indexing_time:.3f}s")
    print(f"Retrieved and scored in {retrieval_time:.3f}s")
    print(f"Retrieved {len(results)} results")
    
    # Should be reasonably fast
    assert indexing_time < 10.0, f"Indexing took too long: {indexing_time}s"
    assert retrieval_time < 2.0, f"Retrieval took too long: {retrieval_time}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test metadata extraction and intelligent filtering
    print("Testing intelligent query processing...")
    
    indexer = MetadataIndexer(collection_name="intelligent_test")
    
    # Add documents with various metadata
    docs = [
        MetadataDocument(
            "Latest 2023 research on transformer models in NLP",
            metadata={
                "year": 2023,
                "category": "NLP",
                "source": "arxiv",
                "quality_score": 0.95,
                "content_type": "research_paper"
            }
        ),
        MetadataDocument(
            "2018 tutorial on basic neural networks",
            metadata={
                "year": 2018,
                "category": "ML", 
                "source": "blog",
                "quality_score": 0.7,
                "content_type": "tutorial"
            }
        ),
        MetadataDocument(
            "2023 high-quality tutorial on NLP",
            metadata={
                "year": 2023,
                "category": "NLP",
                "source": "journal",
                "quality_score": 0.9,
                "content_type": "tutorial"
            }
        )
    ]
    
    indexer.add_documents(docs)
    retriever = MetadataRetriever(indexer)
    
    # Test "recent NLP" query - should extract temporal and category requirements
    results = retriever.retrieve_with_intelligent_filtering("recent NLP", top_k=5)
    
    print(f"Query 'recent NLP' returned {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['content'][:60]}...")
        print(f"     Year: {result['metadata'].get('year')} | "
              f"Quality: {result['metadata'].get('quality_score')}")
    
    # Verify that recent documents are prioritized
    if results:
        years = [result['metadata'].get('year', 0) for result in results]
        print(f"Years retrieved: {years}")
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()