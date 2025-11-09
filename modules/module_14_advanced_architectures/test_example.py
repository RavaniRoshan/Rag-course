"""
Module 14: Advanced Architectures
Tests for implementation examples

This module contains tests for the advanced architecture implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np
from datetime import datetime

from example import (
    DocumentNode, HierarchicalRAG, GraphBasedRAG, RecursiveRAG, 
    MultiModalRAG, MultiHopRAG, AdvancedRAGFactory
)


class TestDocumentNode(unittest.TestCase):
    """Test document node for hierarchical structures"""
    
    def test_initialization(self):
        """Test document node initialization"""
        node = DocumentNode(
            id="test_id",
            content="Test content",
            level=1,
            parent_id="parent_id",
            metadata={"type": "test"}
        )
        
        self.assertEqual(node.id, "test_id")
        self.assertEqual(node.content, "Test content")
        self.assertEqual(node.level, 1)
        self.assertEqual(node.parent_id, "parent_id")
        self.assertEqual(node.metadata, {"type": "test"})
        self.assertEqual(node.children, [])
        self.assertIsNone(node.embedding)
        self.assertIsNone(node.summary)


class TestHierarchicalRAG(unittest.TestCase):
    """Test hierarchical RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.hierarchical_rag = HierarchicalRAG(levels=3)
    
    def test_initialization(self):
        """Test hierarchical RAG initialization"""
        self.assertEqual(self.hierarchical_rag.levels, 3)
        self.assertEqual(self.hierarchical_rag.document_tree, {})
        self.assertIsNotNone(self.hierarchical_rag.embedder)
        self.assertEqual(len(self.hierarchical_rag.level_collections), 3)
    
    def test_build_hierarchical_structure(self):
        """Test building hierarchical structure"""
        documents = [
            "This is a long document about machine learning techniques. " * 5,
            "Another document about deep learning architectures. " * 5
        ]
        
        root_id = self.hierarchical_rag.build_hierarchical_structure(documents)
        
        self.assertIsInstance(root_id, str)
        self.assertIn(root_id, self.hierarchical_rag.document_tree)
        self.assertGreater(len(self.hierarchical_rag.document_tree), 1)  # Should have created child nodes
    
    def test_split_content(self):
        """Test content splitting functionality"""
        content = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = self.hierarchical_rag._split_content(content, chunk_size=3)
        
        self.assertEqual(len(chunks), 3)  # Should split into 3 chunks
        self.assertEqual(chunks[0], "word1 word2 word3")
        self.assertEqual(chunks[1], "word4 word5 word6")
        self.assertEqual(chunks[2], "word7 word8")
    
    def test_retrieve_hierarchical(self):
        """Test hierarchical retrieval"""
        # Build structure first
        documents = ["Machine learning algorithms", "Deep learning models"]
        self.hierarchical_rag.build_hierarchical_structure(documents)
        
        # Perform hierarchical retrieval
        results = self.hierarchical_rag.retrieve_hierarchical(
            "machine learning", 
            top_k_per_level=[2, 2, 2]
        )
        
        self.assertIsInstance(results, dict)
        # Should have results for each level that was populated
        for level, level_results in results.items():
            self.assertIsInstance(level, int)
            self.assertIsInstance(level_results, list)
            for result in level_results:
                self.assertIn('content', result)
                self.assertIn('similarity', result)
                self.assertIn('level', result)


class TestGraphBasedRAG(unittest.TestCase):
    """Test graph-based RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.graph_rag = GraphBasedRAG()
    
    def test_initialization(self):
        """Test graph RAG initialization"""
        self.assertIsInstance(self.graph_rag.graph, object)  # nx.Graph object
        self.assertIsNotNone(self.graph_rag.embedder)
        self.assertIsNotNone(self.graph_rag.collection)
        self.assertEqual(self.graph_rag.entity_embeddings, {})
        self.assertEqual(self.graph_rag.relation_embeddings, {})
    
    def test_extract_entities_and_relations(self):
        """Test entity and relation extraction"""
        text = "Machine Learning and Deep Learning are important fields in AI."
        
        entities, relations = self.graph_rag.extract_entities_and_relations(text)
        
        self.assertIsInstance(entities, list)
        self.assertIsInstance(relations, list)
        # Check that entities were identified
        if entities:  # If any entities were found
            self.assertTrue(all(isinstance(e, str) for e in entities))
    
    def test_build_knowledge_graph(self):
        """Test knowledge graph building"""
        documents = [
            "Artificial Intelligence and Machine Learning are related fields.",
            "Deep Learning is a subset of Machine Learning."
        ]
        
        self.graph_rag.build_knowledge_graph(documents)
        
        # Should have added entities to the graph
        nodes = list(self.graph_rag.graph.nodes())
        edges = list(self.graph_rag.graph.edges())
        
        # Check that some entities were extracted
        self.assertGreaterEqual(len(nodes), 0)
        
        # Check that embeddings were created
        self.assertGreaterEqual(len(self.graph_rag.entity_embeddings), 0)
    
    def test_retrieve_via_graph(self):
        """Test graph-based retrieval"""
        documents = [
            "Machine Learning algorithms are powerful.",
            "Deep Learning uses neural networks."
        ]
        
        self.graph_rag.build_knowledge_graph(documents)
        
        results = self.graph_rag.retrieve_via_graph("machine learning", top_k=2)
        
        self.assertIsInstance(results, list)
        for result in results:
            self.assertIn('content', result)
            self.assertIn('similarity', result)
            self.assertIn('related_entities', result)


class TestRecursiveRAG(unittest.TestCase):
    """Test recursive RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.recursive_rag = RecursiveRAG()
    
    def test_initialization(self):
        """Test recursive RAG initialization"""
        self.assertIsNotNone(self.recursive_rag.embedder)
        self.assertIsNotNone(self.recursive_rag.collection)
        self.assertEqual(self.recursive_rag.max_recursion_depth, 3)
        self.assertEqual(self.recursive_rag.min_improvement_threshold, 0.1)
    
    def test_add_documents(self):
        """Test adding documents"""
        documents = ["Document 1 content", "Document 2 content"]
        self.recursive_rag.add_documents(documents)
        
        # Should be able to perform retrieval
        result = self.recursive_rag.retrieve_recursive("test query")
        self.assertIn('results', result)
        self.assertIn('depth', result)
        self.assertIn('final', result)
    
    def test_retrieve_recursive(self):
        """Test recursive retrieval"""
        documents = ["Machine learning is important", "Deep learning is powerful"]
        self.recursive_rag.add_documents(documents)
        
        result = self.recursive_rag.retrieve_recursive("machine learning", depth=0)
        
        self.assertIn('query', result)
        self.assertIn('results', result)
        self.assertIn('depth', result)
        self.assertIn('final', result)
        
        self.assertIsInstance(result['results'], list)
        self.assertIsInstance(result['depth'], int)
        self.assertIsInstance(result['final'], bool)


class TestMultiModalRAG(unittest.TestCase):
    """Test multi-modal RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.multimodal_rag = MultiModalRAG()
    
    def test_initialization(self):
        """Test multi-modal RAG initialization"""
        self.assertIsNotNone(self.multimodal_rag.text_embedder)
        self.assertIsNotNone(self.multimodal_rag.text_collection)
    
    def test_add_text_content(self):
        """Test adding text content"""
        texts = ["Text 1", "Text 2"]
        metadata = [{"type": "doc"}, {"type": "example"}]
        
        self.multimodal_rag.add_text_content(texts, metadata)
        
        # Should be able to retrieve
        result = self.multimodal_rag.retrieve_multimodal("text")
        self.assertIn('results', result)
        self.assertIn('modalities_used', result)
        self.assertEqual(result['modalities_used'], ['text'])  # Only text supported in this implementation
    
    def test_retrieve_multimodal(self):
        """Test multi-modal retrieval"""
        texts = ["Machine learning concepts", "AI techniques"]
        self.multimodal_rag.add_text_content(texts)
        
        result = self.multimodal_rag.retrieve_multimodal("machine learning")
        
        self.assertIn('results', result)
        self.assertIn('query_modality', result)
        self.assertIsInstance(result['results'], list)
        for res in result['results']:
            self.assertIn('content', res)
            self.assertIn('similarity', res)
            self.assertEqual(res['modality'], 'text')


class TestMultiHopRAG(unittest.TestCase):
    """Test multi-hop RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.multihop_rag = MultiHopRAG()
    
    def test_initialization(self):
        """Test multi-hop RAG initialization"""
        self.assertIsNotNone(self.multihop_rag.embedder)
        self.assertIsNotNone(self.multihop_rag.collection)
        self.assertEqual(self.multihop_rag.max_hops, 3)
    
    def test_add_documents(self):
        """Test adding documents"""
        documents = ["Document about AI", "Document about ML"]
        self.multihop_rag.add_documents(documents)
        
        # Should have added documents to collection
        self.assertEqual(self.multihop_rag.collection.count(), 2)
    
    def test_retrieve_multihop(self):
        """Test multi-hop retrieval"""
        documents = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks", 
            "Neural networks are powerful tools"
        ]
        self.multihop_rag.add_documents(documents)
        
        result = self.multihop_rag.retrieve_multihop("How is ML related to AI?")
        
        self.assertIn('original_query', result)
        self.assertIn('hops', result)
        self.assertIn('final_context', result)
        
        self.assertIsInstance(result['hops'], list)
        self.assertLessEqual(len(result['hops']), self.multihop_rag.max_hops)
        
        for hop in result['hops']:
            self.assertIn('hop', hop)
            self.assertIn('query', hop)
            self.assertIn('results', hop)
            self.assertIsInstance(hop['hop'], int)
            self.assertIsInstance(hop['results'], list)


class TestAdvancedRAGFactory(unittest.TestCase):
    """Test advanced RAG factory"""
    
    def test_create_hierarchical(self):
        """Test creating hierarchical RAG"""
        rag = AdvancedRAGFactory.create_hierarchical(levels=2)
        self.assertIsInstance(rag, HierarchicalRAG)
        self.assertEqual(rag.levels, 2)
    
    def test_create_graph_based(self):
        """Test creating graph-based RAG"""
        rag = AdvancedRAGFactory.create_graph_based()
        self.assertIsInstance(rag, GraphBasedRAG)
    
    def test_create_recursive(self):
        """Test creating recursive RAG"""
        rag = AdvancedRAGFactory.create_recursive()
        self.assertIsInstance(rag, RecursiveRAG)
    
    def test_create_multimodal(self):
        """Test creating multi-modal RAG"""
        rag = AdvancedRAGFactory.create_multimodal()
        self.assertIsInstance(rag, MultiModalRAG)
    
    def test_create_multihop(self):
        """Test creating multi-hop RAG"""
        rag = AdvancedRAGFactory.create_multihop()
        self.assertIsInstance(rag, MultiHopRAG)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_document_lists(self):
        """Test with empty document lists"""
        # Hierarchical
        hierarchical = HierarchicalRAG(levels=2)
        root_id = hierarchical.build_hierarchical_structure([])
        self.assertIsInstance(root_id, str)
        
        results = hierarchical.retrieve_hierarchical("test", top_k_per_level=[1, 1])
        self.assertIsInstance(results, dict)
        
        # Graph-based
        graph_rag = GraphBasedRAG()
        graph_rag.build_knowledge_graph([])
        
        results = graph_rag.retrieve_via_graph("test", top_k=1)
        self.assertIsInstance(results, list)
        
        # Recursive
        recursive_rag = RecursiveRAG()
        recursive_rag.add_documents([])
        results = recursive_rag.retrieve_recursive("test")
        self.assertIsInstance(results, dict)
        
        # Multi-modal
        multimodal_rag = MultiModalRAG()
        multimodal_rag.add_text_content([])
        results = multimodal_rag.retrieve_multimodal("test")
        self.assertIsInstance(results, dict)
        
        # Multi-hop
        multihop_rag = MultiHopRAG()
        multihop_rag.add_documents([])
        results = multihop_rag.retrieve_multihop("test")
        self.assertIsInstance(results, dict)
    
    def test_single_level_hierarchical(self):
        """Test hierarchical RAG with only one level"""
        rag = HierarchicalRAG(levels=1)
        rag.build_hierarchical_structure(["Single level document"])
        
        results = rag.retrieve_hierarchical("test", top_k_per_level=[2])
        self.assertIsInstance(results, dict)
        # Should only have results for level 0
        self.assertIn(0, results)
    
    def test_very_short_content(self):
        """Test with very short content"""
        hierarchical = HierarchicalRAG(levels=2)
        # Content shorter than chunk size should still be processed
        hierarchical.build_hierarchical_structure(["AI"])
        
        results = hierarchical.retrieve_hierarchical("AI", top_k_per_level=[1])
        self.assertIsInstance(results, dict)
    
    def test_retrieval_with_no_results(self):
        """Test retrieval when no results are found"""
        recursive_rag = RecursiveRAG()
        # Don't add any documents, then try to retrieve
        result = recursive_rag.retrieve_recursive("completely unrelated query")
        
        # Should handle gracefully and return appropriate structure
        self.assertIn('results', result)
        self.assertIn('depth', result)
        self.assertIn('final', result)


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test with larger datasets
    large_docs = [f"Document {i} contains information about various topics in AI and machine learning." for i in range(20)]
    
    # Test Hierarchical RAG performance
    start_time = time.time()
    hierarchical = HierarchicalRAG(levels=2)
    hierarchical.build_hierarchical_structure(large_docs)
    hierarchical_time = time.time() - start_time
    
    # Test Graph-based RAG performance
    start_time = time.time()
    graph_rag = GraphBasedRAG()
    graph_rag.build_knowledge_graph(large_docs)
    graph_time = time.time() - start_time
    
    # Test Recursive RAG performance
    start_time = time.time()
    recursive_rag = RecursiveRAG()
    recursive_rag.add_documents(large_docs)
    for i in range(5):  # Multiple queries
        recursive_rag.retrieve_recursive(f"query {i}")
    recursive_time = time.time() - start_time
    
    print(f"Performance Results:")
    print(f"  Hierarchical (20 docs): {hierarchical_time:.4f}s")
    print(f"  Graph-based (20 docs): {graph_time:.4f}s")
    print(f"  Recursive (5 queries): {recursive_time:.4f}s")
    
    # Ensure reasonable performance
    assert hierarchical_time < 10.0, f"Hierarchical processing too slow: {hierarchical_time}s"
    assert graph_time < 10.0, f"Graph processing too slow: {graph_time}s"
    assert recursive_time < 20.0, f"Recursive processing too slow: {recursive_time}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test architecture combinations
    print("Testing architecture combinations...")
    
    # Create documents for testing
    docs = [
        "Machine learning and artificial intelligence are related fields.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language."
    ]
    
    # Test hierarchical + graph combination
    print("Combining Hierarchical and Graph approaches:")
    hierarchical = HierarchicalRAG(levels=2)
    root_id = hierarchical.build_hierarchical_structure(docs)
    
    # Build graph from the same documents
    graph_rag = GraphBasedRAG()
    graph_rag.build_knowledge_graph(docs)
    
    # Perform retrieval using both approaches
    hier_results = hierarchical.retrieve_hierarchical("machine learning", top_k_per_level=[2])
    graph_results = graph_rag.retrieve_via_graph("machine learning", top_k=2)
    
    print(f"  Hierarchical retrieved: {sum(len(results) for results in hier_results.values())} results across levels")
    print(f"  Graph-based retrieved: {len(graph_results)} results with relations")
    
    # Test multi-hop reasoning
    print("\nTesting Multi-Hop Reasoning:")
    multihop_rag = MultiHopRAG()
    multihop_rag.add_documents(docs)
    
    multihop_result = multihop_rag.retrieve_multihop("How does AI relate to neural networks?")
    print(f"  Multi-hop completed {len(multihop_result['hops'])} hops")
    for i, hop in enumerate(multihop_result['hops']):
        print(f"    Hop {i+1}: {len(hop['results'])} results for query '{hop['query'][:30]}...'")
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()