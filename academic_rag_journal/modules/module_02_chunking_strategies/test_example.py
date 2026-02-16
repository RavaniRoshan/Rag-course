"""
Test file for Module 2: Chunking Strategies Implementation
This file contains unit tests to validate the chunking functionality.
"""

import unittest
import sys
import os

# Add the module directory to the path so we can import the example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from example import AdvancedChunker, ContentAwareChunker, AcademicPaperChunker, Chunk, ChunkingStrategy


class TestChunkingModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.chunker = AdvancedChunker(chunk_size=100, overlap=20)
        self.sample_text = (
            "This is a sample text for chunking. It contains multiple sentences. "
            "Each sentence should be properly handled during chunking. "
            "The chunking algorithm needs to preserve semantic meaning. "
            "This is important for downstream tasks. "
            "Additional content to make the text long enough for meaningful chunking."
        )
    
    def test_fixed_length_chunking(self):
        """Test fixed length chunking strategy"""
        chunks = self.chunker.chunk_document(self.sample_text, ChunkingStrategy.FIXED_LENGTH)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) > 0 for chunk in chunks))
        
        # Check that text is preserved (approximately)
        reconstructed = ' '.join([chunk.content for chunk in chunks])
        self.assertGreaterEqual(len(reconstructed), len(self.sample_text) * 0.9)
    
    def test_semantic_chunking(self):
        """Test semantic chunking strategy"""
        chunks = self.chunker.chunk_document(self.sample_text, ChunkingStrategy.SEMANTIC)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) > 0 for chunk in chunks))
    
    def test_recursive_chunking(self):
        """Test recursive chunking strategy"""
        chunks = self.chunker.chunk_document(self.sample_text, ChunkingStrategy.RECURSIVE)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) > 0 for chunk in chunks))
    
    def test_sliding_window_chunking(self):
        """Test sliding window chunking strategy"""
        chunks = self.chunker.chunk_document(self.sample_text, ChunkingStrategy.SLIDING_WINDOW)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) > 0 for chunk in chunks))
    
    def test_chunk_content_preservation(self):
        """Test that chunking preserves original content"""
        original_tokens = self.chunker.encoding.encode(self.sample_text)
        
        strategies = [
            ChunkingStrategy.FIXED_LENGTH,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.SLIDING_WINDOW
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                chunks = self.chunker.chunk_document(self.sample_text, strategy)
                chunked_tokens = []
                
                for chunk in chunks:
                    chunked_tokens.extend(self.chunker.encoding.encode(chunk.content))
                
                # Allow for some differences due to overlap and boundaries
                self.assertLessEqual(
                    abs(len(original_tokens) - len(chunked_tokens)), 
                    len(original_tokens) * 0.1  # 10% tolerance
                )
    
    def test_content_aware_chunker(self):
        """Test content-aware chunking for HTML and code"""
        html_text = "<h1>Header</h1><p>This is a paragraph with some content.</p>"
        code_text = "def sample_function():\n    return 'hello world'"
        
        content_chunker = ContentAwareChunker(chunk_size=100)
        
        html_chunks = content_chunker.chunk_structured_document(html_text, "html")
        self.assertGreater(len(html_chunks), 0)
        
        code_chunks = content_chunker.chunk_structured_document(code_text, "code")
        self.assertGreater(len(code_chunks), 0)
        
        # Check that HTML chunks have proper metadata
        for chunk in html_chunks:
            self.assertIn('element_type', chunk.metadata)
    
    def test_academic_paper_chunker(self):
        """Test academic paper chunking"""
        academic_text = """
Abstract
This paper explores document chunking strategies.

1. Introduction
The field of natural language processing has seen advances.

2. Conclusion
This study demonstrates the importance of appropriate chunking strategies.
"""
        
        academic_chunker = AcademicPaperChunker(chunk_size=100, overlap=20)
        chunks = academic_chunker.chunk_academic_paper(academic_text)
        
        self.assertGreater(len(chunks), 0)
        
        # Validate the chunking
        validation = academic_chunker.validate_academic_chunking(academic_text, chunks)
        self.assertTrue(validation['text_preserved'])


class TestChunkStructure(unittest.TestCase):
    
    def test_chunk_dataclass(self):
        """Test the Chunk dataclass structure"""
        chunk = Chunk(
            id="test_chunk",
            content="This is test content",
            start_pos=0,
            end_pos=100,
            metadata={"test": "value"}
        )
        
        self.assertEqual(chunk.id, "test_chunk")
        self.assertEqual(chunk.content, "This is test content")
        self.assertEqual(chunk.start_pos, 0)
        self.assertEqual(chunk.end_pos, 100)
        self.assertEqual(chunk.metadata, {"test": "value"})
        self.assertIsNone(chunk.embedding)
        self.assertTrue(chunk.semantic_boundary)


def run_tests():
    """Run all tests in the module"""
    print("Running tests for Module 2: Chunking Strategies")
    print("=" * 50)
    
    # Create test suites
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestChunkingModule)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestChunkStructure)
    
    # Combine suites
    full_suite = unittest.TestSuite([suite1, suite2])
    
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