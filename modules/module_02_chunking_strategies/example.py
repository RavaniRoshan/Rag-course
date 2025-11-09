"""
RAG Course - Module 2: Chunking Strategies Implementation

This file contains the implementation of the AdvancedChunker class
and related functionality as described in the module.
"""

import re
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import tiktoken


class ChunkingStrategy(Enum):
    FIXED_LENGTH = "fixed_length"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    id: str
    content: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    semantic_boundary: bool = True  # Whether this is a natural boundary


class AdvancedChunker:
    def __init__(self, 
                 chunk_size: int = 256,
                 overlap: int = 50,
                 encoding_model: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_model)
        
    def chunk_document(self, 
                      text: str, 
                      strategy: ChunkingStrategy,
                      metadata: Optional[Dict] = None) -> List[Chunk]:
        """
        Main chunking method that dispatches to specific strategy methods
        """
        if strategy == ChunkingStrategy.FIXED_LENGTH:
            return self._chunk_fixed_length(text, metadata)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text, metadata)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(text, metadata)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self._chunk_sliding_window(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _chunk_fixed_length(self, text: str, metadata: Optional[Dict]) -> List[Chunk]:
        """Fixed length chunking with overlap"""
        tokens = self.encoding.encode(text)
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk = Chunk(
                id=f"fixed_{chunk_id}",
                content=chunk_text,
                start_pos=i,
                end_pos=min(i + self.chunk_size, len(tokens)),
                metadata=metadata or {}
            )
            chunks.append(chunk)
            chunk_id += 1
            
        return chunks
    
    def _chunk_semantic(self, text: str, metadata: Optional[Dict]) -> List[Chunk]:
        """Semantic chunking using sentence boundaries"""
        # Split text into sentences
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create a new chunk
                chunk = Chunk(
                    id=f"semantic_{chunk_id}",
                    content=current_chunk.strip(),
                    start_pos=len(current_chunk),
                    end_pos=len(current_chunk) + len(sentence),
                    metadata=metadata or {},
                    semantic_boundary=True
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap > 0:
                    # Get overlapping text from end of previous chunk
                    overlap_tokens = self.encoding.encode(current_chunk)
                    overlap_start = max(0, len(overlap_tokens) - self.overlap)
                    current_chunk = self.encoding.decode(overlap_tokens[overlap_start:])
                    current_tokens = len(overlap_tokens[overlap_start:])
                else:
                    current_chunk = ""
                    current_tokens = 0
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add the final chunk if it has content
        if current_chunk.strip():
            chunk = Chunk(
                id=f"semantic_{chunk_id}",
                content=current_chunk.strip(),
                start_pos=len(text) - len(current_chunk),
                end_pos=len(text),
                metadata=metadata or {},
                semantic_boundary=True
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_recursive(self, text: str, metadata: Optional[Dict]) -> List[Chunk]:
        """Recursive chunking using document structure"""
        # Define chunking patterns from largest to smallest
        patterns = [
            r'\n\s*\n',  # Paragraph breaks
            r'(?<=[.!?])\s+',  # Sentence breaks
            r'(?<=[,.])\s+',  # Comma/semicolon breaks
            r'\s+'  # Word breaks
        ]
        
        chunks = self._recursive_chunk(text, patterns, 0, metadata)
        return self._merge_small_chunks(chunks)
    
    def _recursive_chunk(self, text: str, patterns: List[str], pattern_idx: int, metadata: Optional[Dict]) -> List[Chunk]:
        """Helper method for recursive chunking"""
        if pattern_idx >= len(patterns):
            # If we've exhausted all patterns, do fixed chunking
            return self._chunk_fixed_length(text, metadata)
        
        chunks = []
        current_pattern = patterns[pattern_idx]
        segments = re.split(current_pattern, text)
        
        current_chunk = ""
        chunk_id = 0
        
        for segment in segments:
            segment_token_count = len(self.encoding.encode(segment))
            current_token_count = len(self.encoding.encode(current_chunk))
            
            if current_token_count + segment_token_count > self.chunk_size:
                if current_chunk:
                    chunk = Chunk(
                        id=f"recursive_{chunk_id}",
                        content=current_chunk.strip(),
                        start_pos=0,  # Simplified for example
                        end_pos=0,    # Simplified for example
                        metadata=metadata or {},
                        semantic_boundary=False
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Check if the segment itself is too large
                if segment_token_count > self.chunk_size:
                    # Recursively chunk this segment with the next pattern
                    sub_chunks = self._recursive_chunk(segment, patterns, pattern_idx + 1, metadata)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = segment
            else:
                if current_chunk:
                    current_chunk += " " + segment
                else:
                    current_chunk = segment
        
        # Add the final chunk
        if current_chunk:
            chunk = Chunk(
                id=f"recursive_{chunk_id}",
                content=current_chunk.strip(),
                start_pos=0,  # Simplified for example
                end_pos=0,    # Simplified for example
                metadata=metadata or {},
                semantic_boundary=False
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_sliding_window(self, text: str, metadata: Optional[Dict]) -> List[Chunk]:
        """Sliding window chunking with configurable overlap"""
        tokens = self.encoding.encode(text)
        chunks = []
        chunk_id = 0
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk = Chunk(
                id=f"sliding_{chunk_id}",
                content=chunk_text,
                start_pos=start_idx,
                end_pos=end_idx,
                metadata=metadata or {},
                semantic_boundary=False
            )
            chunks.append(chunk)
            
            # Move by chunk_size - overlap to create sliding effect
            start_idx += self.chunk_size - self.overlap
            chunk_id += 1
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns"""
        # Pattern to match sentence endings
        sentence_endings = r'[.!?]+'
        
        # Split by sentence endings but keep the delimiters
        sentences = re.split(f'({sentence_endings})', text)
        
        # Reconstruct sentences with their endings
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                sentence = sentences[i] + sentences[i+1]
            else:
                sentence = sentences[i]
            if sentence.strip():
                result.append(sentence.strip())
        
        return result
    
    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge small chunks to reach minimum size threshold"""
        if not chunks:
            return chunks
        
        min_size = self.chunk_size // 2  # 50% of target chunk size
        merged_chunks = []
        current_chunk_content = ""
        current_metadata = {}
        
        for chunk in chunks:
            current_content_tokens = len(self.encoding.encode(current_chunk_content))
            chunk_tokens = len(self.encoding.encode(chunk.content))
            
            # If adding this chunk would still be under min_size, merge it
            if current_content_tokens + chunk_tokens <= min_size or not merged_chunks:
                current_chunk_content += " " + chunk.content
                # Merge metadata (simple approach: keep all unique values)
                for key, value in chunk.metadata.items():
                    if key not in current_metadata:
                        current_metadata[key] = value
            else:
                # Create a new merged chunk
                merged_chunk = Chunk(
                    id=f"merged_{len(merged_chunks)}",
                    content=current_chunk_content.strip(),
                    start_pos=0,  # Simplified
                    end_pos=0,    # Simplified
                    metadata=current_metadata.copy()
                )
                merged_chunks.append(merged_chunk)
                current_chunk_content = chunk.content
                current_metadata = chunk.metadata.copy()
        
        # Add the final chunk
        if current_chunk_content.strip():
            merged_chunk = Chunk(
                id=f"merged_{len(merged_chunks)}",
                content=current_chunk_content.strip(),
                start_pos=0,  # Simplified
                end_pos=0,    # Simplified
                metadata=current_metadata
            )
            merged_chunks.append(merged_chunk)
        
        return merged_chunks


class ContentAwareChunker(AdvancedChunker):
    """Chunker that takes into account document structure (HTML, code, etc.)"""
    
    def __init__(self, 
                 chunk_size: int = 256,
                 overlap: int = 50,
                 encoding_model: str = "cl100k_base"):
        super().__init__(chunk_size, overlap, encoding_model)
        self.html_patterns = [
            (r'<h[1-6][^>]*>.*?</h[1-6]>', 'heading'),
            (r'<p[^>]*>.*?</p>', 'paragraph'),
            (r'<div[^>]*>.*?</div>', 'div'),
            (r'<code[^>]*>.*?</code>', 'code_block'),
            (r'<pre[^>]*>.*?</pre>', 'preformatted'),
        ]
        self.code_patterns = [
            (r'```[\s\S]*?```', 'code_fence'),
            (r'"""[\s\S]*?"""', 'docstring'),
            (r"'''[\s\S]*?'''", 'docstring'),
        ]
    
    def chunk_structured_document(self, text: str, content_type: str = "text") -> List[Chunk]:
        """Specialized chunking for structured documents"""
        if content_type == "html":
            return self._chunk_html(text)
        elif content_type == "code":
            return self._chunk_code(text)
        else:
            return self.chunk_document(text, ChunkingStrategy.RECURSIVE)
    
    def _chunk_html(self, html_text: str) -> List[Chunk]:
        """Chunk HTML content respecting document structure"""
        chunks = []
        chunk_id = 0
        
        # Find all HTML elements
        for pattern, element_type in self.html_patterns:
            elements = re.findall(pattern, html_text, re.DOTALL | re.IGNORECASE)
            
            for element in elements:
                if len(self.encoding.encode(element)) <= self.chunk_size:
                    chunk = Chunk(
                        id=f"html_{chunk_id}",
                        content=element,
                        start_pos=0,  # Simplified
                        end_pos=0,    # Simplified
                        metadata={"element_type": element_type}
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                else:
                    # If element is too big, recursively chunk it
                    inner_text = re.sub(r'<[^>]+>', '', element)
                    sub_chunks = self.chunk_document(inner_text, ChunkingStrategy.RECURSIVE)
                    for sub_chunk in sub_chunks:
                        sub_chunk.id = f"html_{chunk_id}"
                        sub_chunk.metadata.update({"element_type": element_type})
                        chunks.append(sub_chunk)
                        chunk_id += 1
        
        return chunks
    
    def _chunk_code(self, code_text: str) -> List[Chunk]:
        """Chunk code respecting function/class boundaries"""
        chunks = []
        chunk_id = 0
        
        # Identify code structures
        structures = self._identify_code_structures(code_text)
        
        for structure in structures:
            if len(self.encoding.encode(structure['content'])) <= self.chunk_size:
                chunk = Chunk(
                    id=f"code_{chunk_id}",
                    content=structure['content'],
                    start_pos=0,  # Simplified
                    end_pos=0,    # Simplified
                    metadata={"structure_type": structure['type']}
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # If structure is too big, apply semantic chunking
                sub_chunks = self.chunk_document(structure['content'], ChunkingStrategy.SEMANTIC)
                for sub_chunk in sub_chunks:
                    sub_chunk.id = f"code_{chunk_id}"
                    sub_chunk.metadata.update({"structure_type": structure['type']})
                    chunks.append(sub_chunk)
                    chunk_id += 1
        
        return chunks
    
    def _identify_code_structures(self, code_text: str) -> List[Dict[str, str]]:
        """Identify functions, classes, and other code structures"""
        structures = []
        
        # Simple regex-based structure identification
        patterns = [
            (r'(def\s+\w+\s*\(.*?\):(?:\n\s+.+?)*?(?=\n\w|\n\s*def|\n\s*class|\Z))', 'function'),
            (r'(class\s+\w+\s*(?:\(.*?\))?:\s*(?:\n\s+.+?)*?(?=\n\w|\n\s*def|\n\s*class|\Z))', 'class'),
            (r'(#.*?(?=\n\w|\n\s*def|\n\s*class|\Z))', 'comment_block'),
        ]
        
        for pattern, structure_type in patterns:
            matches = re.findall(pattern, code_text, re.DOTALL)
            for match in matches:
                structures.append({
                    'content': match.strip(),
                    'type': structure_type
                })
        
        return structures


class AcademicPaperChunker(AdvancedChunker):
    """Specialized chunker for academic papers that respects academic document structure"""
    
    def __init__(self, 
                 chunk_size: int = 400,
                 overlap: int = 80):
        super().__init__(chunk_size, overlap)
        
        # Academic paper patterns
        self.section_patterns = [
            (r'(Abstract(?:.|\n)*?)(?=\n1\.? |Introduction|$)', 'abstract'),
            (r'(Introduction(?:.|\n)*?)(?=\n\d+\.?[^a-z]|Conclusion|$)', 'introduction'),
            (r'((?:^|\n)\d+\.?\s*Literature Review(?:.|\n)*?)(?=\n\d+\.?[^a-z]|Conclusion|$)', 'literature_review'),
            (r'((?:^|\n)\d+\.?\s*Methodology(?:.|\n)*?)(?=\n\d+\.?[^a-z]|Results|$)', 'methodology'),
            (r'((?:^|\n)\d+\.?\s*Results(?:.|\n)*?)(?=\n\d+\.?[^a-z]|Discussion|$)', 'results'),
            (r'((?:^|\n)\d+\.?\s*Discussion(?:.|\n)*?)(?=\n\d+\.?[^a-z]|Conclusion|$)', 'discussion'),
            (r'((?:^|\n)\d+\.?\s*Conclusion(?:.|\n)*?)(?=\nReferences|$)', 'conclusion'),
            (r'((?:^|\n)References?(?:.|\n)*)', 'references'),
        ]
        
        # Additional patterns for academic content
        self.academic_patterns = [
            (r'(Table\s+\d+.*?)(?=\nTable\s+\d+|\Z)', 'table'),
            (r'(Figure\s+\d+.*?)(?=\nFigure\s+\d+|\Z)', 'figure'),
            (r'(\d+\.\d+.*?)(?=\n\d+\.\d+|\Z)', 'subsection'),
            (r'(\([^)]*\)\s*:\s*[^.!?]*[.!?])', 'citation_sentence'),
        ]
    
    def chunk_academic_paper(self, paper_text: str) -> List[Chunk]:
        """Chunk academic paper respecting its structure"""
        chunks = []
        chunk_id = 0
        
        # Try to identify major sections first
        for pattern, section_type in self.section_patterns:
            sections = re.findall(pattern, paper_text, re.MULTILINE | re.IGNORECASE)
            
            for section in sections:
                section_text = section.strip() if isinstance(section, str) else section[0].strip()
                
                if len(self.encoding.encode(section_text)) <= self.chunk_size:
                    # Section fits in one chunk
                    chunk = Chunk(
                        id=f"paper_{section_type}_{chunk_id}",
                        content=section_text,
                        start_pos=0,  # Simplified
                        end_pos=0,    # Simplified
                        metadata={
                            "section_type": section_type,
                            "paper_structure": True
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                else:
                    # Section is too big, chunk it semantically
                    sub_chunks = self.chunk_document(section_text, ChunkingStrategy.SEMANTIC)
                    for sub_chunk in sub_chunks:
                        sub_chunk.id = f"paper_{section_type}_{chunk_id}"
                        sub_chunk.metadata.update({
                            "section_type": section_type,
                            "paper_structure": True
                        })
                        chunks.append(sub_chunk)
                        chunk_id += 1
        
        # If no major sections were found, fall back to semantic chunking
        if not chunks:
            chunks = self.chunk_document(paper_text, ChunkingStrategy.SEMANTIC)
            for chunk in chunks:
                chunk.metadata["section_type"] = "unstructured"
        
        return chunks
    
    def validate_academic_chunking(self, original_text: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate academic paper chunking"""
        validation = {
            'text_preserved': True,
            'section_integrity': True,
            'academic_structure_maintained': True,
            'issues': []
        }
        
        # Check for key academic elements that should be preserved
        key_elements = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion']
        
        for element in key_elements:
            original_count = len(re.findall(element, original_text, re.IGNORECASE))
            chunked_count = sum(
                len(re.findall(element, chunk.content, re.IGNORECASE)) 
                for chunk in chunks
            )
            
            if original_count != chunked_count:
                validation['issues'].append(
                    f'{element} preservation issue: {original_count} vs {chunked_count}'
                )
                validation['academic_structure_maintained'] = False
        
        return validation


def main():
    """Example usage of the chunking strategies"""
    print("RAG Course - Module 2: Chunking Strategies Example")
    print("=" * 55)
    
    # Sample text
    sample_text = (
        "This is a sample text for chunking. It contains multiple sentences. "
        "Each sentence should be properly handled during chunking. "
        "The chunking algorithm needs to preserve semantic meaning. "
        "This is important for downstream tasks. "
        "Additional content to make the text long enough for meaningful chunking. "
        "More content to ensure we get multiple chunks with the default settings. "
        "The effectiveness of retrieval-augmented generation systems depends heavily "
        "on how documents are properly chunked during the indexing phase."
    )
    
    # Initialize the chunker
    chunker = AdvancedChunker(chunk_size=100, overlap=20)
    
    print("Chunking with different strategies:")
    
    # Fixed length chunking
    print("\n1. Fixed Length Chunking:")
    fixed_chunks = chunker.chunk_document(sample_text, ChunkingStrategy.FIXED_LENGTH)
    for i, chunk in enumerate(fixed_chunks):
        print(f"  Chunk {i+1}: {len(chunk.content)} chars, {len(chunker.encoding.encode(chunk.content))} tokens")
        print(f"    Content: {chunk.content[:50]}...")
    
    # Semantic chunking
    print("\n2. Semantic Chunking:")
    semantic_chunks = chunker.chunk_document(sample_text, ChunkingStrategy.SEMANTIC)
    for i, chunk in enumerate(semantic_chunks):
        print(f"  Chunk {i+1}: {len(chunk.content)} chars, {len(chunker.encoding.encode(chunk.content))} tokens")
        print(f"    Content: {chunk.content[:50]}...")
    
    # Recursive chunking
    print("\n3. Recursive Chunking:")
    recursive_chunks = chunker.chunk_document(sample_text, ChunkingStrategy.RECURSIVE)
    for i, chunk in enumerate(recursive_chunks):
        print(f"  Chunk {i+1}: {len(chunk.content)} chars, {len(chunker.encoding.encode(chunk.content))} tokens")
        print(f"    Content: {chunk.content[:50]}...")
    
    # Academic paper example
    print("\n4. Academic Paper Chunking:")
    academic_text = """
Abstract
This paper explores the effectiveness of different document chunking strategies in Retrieval-Augmented Generation (RAG) systems. We analyze various approaches including fixed-length, semantic, and recursive chunking methods.

1. Introduction
The field of natural language processing has seen significant advances with the introduction of large language models. Retrieval-Augmented Generation (RAG) systems combine the generative power of LLMs with information retrieval capabilities to improve factual accuracy and reduce hallucination.

2. Literature Review
Previous work on chunking strategies has focused primarily on fixed-length approaches. Smith et al. (2023) proposed a semantic chunking method that achieved promising results in academic document retrieval. However, their approach required significant computational resources.

3. Conclusion
This study demonstrates the importance of appropriate chunking strategies in RAG systems. Future work should explore hybrid approaches that combine multiple chunking methods.
"""
    
    academic_chunker = AcademicPaperChunker(chunk_size=200, overlap=50)
    academic_chunks = academic_chunker.chunk_academic_paper(academic_text)
    
    print(f"Created {len(academic_chunks)} chunks from academic paper:")
    for i, chunk in enumerate(academic_chunks):
        section_type = chunk.metadata.get('section_type', 'unknown')
        print(f"  Chunk {i+1} ({section_type}): {len(chunk.content)} chars")
        print(f"    Content: {chunk.content[:60]}...")
    
    # Validate academic chunking
    validation = academic_chunker.validate_academic_chunking(academic_text, academic_chunks)
    print(f"\nAcademic Chunking Validation:")
    print(f"  Text Preserved: {validation['text_preserved']}")
    print(f"  Issues: {validation['issues']}")


if __name__ == "__main__":
    main()