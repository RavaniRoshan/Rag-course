# **Deep Dive: Module 2 - Chunking Strategies (Document Segmentation Techniques)**

## **1. Advanced Theoretical Foundations**

### **The Mathematical Basis of Chunking**

Chunking strategies are fundamentally about **information preservation vs. context fragmentation optimization**. The goal is to find the optimal balance between:

- **Semantic Coherence**: Maintaining meaningful, self-contained information units
- **Retrieval Efficiency**: Ensuring chunks are appropriately sized for retrieval
- **Context Preservation**: Maintaining relationships between related concepts

The mathematical approach to chunking involves:

```
Optimization Function: F(chunk_size, context_preservation, retrieval_efficiency)

Where:
- chunk_size ∈ [min_size, max_size] (typically 100-500 tokens for text)
- context_preservation = f(coherence, semantic_density)
- retrieval_efficiency = f(embedding_quality, search_speed)
```

### **Information Theory Perspective**

From an information theory standpoint, chunking aims to:

1. **Maximize Information Density**: Each chunk should contain maximum relevant information per token
2. **Minimize Information Leakage**: Ensure related information isn't unnecessarily split across chunks
3. **Preserve Semantic Relationships**: Maintain connections between related concepts within chunks

### **Types of Chunking Strategies**

- **Fixed-Length Chunking**: Most basic approach with consistent size boundaries
- **Semantic Chunking**: Uses sentence boundaries and document structure
- **Content-Aware Chunking**: Incorporates content structure (headers, tables, code blocks)
- **Sliding Window Chunking**: Overlapping chunks to preserve context
- **Recursive Chunking**: Hierarchical chunking based on document structure

## **2. Extended Technical Implementation**

### **Advanced Chunking Implementation with Multiple Strategies**

```python
import re
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
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
                 encoding_model: str = "cl100k_base",
                 sentence_model: Optional[SentenceTransformer] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_model)
        self.sentence_model = sentence_model or SentenceTransformer('all-MiniLM-L6-v2')
        
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
            
            # Ensure we don't cut in the middle of a word
            if i + self.chunk_size < len(tokens):
                # Find the next natural boundary
                next_space = text.find(' ', self.encoding.decode(tokens[:i + self.chunk_size]).rfind(' ') + 1)
                if next_space != -1:
                    chunk_text = text[self.encoding.decode(tokens[:i]).rfind(' '):next_space]
            
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
        
        # Simple regex-based structure identification (in practice, you'd use AST)
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
```

### **Chunk Quality Evaluation and Optimization**

```python
class ChunkQualityEvaluator:
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def evaluate_chunk_quality(self, chunks: List[Chunk]) -> Dict[str, float]:
        """Evaluate the quality of chunks based on several metrics"""
        metrics = {}
        
        # 1. Semantic Coherence: How well the chunk holds together semantically
        coherence_scores = []
        for chunk in chunks:
            sentences = chunk.content.split('.')
            if len(sentences) > 1:
                sentence_embeddings = self.model.encode(sentences)
                # Calculate average similarity between sentences
                similarities = []
                for i in range(len(sentence_embeddings) - 1):
                    sim = np.dot(sentence_embeddings[i], sentence_embeddings[i+1]) / (
                        np.linalg.norm(sentence_embeddings[i]) * 
                        np.linalg.norm(sentence_embeddings[i+1])
                    )
                    similarities.append(sim)
                coherence = np.mean(similarities) if similarities else 0
                coherence_scores.append(coherence)
        
        metrics['semantic_coherence'] = np.mean(coherence_scores) if coherence_scores else 0
        
        # 2. Size Distribution
        sizes = [len(self.encoding.encode(chunk.content)) for chunk in chunks]
        metrics['avg_size'] = np.mean(sizes)
        metrics['std_size'] = np.std(sizes)
        metrics['size_uniformity'] = 1 - (metrics['std_size'] / (metrics['avg_size'] + 1e-8))
        
        # 3. Boundary Quality: How well chunks respect semantic boundaries
        boundary_quality = self._evaluate_boundary_quality(chunks)
        metrics['boundary_quality'] = boundary_quality
        
        return metrics
    
    def _evaluate_boundary_quality(self, chunks: List[Chunk]) -> float:
        """Evaluate how well chunks respect semantic boundaries"""
        if len(chunks) < 2:
            return 1.0  # Perfect if there's only one chunk
        
        boundary_scores = []
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i+1]
            
            # Check if boundary is at a natural break (end of sentence, paragraph, etc.)
            last_chars_chunk1 = chunk1.content[-10:]  # Last 10 characters
            first_chars_chunk2 = chunk2.content[:10]   # First 10 characters
            
            # Score based on natural boundary indicators
            score = 0
            if any(c in '.!?' for c in last_chars_chunk1[-3:]):  # Ends with sentence ending
                score += 0.5
            if any(c in '\n\r' for c in last_chars_chunk1):  # Ends with newline
                score += 0.3
            if any(c in ' \t' for c in first_chars_chunk2[:2]):  # Starts with space/tab
                score += 0.2
            
            boundary_scores.append(min(score, 1.0))
        
        return np.mean(boundary_scores) if boundary_scores else 0.0
    
    def suggest_optimal_chunking_params(self, text: str) -> Dict[str, Any]:
        """Suggest optimal chunking parameters for a given text"""
        # Analyze text characteristics
        avg_sentence_length = self._analyze_sentence_length(text)
        text_complexity = self._analyze_text_complexity(text)
        semantic_density = self._estimate_semantic_density(text)
        
        # Suggest parameters based on analysis
        if text_complexity > 0.7:  # High complexity
            suggested_size = min(150, int(avg_sentence_length * 3))
        elif text_complexity < 0.3:  # Low complexity
            suggested_size = min(400, int(avg_sentence_length * 8))
        else:  # Medium complexity
            suggested_size = min(250, int(avg_sentence_length * 5))
        
        # Adjust overlap based on semantic density
        if semantic_density > 0.8:  # High density
            suggested_overlap = int(suggested_size * 0.3)
        else:
            suggested_overlap = int(suggested_size * 0.15)
        
        return {
            'suggested_chunk_size': suggested_size,
            'suggested_overlap': suggested_overlap,
            'text_complexity': text_complexity,
            'semantic_density': semantic_density,
            'avg_sentence_length': avg_sentence_length
        }
    
    def _analyze_sentence_length(self, text: str) -> float:
        """Analyze average sentence length in tokens"""
        sentences = re.split(r'[.!?]+', text)
        token_counts = [len(self.encoding.encode(s.strip())) for s in sentences if s.strip()]
        return np.mean(token_counts) if token_counts else 10
    
    def _analyze_text_complexity(self, text: str) -> float:
        """Estimate text complexity on a 0-1 scale"""
        # Simple complexity measure based on vocabulary diversity and sentence structure
        tokens = self.encoding.encode(text)
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        lexical_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Also consider average word length (as proxy for complex vocabulary)
        words = re.findall(r'\b\w+\b', text.lower())
        avg_word_length = np.mean([len(w) for w in words]) if words else 5
        
        # Normalize and combine metrics
        complexity = (lexical_diversity * 0.6) + (min(avg_word_length / 10, 1) * 0.4)
        return min(complexity, 1.0)
    
    def _estimate_semantic_density(self, text: str) -> float:
        """Estimate semantic density (information per token)"""
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 10]  # Filter very short sentences
        
        if not meaningful_sentences:
            return 0.5
        
        # Estimate based on sentence structure variety and keyword density
        avg_sentence_length = np.mean([len(s) for s in meaningful_sentences]) if meaningful_sentences else 50
        keyword_indicators = ['the', 'and', 'or', 'but', 'however', 'therefore', 'because']
        text_lower = text.lower()
        keyword_density = sum(text_lower.count(kw) for kw in keyword_indicators) / len(meaningful_sentences) if meaningful_sentences else 0
        
        # Normalize to 0-1 scale
        density = (min(avg_sentence_length / 100, 1) * 0.4) + (min(keyword_density / 5, 1) * 0.6)
        return min(density, 1.0)
```

## **3. Advanced Real-World Applications**

### **Legal Document Processing System**

```python
class LegalDocumentChunker(AdvancedChunker):
    """Specialized chunker for legal documents that respects legal structure"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 100):
        super().__init__(chunk_size, overlap)
        
        # Legal document patterns
        self.legal_patterns = [
            (r'(Section\s+\d+[A-Z]*\.?.*?)(?=Section\s+\d+[A-Z]*\.?|$)', 'section'),
            (r'(Article\s+\d+\.?.*?)(?=Article\s+\d+\.?|$)', 'article'),
            (r'(Clause\s+\d+\.?.*?)(?=Clause\s+\d+\.?|$)', 'clause'),
            (r'(Subsection\s+\([a-zA-Z0-9]+\).*?)(?=\n\s*Subsection\s+\([a-zA-Z0-9]+\)|$)', 'subsection'),
            (r'(Paragraph\s+\d+\.?.*?)(?=Paragraph\s+\d+\.?|$)', 'paragraph'),
        ]
    
    def chunk_legal_document(self, legal_text: str, jurisdiction: str = None) -> List[Chunk]:
        """Chunk legal document respecting legal document structure"""
        chunks = []
        chunk_id = 0
        
        # First, try to extract by legal structure
        for pattern, element_type in self.legal_patterns:
            elements = re.findall(pattern, legal_text, re.DOTALL | re.IGNORECASE)
            
            for element in elements:
                element_clean = element.strip()
                element_tokens = len(self.encoding.encode(element_clean))
                
                if element_tokens <= self.chunk_size:
                    # Element fits in one chunk
                    chunk = Chunk(
                        id=f"legal_{element_type}_{chunk_id}",
                        content=element_clean,
                        start_pos=0,  # Simplified
                        end_pos=0,    # Simplified
                        metadata={
                            "element_type": element_type,
                            "jurisdiction": jurisdiction,
                            "legal_structure": True
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                else:
                    # Element is too big, chunk it semantically
                    sub_chunks = self.chunk_document(element_clean, ChunkingStrategy.SEMANTIC)
                    for sub_chunk in sub_chunks:
                        sub_chunk.id = f"legal_{element_type}_{chunk_id}"
                        sub_chunk.metadata.update({
                            "element_type": element_type,
                            "jurisdiction": jurisdiction,
                            "legal_structure": True
                        })
                        chunks.append(sub_chunk)
                        chunk_id += 1
        
        # Handle any remaining content that wasn't captured by legal patterns
        if not chunks:
            # Fall back to recursive chunking for the entire document
            fallback_chunks = self.chunk_document(legal_text, ChunkingStrategy.RECURSIVE)
            for chunk in fallback_chunks:
                chunk.metadata.update({
                    "element_type": "unstructured",
                    "jurisdiction": jurisdiction,
                    "legal_structure": False
                })
                chunks.append(chunk)
        
        return chunks
    
    def validate_legal_chunking(self, original_text: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate that legal document chunking preserves important information"""
        validation_results = {
            'text_preservation': True,
            'structural_integrity': True,
            'no_information_loss': True,
            'issues': []
        }
        
        # Check text preservation
        reconstructed_text = ' '.join([chunk.content for chunk in chunks])
        if original_text.replace('\n', ' ').replace('\r', ' ') != reconstructed_text.replace('\n', ' ').replace('\r', ' '):
            validation_results['text_preservation'] = False
            validation_results['issues'].append('Text preservation failed')
        
        # Check for important legal elements that might be split
        # (This is a simplified check - in practice, more sophisticated validation would be needed)
        important_patterns = [
            r'section\s+\d+[a-z]*',
            r'article\s+\d+',
            r'clause\s+\d+',
            r'subsection\s+\([a-z0-9]+\)',
        ]
        
        for pattern in important_patterns:
            original_matches = len(re.findall(pattern, original_text, re.IGNORECASE))
            chunked_matches = sum(len(re.findall(pattern, chunk.content, re.IGNORECASE)) for chunk in chunks)
            
            if original_matches != chunked_matches:
                validation_results['structural_integrity'] = False
                validation_results['issues'].append(f'Pattern "{pattern}" not preserved: {original_matches} vs {chunked_matches}')
        
        return validation_results
```

### **Code Repository Chunking System**

```python
import ast
import tokenize
from io import StringIO


class CodeChunker(ContentAwareChunker):
    """Advanced code chunker that uses AST and tokenization for better code understanding"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 50):
        super().__init__(chunk_size, overlap)
        self.language_patterns = {
            'python': {
                'function': [r'def\s+\w+\s*\(.*?\):(?:\n\s+.+?)*?(?=\n\w|\n\s*def|\n\s*class|\n\s*if|\n\s*for|\n\s*while|\Z)'],
                'class': [r'class\s+\w+\s*(?:\(.*?\))?:\s*(?:\n\s+.+?)*?(?=\n\w|\n\s*def|\n\s*class|\Z)'],
                'import': [r'^(import|from)\s+.*?$'],
                'comment': [r'""".*?"""', r"'''.*?'''", r'#.*?$'],
            },
            'javascript': {
                'function': [r'function\s+\w+\s*\(.*?\)\s*\{(?:[^{}]*|\{[^{}]*\})*\}',
                             r'const\s+\w+\s*=\s*\([^)]*\)\s*=>\s*\{(?:[^{}]*|\{[^{}]*\})*\}',
                             r'const\s+\w+\s*=\s*function\s*\([^)]*\)\s*\{(?:[^{}]*|\{[^{}]*\})*\}'],
                'class': [r'class\s+\w+\s*(?:extends\s+\w+)?\s*\{(?:[^{}]*|\{[^{}]*\})*\}'],
                'import': [r'import\s+.*?from\s+[\'\"].*?[\'\"]', r'import\s+[\'\"].*?[\'\"]'],
                'comment': [r'//.*?$', r'/\*.*?\*/'],
            }
        }
    
    def chunk_code_with_ast(self, code: str, language: str = 'python') -> List[Chunk]:
        """Chunk code using AST analysis for better structural understanding"""
        chunks = []
        chunk_id = 0
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Walk the AST and extract code structures
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                    start_line = node.lineno - 1
                    end_line = getattr(node, 'end_lineno', start_line)
                    
                    # Extract the code for this node
                    node_lines = code.split('\n')[start_line:end_line + 1]
                    node_code = '\n'.join(node_lines)
                    
                    # Check if it exceeds chunk size
                    node_tokens = len(self.encoding.encode(node_code))
                    
                    if node_tokens <= self.chunk_size:
                        chunk = Chunk(
                            id=f"code_{type(node).__name__}_{chunk_id}",
                            content=node_code,
                            start_pos=start_line,
                            end_pos=end_line,
                            metadata={
                                "node_type": type(node).__name__,
                                "language": language,
                                "ast_node": True
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                    else:
                        # If the node is too big, fall back to pattern-based chunking
                        sub_chunks = self._chunk_large_code_node(node_code, language)
                        for sub_chunk in sub_chunks:
                            sub_chunk.id = f"code_{type(node).__name__}_{chunk_id}"
                            sub_chunk.metadata.update({
                                "node_type": type(node).__name__,
                                "language": language,
                                "ast_node": True
                            })
                            chunks.append(sub_chunk)
                            chunk_id += 1
        except SyntaxError:
            # If parsing fails, fall back to pattern-based chunking
            print(f"Warning: Could not parse code as valid {language}. Falling back to pattern-based chunking.")
            chunks = self._chunk_code_by_patterns(code, language)
        
        return chunks
    
    def _chunk_large_code_node(self, code: str, language: str) -> List[Chunk]:
        """Handle code nodes that exceed chunk size"""
        return self.chunk_document(code, ChunkingStrategy.SEMANTIC)
    
    def _chunk_code_by_patterns(self, code: str, language: str) -> List[Chunk]:
        """Chunk code using regex patterns when AST parsing fails"""
        chunks = []
        chunk_id = 0
        
        if language in self.language_patterns:
            patterns = self.language_patterns[language]
            
            for element_type, regexes in patterns.items():
                for regex in regexes:
                    elements = re.findall(regex, code, re.MULTILINE | re.DOTALL)
                    
                    for element in elements:
                        element_tokens = len(self.encoding.encode(element))
                        
                        if element_tokens <= self.chunk_size:
                            chunk = Chunk(
                                id=f"code_{element_type}_{chunk_id}",
                                content=element,
                                start_pos=0,  # Simplified
                                end_pos=0,    # Simplified
                                metadata={
                                    "element_type": element_type,
                                    "language": language,
                                    "ast_node": False
                                }
                            )
                            chunks.append(chunk)
                            chunk_id += 1
                        else:
                            # Recursively chunk if too large
                            sub_chunks = self.chunk_document(element, ChunkingStrategy.SEMANTIC)
                            for sub_chunk in sub_chunks:
                                sub_chunk.id = f"code_{element_type}_{chunk_id}"
                                sub_chunk.metadata.update({
                                    "element_type": element_type,
                                    "language": language,
                                    "ast_node": False
                                })
                                chunks.append(sub_chunk)
                                chunk_id += 1
        
        # Handle remaining content
        if not chunks:
            chunks = self.chunk_document(code, ChunkingStrategy.SEMANTIC)
        
        return chunks
    
    def chunk_readme_file(self, readme_content: str) -> List[Chunk]:
        """Specialized chunking for README files with markdown structure"""
        chunks = []
        chunk_id = 0
        
        # Split by markdown headers
        header_pattern = r'^(#+\s+.*)$'
        lines = readme_content.split('\n')
        
        current_section = []
        current_header = "Introduction"
        
        for line in lines:
            if re.match(header_pattern, line.strip()):
                # Save previous section if it exists
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        chunk = Chunk(
                            id=f"readme_section_{chunk_id}",
                            content=section_text,
                            start_pos=0,  # Simplified
                            end_pos=0,    # Simplified
                            metadata={
                                "section_header": current_header,
                                "content_type": "readme_section"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                
                # Start new section
                current_header = line.strip()
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                chunk = Chunk(
                    id=f"readme_section_{chunk_id}",
                    content=section_text,
                    start_pos=0,  # Simplified
                    end_pos=0,    # Simplified
                    metadata={
                        "section_header": current_header,
                        "content_type": "readme_section"
                    }
                )
                chunks.append(chunk)
        
        return chunks
```

## **4. Performance Optimization Strategies**

### **Efficient Chunking for Large Documents**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import partial


class OptimizedChunker(AdvancedChunker):
    """Optimized chunker for processing large documents efficiently"""
    
    def __init__(self, 
                 chunk_size: int = 256,
                 overlap: int = 50,
                 encoding_model: str = "cl100k_base",
                 num_workers: int = 4):
        super().__init__(chunk_size, overlap, encoding_model)
        self.num_workers = num_workers
    
    def chunk_large_document(self, 
                           document: str, 
                           strategy: ChunkingStrategy,
                           chunk_threshold: int = 10000) -> List[Chunk]:
        """Chunk large documents using optimized strategies"""
        
        # For very large documents, first split into segments
        if len(document) > chunk_threshold:
            segments = self._create_segments(document, strategy)
            all_chunks = []
            
            # Process segments in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                process_func = partial(self._process_segment, strategy=strategy)
                segment_chunks_list = list(executor.map(process_func, segments))
            
            # Flatten the list of chunk lists
            for segment_chunks in segment_chunks_list:
                all_chunks.extend(segment_chunks)
            
            return all_chunks
        else:
            return self.chunk_document(document, strategy)
    
    def _create_segments(self, document: str, strategy: ChunkingStrategy) -> List[str]:
        """Create document segments for parallel processing"""
        if strategy in [ChunkingStrategy.FIXED_LENGTH, ChunkingStrategy.SLIDING_WINDOW]:
            # For these strategies, create segments of 10x chunk_size to avoid boundary issues
            segment_size = self.chunk_size * 15  # Add buffer for overlap handling
            segments = []
            
            for i in range(0, len(document), segment_size):
                # Add overlap between segments to handle boundary chunks properly
                end_pos = min(i + segment_size + self.overlap * 2, len(document))
                segments.append(document[i:end_pos])
            
            return segments
        else:
            # For semantic and recursive strategies, try to split at natural boundaries
            sentences = self._split_sentences(document)
            segments = []
            current_segment = ""
            
            for sentence in sentences:
                if len(current_segment) + len(sentence) > self.chunk_size * 5:  # 5x for buffer
                    segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += " " + sentence
            
            if current_segment:
                segments.append(current_segment)
            
            return segments
    
    def _process_segment(self, segment: str, strategy: ChunkingStrategy) -> List[Chunk]:
        """Process a single document segment"""
        return self.chunk_document(segment, strategy)
    
    def batch_chunk_documents(self, 
                            documents: List[str], 
                            strategy: ChunkingStrategy) -> List[List[Chunk]]:
        """Process multiple documents in batch"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            process_func = partial(self.chunk_document, strategy=strategy)
            results = list(executor.map(process_func, documents))
        
        return results
    
    def adaptive_chunking(self, 
                         text: str, 
                         target_chunks: int) -> List[Chunk]:
        """Adaptively determine chunk size to produce target number of chunks"""
        text_length = len(self.encoding.encode(text))
        optimal_chunk_size = max(self.chunk_size, text_length // target_chunks)
        
        # Adjust chunk size but keep it within reasonable bounds
        optimal_chunk_size = max(100, min(optimal_chunk_size, 1000))
        
        # Create a temporary chunker with the calculated size
        temp_chunker = AdvancedChunker(
            chunk_size=optimal_chunk_size,
            overlap=self.overlap,
            encoding_model=self.encoding.name
        )
        
        return temp_chunker.chunk_document(text, ChunkingStrategy.SEMANTIC)


class MemoryEfficientChunker:
    """Chunker designed to minimize memory usage for very large documents"""
    
    def __init__(self, 
                 chunk_size: int = 256,
                 overlap: int = 50,
                 encoding_model: str = "cl100k_base",
                 max_memory_chunks: int = 1000):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_model)
        self.max_memory_chunks = max_memory_chunks
    
    def chunk_document_streaming(self, 
                                file_path: str, 
                                strategy: ChunkingStrategy,
                                output_callback: Callable[[List[Chunk]], None]) -> None:
        """Process document in streaming fashion to minimize memory usage"""
        
        with open(file_path, 'r', encoding='utf-8') as file:
            buffer = ""
            chunk_count = 0
            chunk_id = 0
            
            while True:
                # Read in chunks to avoid loading entire file into memory
                chunk_data = file.read(1024 * 1024)  # 1MB chunks
                if not chunk_data:
                    break
                
                buffer += chunk_data
                
                # Process buffer when it reaches a reasonable size
                if len(buffer) > self.chunk_size * 10:
                    # Find a good breaking point (sentence boundary)
                    break_point = self._find_break_point(buffer)
                    
                    if break_point > 0:
                        text_segment = buffer[:break_point]
                        buffer = buffer[break_point:]
                        
                        # Chunk this segment
                        chunker = AdvancedChunker(self.chunk_size, self.overlap, self.encoding.name)
                        chunks = chunker.chunk_document(text_segment, strategy)
                        
                        # Update chunk IDs to be unique
                        for chunk in chunks:
                            chunk.id = f"stream_{chunk_id}"
                            chunk_id += 1
                        
                        output_callback(chunks)
                        chunk_count += len(chunks)
            
            # Process any remaining buffer content
            if buffer.strip():
                chunker = AdvancedChunker(self.chunk_size, self.overlap, self.encoding.name)
                chunks = chunker.chunk_document(buffer, strategy)
                
                for chunk in chunks:
                    chunk.id = f"stream_{chunk_id}"
                    chunk_id += 1
                
                output_callback(chunks)
                chunk_count += len(chunks)
    
    def _find_break_point(self, text: str) -> int:
        """Find a good point to break the text (sentence boundary)"""
        # Look for sentence endings
        for i in range(min(len(text), 2000), 0, -1):  # Look back up to 2000 chars
            if text[i] in '.!?':
                # Make sure we're not breaking in the middle of an abbreviation
                if i + 1 < len(text) and text[i+1].isspace():
                    return i + 1
        
        # If no sentence boundary found, return at a reasonable word boundary
        for i in range(min(len(text), 2000), 0, -1):
            if text[i].isspace():
                return i
        
        return len(text)  # Worst case: break at the end
```

## **5. Evaluation Framework for Chunking**

```python
class ChunkingEvaluator:
    def __init__(self):
        self.metrics = {
            'chunk_size_distribution': [],
            'semantic_boundary_preservation': [],
            'overlap_effectiveness': [],
            'context_preservation_score': [],
            'query_response_alignment': []
        }
    
    def evaluate_chunking_strategy(self, 
                                 original_text: str, 
                                 chunks: List[Chunk], 
                                 strategy: ChunkingStrategy) -> Dict[str, float]:
        """Comprehensive evaluation of a chunking strategy"""
        
        evaluation_results = {
            'strategy': strategy.value,
            'total_chunks': len(chunks),
            'avg_chunk_size': np.mean([len(self.encoding.encode(c.content)) for c in chunks]) if chunks else 0,
            'size_std': np.std([len(self.encoding.encode(c.content)) for c in chunks]) if chunks else 0,
            'text_preservation': self._check_text_preservation(original_text, chunks),
            'semantic_coherence': self._evaluate_semantic_coherence(chunks),
            'boundary_quality': self._evaluate_boundary_quality(original_text, chunks),
            'similarity_analysis': self._analyze_chunk_similarity(chunks)
        }
        
        return evaluation_results
    
    def _check_text_preservation(self, original: str, chunks: List[Chunk]) -> bool:
        """Check if all original text is preserved in chunks"""
        reconstructed = ''.join([chunk.content for chunk in chunks])
        
        # Simple check - in practice, you'd want more sophisticated comparison
        # accounting for overlaps and formatting
        return len(reconstructed) >= len(original) * 0.95  # Allow 5% difference for overlaps
    
    def _evaluate_semantic_coherence(self, chunks: List[Chunk]) -> float:
        """Evaluate how semantically coherent each chunk is"""
        if not chunks:
            return 0.0
        
        # Use sentence transformer to evaluate semantic coherence
        model = SentenceTransformer('all-MiniLM-L6-v2')
        coherence_scores = []
        
        for chunk in chunks:
            sentences = [s.strip() for s in re.split(r'[.!?]+', chunk.content) if s.strip()]
            
            if len(sentences) > 1:
                # Calculate similarity between consecutive sentences
                embeddings = model.encode(sentences)
                similarities = []
                
                for i in range(len(embeddings) - 1):
                    sim = np.dot(embeddings[i], embeddings[i+1]) / (
                        np.linalg.norm(embeddings[i]) * 
                        np.linalg.norm(embeddings[i+1])
                    )
                    similarities.append(sim)
                
                coherence_scores.append(np.mean(similarities))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _evaluate_boundary_quality(self, original: str, chunks: List[Chunk]) -> float:
        """Evaluate how well chunk boundaries align with natural text boundaries"""
        if len(chunks) < 2:
            return 1.0
        
        boundary_scores = []
        
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i].content[-20:] if len(chunks[i].content) > 20 else chunks[i].content
            chunk2_start = chunks[i+1].content[:20] if len(chunks[i+1].content) > 20 else chunks[i+1].content
            
            # Score based on natural boundary indicators
            score = 0.0
            
            # Check for sentence boundaries
            if any(boundary_char in chunk1_end[-5:] for boundary_char in '.!?'):
                score += 0.4
            
            # Check for paragraph boundaries
            if any(boundary_char in chunk1_end[-3:] for boundary_char in '\n\r'):
                score += 0.3
            
            # Check for natural starting points
            if chunk2_start[:2].strip() and chunk2_start[0].isupper():
                score += 0.3
            
            boundary_scores.append(min(score, 1.0))
        
        return np.mean(boundary_scores) if boundary_scores else 0.0
    
    def _analyze_chunk_similarity(self, chunks: List[Chunk]) -> Dict[str, float]:
        """Analyze similarity between adjacent and distant chunks"""
        if len(chunks) < 2:
            return {'avg_similarity': 0.0, 'overlap_similarity': 0.0}
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([chunk.content for chunk in chunks])
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * 
                np.linalg.norm(embeddings[i+1])
            )
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Also calculate similarity for chunks that should overlap
        overlap_similarities = []
        for i in range(len(chunks)):
            for j in range(i+2, min(i+5, len(chunks))):  # Check chunks 2-4 positions away
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * 
                    np.linalg.norm(embeddings[j])
                )
                overlap_similarities.append(sim)
        
        avg_overlap_similarity = np.mean(overlap_similarities) if overlap_similarities else 0.0
        
        return {
            'adjacent_similarity': avg_similarity,
            'distant_similarity': avg_overlap_similarity
        }
    
    def compare_strategies(self, 
                          original_text: str, 
                          strategies: List[ChunkingStrategy]) -> Dict[str, Dict[str, float]]:
        """Compare multiple chunking strategies on the same text"""
        results = {}
        
        chunker = AdvancedChunker()
        
        for strategy in strategies:
            chunks = chunker.chunk_document(original_text, strategy)
            results[strategy.value] = self.evaluate_chunking_strategy(original_text, chunks, strategy)
        
        return results
```

## **6. Production Deployment Considerations**

### **Scalable Chunking Service**

```python
import asyncio
import aiofiles
from typing import AsyncGenerator
import json
from dataclasses import asdict


class ScalableChunkingService:
    def __init__(self, 
                 chunk_size: int = 256,
                 overlap: int = 50,
                 max_workers: int = 8,
                 cache_size: int = 10000):
        self.chunker = AdvancedChunker(chunk_size, overlap)
        self.max_workers = max_workers
        self.cache = {}  # Simple cache - in production use Redis
        self.cache_size = cache_size
    
    async def process_document_async(self, 
                                   text: str, 
                                   strategy: ChunkingStrategy,
                                   doc_id: str) -> List[Chunk]:
        """Asynchronously process a document with caching"""
        
        # Create cache key
        cache_key = f"{doc_id}_{strategy.value}_{len(text)}_{hash(text[:100])}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Process document
        chunks = self.chunker.chunk_document(text, strategy)
        
        # Add metadata about processing
        for chunk in chunks:
            chunk.metadata['processed_by'] = 'scalable_chunking_service'
            chunk.metadata['doc_id'] = doc_id
            chunk.metadata['processing_timestamp'] = time.time()
        
        # Add to cache with size management
        if len(self.cache) >= self.cache_size:
            # Remove oldest entries (simplified LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = chunks
        
        return chunks
    
    async def process_batch_async(self, 
                                 documents: List[Dict], 
                                 strategy: ChunkingStrategy) -> List[List[Chunk]]:
        """Process multiple documents in parallel"""
        
        async def process_single(doc):
            return await self.process_document_async(
                doc['text'], 
                strategy, 
                doc.get('id', f"doc_{hash(doc['text'])}")
            )
        
        # Process in parallel
        tasks = [process_single(doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def stream_process_large_document(self, 
                                          file_path: str, 
                                          strategy: ChunkingStrategy,
                                          chunk_callback: Callable[[Chunk], None]) -> None:
        """Stream process a large document without loading it all into memory"""
        
        async with aiofiles.open(file_path, 'r') as file:
            buffer = ""
            
            while True:
                # Read in chunks
                chunk_data = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk_data:
                    break
                
                buffer += chunk_data
                
                # Process when buffer is large enough
                if len(buffer) > self.chunker.chunk_size * 10:
                    # Find a good breaking point
                    break_point = self._find_break_point(buffer)
                    
                    if break_point > 0:
                        text_segment = buffer[:break_point]
                        buffer = buffer[break_point:]
                        
                        # Process this segment
                        chunks = self.chunker.chunk_document(text_segment, strategy)
                        
                        # Send chunks via callback
                        for chunk in chunks:
                            await chunk_callback(chunk)
            
            # Process remaining buffer
            if buffer.strip():
                chunks = self.chunker.chunk_document(buffer, strategy)
                for chunk in chunks:
                    await chunk_callback(chunk)
    
    def _find_break_point(self, text: str) -> int:
        """Find a good point to break the text (sentence boundary)"""
        # Look for sentence endings
        for i in range(min(len(text), 2000), 0, -1):  # Look back up to 2000 chars
            if text[i] in '.!?':
                # Make sure we're not breaking in the middle of an abbreviation
                if i + 1 < len(text) and text[i+1].isspace():
                    return i + 1
        
        # If no sentence boundary found, return at a reasonable word boundary
        for i in range(min(len(text), 2000), 0, -1):
            if text[i].isspace():
                return i
        
        return len(text)  # Worst case: break at the end
    
    async def export_chunks_to_file(self, 
                                  chunks: List[Chunk], 
                                  output_path: str) -> None:
        """Export chunks to a structured file format"""
        
        async with aiofiles.open(output_path, 'w') as file:
            for chunk in chunks:
                chunk_dict = asdict(chunk)
                # Convert numpy array to list for JSON serialization
                if chunk_dict['embedding'] is not None:
                    chunk_dict['embedding'] = chunk_dict['embedding'].tolist()
                
                await file.write(json.dumps(chunk_dict) + '\n')


class ChunkingPipeline:
    """Complete pipeline for chunking with preprocessing and postprocessing"""
    
    def __init__(self, 
                 chunker: AdvancedChunker,
                 preprocessing_steps: Optional[List[Callable]] = None,
                 postprocessing_steps: Optional[List[Callable]] = None):
        self.chunker = chunker
        self.preprocessing_steps = preprocessing_steps or []
        self.postprocessing_steps = postprocessing_steps or []
    
    def process(self, 
               text: str, 
               strategy: ChunkingStrategy,
               metadata: Optional[Dict] = None) -> List[Chunk]:
        """Complete processing pipeline"""
        
        # Preprocessing
        processed_text = text
        for step in self.preprocessing_steps:
            processed_text = step(processed_text)
        
        # Chunking
        chunks = self.chunker.chunk_document(processed_text, strategy, metadata)
        
        # Postprocessing
        for step in self.postprocessing_steps:
            chunks = step(chunks)
        
        return chunks


# Preprocessing functions
def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and formatting issues"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize different types of whitespace to standard spaces"""
    text = re.sub(r'[\t\r\n\f\v]+', ' ', text)
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    return text.strip()


# Postprocessing functions
def validate_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Validate and clean up chunks"""
    valid_chunks = []
    
    for chunk in chunks:
        # Remove chunks that are too short
        if len(chunk.content.strip()) < 10:
            continue
        
        # Remove chunks that are mostly special characters
        text_chars = sum(1 for c in chunk.content if c.isalnum() or c.isspace())
        if text_chars / len(chunk.content) < 0.5:
            continue
        
        valid_chunks.append(chunk)
    
    return valid_chunks


def enrich_chunks_with_metadata(chunks: List[Chunk]) -> List[Chunk]:
    """Add additional metadata to chunks"""
    for i, chunk in enumerate(chunks):
        chunk.metadata['order'] = i
        chunk.metadata['length'] = len(chunk.content)
        chunk.metadata['token_count'] = len(tiktoken.get_encoding("cl100k_base").encode(chunk.content))
    
    return chunks


# Create a complete pipeline
def create_default_pipeline() -> ChunkingPipeline:
    """Create a default pipeline with recommended steps"""
    chunker = AdvancedChunker(chunk_size=256, overlap=50)
    
    preprocessing = [
        clean_text,
        normalize_whitespace
    ]
    
    postprocessing = [
        validate_chunks,
        enrich_chunks_with_metadata
    ]
    
    return ChunkingPipeline(chunker, preprocessing, postprocessing)
```

## **7. Testing and Validation**

### **Unit Tests for Chunking Module**

```python
import unittest
from unittest.mock import Mock, patch


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
        chunks = self.chunker.chunk_document(
            self.sample_text, 
            ChunkingStrategy.FIXED_LENGTH
        )
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) > 0 for chunk in chunks))
        
        # Check that text is preserved
        reconstructed = ' '.join([chunk.content for chunk in chunks])
        self.assertGreaterEqual(len(reconstructed), len(self.sample_text) * 0.9)
    
    def test_semantic_chunking(self):
        """Test semantic chunking strategy"""
        chunks = self.chunker.chunk_document(
            self.sample_text, 
            ChunkingStrategy.SEMANTIC
        )
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) > 0 for chunk in chunks))
    
    def test_recursive_chunking(self):
        """Test recursive chunking strategy"""
        chunks = self.chunker.chunk_document(
            self.sample_text, 
            ChunkingStrategy.RECURSIVE
        )
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) > 0 for chunk in chunks))
    
    def test_sliding_window_chunking(self):
        """Test sliding window chunking strategy"""
        chunks = self.chunker.chunk_document(
            self.sample_text, 
            ChunkingStrategy.SLIDING_WINDOW
        )
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(len(chunk.content) > 0 for chunk in chunks))
    
    def test_chunk_content_preservation(self):
        """Test that chunking preserves original content"""
        original_tokens = self.chunker.encoding.encode(self.sample_text)
        
        strategies = [
            ChunkingStrategy.FIXED_LENGTH,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.RECURSIVE
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


class TestChunkQualityEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for evaluator tests."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a lightweight model for tests
        self.evaluator = ChunkQualityEvaluator(model)
        self.sample_chunks = [
            Chunk(
                id="test1",
                content="This is the first chunk with meaningful content.",
                start_pos=0,
                end_pos=50,
                metadata={}
            ),
            Chunk(
                id="test2", 
                content="This is the second chunk with related content.",
                start_pos=50,
                end_pos=100,
                metadata={}
            )
        ]
    
    def test_evaluate_chunk_quality(self):
        """Test chunk quality evaluation"""
        quality_metrics = self.evaluator.evaluate_chunk_quality(self.sample_chunks)
        
        self.assertIn('semantic_coherence', quality_metrics)
        self.assertIn('avg_size', quality_metrics)
        self.assertIn('size_uniformity', quality_metrics)
        self.assertIn('boundary_quality', quality_metrics)
        
        # Check values are in reasonable ranges
        self.assertGreaterEqual(quality_metrics['semantic_coherence'], 0)
        self.assertLessEqual(quality_metrics['semantic_coherence'], 1)
        self.assertGreaterEqual(quality_metrics['size_uniformity'], 0)
        self.assertLessEqual(quality_metrics['size_uniformity'], 1)
        self.assertGreaterEqual(quality_metrics['boundary_quality'], 0)
        self.assertLessEqual(quality_metrics['boundary_quality'], 1)


def run_tests():
    """Run all tests in the module"""
    print("Running tests for Module 2: Chunking Strategies")
    print("=" * 50)
    
    # Create test suites
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestChunkingModule)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestChunkQualityEvaluator)
    
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
```

## **8. Hands-On Exercise**

### **Build a Custom Chunker with Domain Knowledge**

```python
# Exercise: Create a specialized chunker for academic papers
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


# Example usage and testing
def main():
    """Example usage of the Academic Paper Chunker"""
    print("RAG Course - Module 2: Chunking Strategies Example")
    print("=" * 55)
    
    # Sample academic paper text (simplified)
    sample_paper = """
Abstract
This paper explores the effectiveness of different document chunking strategies in Retrieval-Augmented Generation (RAG) systems. We analyze various approaches including fixed-length, semantic, and recursive chunking methods.

1. Introduction
The field of natural language processing has seen significant advances with the introduction of large language models. Retrieval-Augmented Generation (RAG) systems combine the generative power of LLMs with information retrieval capabilities to improve factual accuracy and reduce hallucination. The effectiveness of RAG systems depends heavily on how documents are chunked during the indexing phase.

2. Literature Review
Previous work on chunking strategies has focused primarily on fixed-length approaches. Smith et al. (2023) proposed a semantic chunking method that achieved promising results in academic document retrieval. However, their approach required significant computational resources.

3. Methodology
We conducted experiments comparing four different chunking strategies on a corpus of 1000 academic papers. Our evaluation metrics included retrieval accuracy, generation quality, and computational efficiency.

4. Results
Our findings indicate that semantic chunking outperforms fixed-length approaches by 15% in retrieval accuracy. Recursive chunking showed the best balance between performance and computational cost.

5. Discussion
The results suggest that preserving semantic boundaries during chunking is crucial for RAG system performance. However, the optimal strategy may vary depending on document type and domain.

6. Conclusion
This study demonstrates the importance of appropriate chunking strategies in RAG systems. Future work should explore hybrid approaches that combine multiple chunking methods.
"""
    
    # Initialize the academic paper chunker
    paper_chunker = AcademicPaperChunker(chunk_size=300, overlap=50)
    
    print("Chunking academic paper...")
    chunks = paper_chunker.chunk_academic_paper(sample_paper)
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Section: {chunk.metadata.get('section_type', 'unknown')}")
        print(f"  Content preview: {chunk.content[:100]}...")
        print(f"  Tokens: {len(paper_chunker.encoding.encode(chunk.content))}")
    
    # Validate the chunking
    validation_result = paper_chunker.validate_academic_chunking(sample_paper, chunks)
    print(f"\nValidation Results:")
    print(f"  Text Preserved: {validation_result['text_preserved']}")
    print(f"  Academic Structure Maintained: {validation_result['academic_structure_maintained']}")
    if validation_result['issues']:
        print(f"  Issues: {validation_result['issues']}")


if __name__ == "__main__":
    main()
```

This deep dive into chunking strategies provides comprehensive coverage from theoretical foundations to production-ready implementations. The module includes:

- **Advanced theoretical concepts** with mathematical foundations
- **Multiple chunking strategies** with detailed implementations
- **Content-aware chunking** for structured documents (HTML, code, legal, academic)
- **Performance optimization** techniques for large-scale processing
- **Quality evaluation** frameworks for measuring chunk effectiveness
- **Production deployment** considerations with scalable services
- **Testing strategies** for validation
- **Hands-on exercises** for practical learning