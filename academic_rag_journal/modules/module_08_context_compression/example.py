"""
Module 8: Context Compression
Implementation Examples

This module demonstrates context compression techniques to optimize
the amount of information passed to language models in RAG systems.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import openai
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import uuid
from itertools import compress


class ContextCompressorBase:
    """Base class for context compression"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def compress(self, context: List[str], query: str, target_length: int = 2000) -> List[str]:
        """Compress context to target length"""
        raise NotImplementedError
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        # Simple approximation: 1 token ≈ 4 characters or 1 word
        return len(text.split())


class LLMContextCompressor(ContextCompressorBase):
    """Compress context using language models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        super().__init__()
        
        try:
            # Initialize transformer model for summarization
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer
            )
            self.use_llm = True
        except:
            # Fallback to simple methods
            self.use_llm = False
            print("LLM model failed to load, using rule-based compression")
    
    def compress(self, context: List[str], query: str, target_length: int = 2000) -> List[str]:
        """Compress context by summarizing with LLM"""
        if not self.use_llm:
            # Fallback to simple truncation-based method
            return self._simple_compression(context, target_length)
        
        # Join context into a single text for summarization
        full_text = " ".join(context)
        
        if len(full_text) < target_length:
            return context  # No compression needed
        
        try:
            # Estimate number of characters to summarize to
            # Note: This is a simplified approach; in practice, you'd work with token limits
            max_length = min(len(full_text) // 2, target_length)
            min_length = max_length // 2
            
            # Generate summary
            summary_result = self.summarizer(
                full_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary_text = summary_result[0]['summary_text']
            return [summary_text]
        except:
            # If summarization fails, fall back to rule-based method
            return self._simple_compression(context, target_length)
    
    def _simple_compression(self, context: List[str], target_length: int = 2000) -> List[str]:
        """Rule-based fallback for compression"""
        # Simple truncation to stay within target length
        current_length = sum(len(doc) for doc in context)
        
        if current_length <= target_length:
            return context
        
        compressed = []
        current_length = 0
        
        for doc in context:
            if current_length + len(doc) <= target_length:
                compressed.append(doc)
                current_length += len(doc)
            else:
                # Add partial document if needed
                remaining = target_length - current_length
                if remaining > 0:
                    compressed.append(doc[:remaining])
                break
        
        return compressed


class EmbeddingBasedCompressor(ContextCompressorBase):
    """Compress context based on embedding similarity to query"""
    
    def __init__(self):
        super().__init__()
    
    def compress(self, context: List[str], query: str, target_length: int = 2000) -> List[str]:
        """Compress context by keeping most relevant sentences"""
        # Encode query and all context sentences
        query_embedding = self.embedder.encode([query])
        context_embeddings = self.embedder.encode(context)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, context_embeddings)[0]
        
        # Create list of (index, similarity) pairs
        indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
        
        # Sort by similarity (descending)
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select sentences until we reach the target length
        selected = []
        current_length = 0
        
        for idx, similarity in indexed_similarities:
            doc = context[idx]
            if current_length + len(doc) <= target_length:
                selected.append(doc)
                current_length += len(doc)
            else:
                break
        
        # If we still exceed the limit, try to truncate individual documents
        if current_length > target_length:
            return self._truncate_to_target(selected, target_length)
        
        return selected
    
    def _truncate_to_target(self, documents: List[str], target_length: int) -> List[str]:
        """Truncate documents to fit within target length"""
        current_length = sum(len(doc) for doc in documents)
        
        if current_length <= target_length:
            return documents
        
        # Calculate reduction ratio
        ratio = target_length / current_length
        
        truncated_docs = []
        remaining_length = target_length
        
        for doc in documents:
            # Calculate how much of this document we can keep
            doc_length = len(doc)
            allowed_length = int(doc_length * ratio)
            
            # Adjust based on remaining space
            allowed_length = min(allowed_length, remaining_length)
            
            if allowed_length > 0:
                truncated_docs.append(doc[:allowed_length])
                remaining_length -= allowed_length
            else:
                break
            
            if remaining_length <= 0:
                break
        
        return truncated_docs


class RelevanceBasedCompressor(ContextCompressorBase):
    """Compress context based on relevance to query using cross-attention"""
    
    def __init__(self):
        super().__init__()
        
        # For this example, we'll use embedding similarity as a proxy for relevance
        # A full implementation would use a cross-encoder model
        pass
    
    def compress(self, context: List[str], query: str, target_length: int = 2000) -> List[str]:
        """Compress context by scoring relevance and selecting top documents"""
        # Create a simple relevance score based on keyword overlap and embedding similarity
        query_words = set(query.lower().split())
        
        scores = []
        for i, doc in enumerate(context):
            doc_words = set(doc.lower().split())
            
            # Calculate keyword overlap
            overlap = len(query_words.intersection(doc_words))
            
            # Calculate embedding similarity
            query_embedding = self.embedder.encode([query])
            doc_embedding = self.embedder.encode([doc])
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            
            # Combine scores
            combined_score = 0.3 * overlap + 0.7 * similarity  # Weighted combination
            scores.append((i, combined_score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select documents until target length is reached
        selected = []
        current_length = 0
        
        for idx, score in scores:
            doc = context[idx]
            if current_length + len(doc) <= target_length:
                selected.append(doc)
                current_length += len(doc)
            else:
                break
        
        return selected


class SummaryBasedCompressor(ContextCompressorBase):
    """Compress context using extractive summarization techniques"""
    
    def __init__(self):
        super().__init__()
    
    def compress(self, context: List[str], query: str, target_length: int = 2000) -> List[str]:
        """Compress context using extractive summarization"""
        # For this example, we'll use a variation of the embedding-based approach
        # to select the most important sentences
        
        # First, split documents into sentences if they're long
        all_sentences = []
        original_mapping = []  # Track which document each sentence came from
        
        for doc_idx, doc in enumerate(context):
            # Split into sentences (simple split on period)
            sentences = re.split(r'[.!?]+', doc)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for sent in sentences:
                if len(sent) > 10:  # Only consider sentences with more than 10 characters
                    all_sentences.append(sent)
                    original_mapping.append(doc_idx)
        
        if not all_sentences:
            return []
        
        # Calculate relevance of each sentence to query
        query_embedding = self.embedder.encode([query])
        sentence_embeddings = self.embedder.encode(all_sentences)
        
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
        
        # Create relevance scores with position as secondary factor
        sentence_scores = [(i, sim, len(sent)) for i, (sent, sim) in enumerate(zip(all_sentences, similarities))]
        
        # Sort by relevance (descending)
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select sentences until target length is reached
        selected_sentences = []
        current_length = 0
        
        for idx, score, sent_len in sentence_scores:
            if current_length + sent_len <= target_length:
                selected_sentences.append(all_sentences[idx])
                current_length += sent_len
            else:
                break
        
        # Join sentences back together
        if selected_sentences:
            return [" ".join(selected_sentences)]
        else:
            # If no sentences were selected, return top documents from original context
            return EmbeddingBasedCompressor().compress(context, query, target_length)


class TokenEfficientCompressor:
    """Main compressor that combines multiple techniques for token efficiency"""
    
    def __init__(self, primary_method: str = "embedding"):
        self.primary_method = primary_method
        
        # Initialize all compressors
        self.llm_compressor = LLMContextCompressor()
        self.embedding_compressor = EmbeddingBasedCompressor()
        self.relevance_compressor = RelevanceBasedCompressor()
        self.summary_compressor = SummaryBasedCompressor()
        
        # Map method names to compressors
        self.compressor_map = {
            'llm': self.llm_compressor,
            'embedding': self.embedding_compressor,
            'relevance': self.relevance_compressor,
            'summary': self.summary_compressor
        }
    
    def compress(self, context: List[str], query: str, target_tokens: int = 1000, 
                 strategy: str = "primary") -> Dict[str, Any]:
        """Compress context using specified strategy"""
        
        # Convert target_tokens to approximate character count
        # (assuming ~4 characters per token)
        target_length = target_tokens * 4
        
        start_time = time.time()
        
        if strategy == "primary":
            # Use the primary specified method
            compressor = self.compressor_map.get(self.primary_method, self.embedding_compressor)
            compressed = compressor.compress(context, query, target_length)
        elif strategy == "ensemble":
            # Try multiple methods and return the best one based on some criteria
            results = {}
            
            for method, comp in self.compressor_map.items():
                compressed = comp.compress(context, query, target_length)
                results[method] = {
                    'compressed': compressed,
                    'length': sum(len(doc) for doc in compressed),
                    'count': len(compressed)
                }
            
            # For now, return the method that achieves closest to target while staying under
            best_method = self.primary_method
            best_result = results[best_method]
            
            for method, result in results.items():
                if result['length'] <= target_length and result['length'] > best_result['length']:
                    best_method = method
                    best_result = result
            
            compressed = results[best_method]['compressed']
        else:
            # Default to embedding-based
            compressed = self.embedding_compressor.compress(context, query, target_length)
        
        compression_time = time.time() - start_time
        
        # Calculate compression metrics
        original_length = sum(len(doc) for doc in context)
        compressed_length = sum(len(doc) for doc in compressed)
        compression_ratio = compressed_length / original_length if original_length > 0 else 0
        
        return {
            'original_context': context,
            'compressed_context': compressed,
            'original_length': original_length,
            'compressed_length': compressed_length,
            'compression_ratio': compression_ratio,
            'tokens_saved': (original_length - compressed_length) // 4,  # Approximate
            'compression_time': compression_time,
            'method_used': self.primary_method if strategy == "primary" else "ensemble"
        }


class ContextCompressionEvaluator:
    """Evaluator for context compression techniques"""
    
    def __init__(self):
        self.sample_contexts = [
            [
                "Machine learning algorithms are methods to train models using data.",
                "Deep learning neural networks require significant computational resources.",
                "Natural language processing uses machine learning to understand text.",
                "Data science combines statistics, programming, and domain expertise.",
                "Vector databases optimize similarity search for high-dimensional vectors.",
                "Reinforcement learning uses rewards to train agents.",
                "Supervised learning requires labeled training examples.",
                "Unsupervised learning discovers patterns in unlabeled data."
            ],
            [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming multiple industries.",
                "Neural networks learn patterns from large datasets.",
                "Python is a popular programming language for AI.",
                "Information retrieval systems find relevant documents.",
                "Query expansion improves search effectiveness.",
                "Context compression optimizes information for LLMs.",
                "Retrieval augmented generation enhances responses."
            ]
        ]
        
        self.sample_queries = [
            "machine learning algorithms",
            "artificial intelligence applications"
        ]
    
    def evaluate_compression_method(self, compressor: TokenEfficientCompressor, 
                                  target_tokens: int = 500) -> Dict[str, Any]:
        """Evaluate a specific compression method"""
        results = []
        
        total_original_length = 0
        total_compressed_length = 0
        total_compression_time = 0
        total_contexts_processed = 0
        
        for i, context in enumerate(self.sample_contexts):
            query = self.sample_queries[i % len(self.sample_queries)]
            
            comp_result = compressor.compress(context, query, target_tokens)
            
            results.append(comp_result)
            
            total_original_length += comp_result['original_length']
            total_compressed_length += comp_result['compressed_length']
            total_compression_time += comp_result['compression_time']
            total_contexts_processed += 1
        
        return {
            'method': compressor.primary_method if hasattr(compressor, 'primary_method') else 'unknown',
            'avg_compression_ratio': total_compressed_length / total_original_length if total_original_length > 0 else 0,
            'total_tokens_saved': (total_original_length - total_compressed_length) // 4,
            'avg_compression_time': total_compression_time / total_contexts_processed if total_contexts_processed > 0 else 0,
            'contexts_processed': total_contexts_processed
        }
    
    def compare_methods(self, target_tokens: int = 500) -> List[Dict[str, Any]]:
        """Compare different compression methods"""
        methods = ['embedding', 'relevance', 'summary', 'llm']
        results = []
        
        for method in methods:
            compressor = TokenEfficientCompressor(primary_method=method)
            result = self.evaluate_compression_method(compressor, target_tokens)
            results.append(result)
        
        return results


def demonstrate_context_compression():
    """Demonstrate context compression techniques"""
    print("=== Context Compression Demonstration ===\n")
    
    # Sample context and query
    sample_context = [
        "Machine learning algorithms are methods to train models using data. These algorithms can be classified into supervised, unsupervised, and reinforcement learning approaches.",
        "Deep learning neural networks require significant computational resources. They are composed of multiple layers that learn hierarchical representations of data.",
        "Natural language processing uses machine learning to understand text. Modern NLP systems employ transformer architectures for various tasks like translation and summarization.",
        "Data science combines statistics, programming, and domain expertise. It involves collecting, cleaning, analyzing, and interpreting large datasets to extract meaningful insights.",
        "Vector databases optimize similarity search for high-dimensional vectors. They use specialized indexing techniques to efficiently find similar embeddings.",
        "Reinforcement learning uses rewards to train agents. The agent learns to make decisions by interacting with an environment and receiving feedback.",
        "Supervised learning requires labeled training examples. The model learns to map inputs to outputs based on example input-output pairs.",
        "Unsupervised learning discovers patterns in unlabeled data. It can identify clusters, reduce dimensionality, or detect anomalies without explicit guidance."
    ]
    
    query = "machine learning algorithms"
    
    print(f"Original context: {sum(len(doc) for doc in sample_context)} characters in {len(sample_context)} documents")
    print(f"Query: '{query}'\n")
    
    # Test different compression methods
    compressors = {
        'Embedding-based': TokenEfficientCompressor(primary_method='embedding'),
        'Relevance-based': TokenEfficientCompressor(primary_method='relevance'),
        'Summary-based': TokenEfficientCompressor(primary_method='summary'),
        'LLM-based': TokenEfficientCompressor(primary_method='llm')
    }
    
    target_tokens = 600  # Target for compression
    
    for name, compressor in compressors.items():
        result = compressor.compress(sample_context, query, target_tokens)
        print(f"{name} compression:")
        print(f"  Original: {result['original_length']} chars")
        print(f"  Compressed: {result['compressed_length']} chars")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}")
        print(f"  Tokens saved: ~{result['tokens_saved']}")
        print(f"  Time: {result['compression_time']:.4f}s")
        print(f"  Selected {len(result['compressed_context'])} segments")
        print()


def performance_comparison():
    """Compare performance of different compression approaches"""
    print("=== Performance Comparison ===\n")
    
    evaluator = ContextCompressionEvaluator()
    results = evaluator.compare_methods(target_tokens=400)
    
    print(f"{'Method':<15} {'Avg Ratio':<10} {'Tokens Saved':<12} {'Time (s)':<10} {'Contexts':<8}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['method']:<15} {result['avg_compression_ratio']:<10.2f} "
              f"{result['total_tokens_saved']:<12} {result['avg_compression_time']:<10.4f} "
              f"{result['contexts_processed']:<8}")


def main():
    """Main function to demonstrate context compression implementations"""
    print("Module 8: Context Compression")
    print("=" * 50)
    
    # Demonstrate context compression
    demonstrate_context_compression()
    
    # Show performance comparison
    performance_comparison()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of detailed processing
    large_context = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by animals including humans.",
        "Leading AI textbooks define the field as the study of intelligent agents: any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
        "Colloquially, the term artificial intelligence is often used to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.",
        "As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect.",
        "A quip in Tesler's Theorem says AI is whatever has not been done yet.",
        "Modern machine learning techniques are at the heart of AI. Problems for AI applications include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects.",
        "General intelligence is among the field's long-term goals. Approaches include statistical methods, computational intelligence, and traditional symbolic AI.",
        "Many tools are used in AI, including versions of search and mathematical optimization, artificial neural networks, and methods based on statistics, probability and economics."
    ]
    
    query = "artificial intelligence definition"
    
    print(f"Processing large context with query: '{query}'")
    
    # Use embedding-based compressor
    compressor = TokenEfficientCompressor(primary_method='embedding')
    result = compressor.compress(large_context, query, target_tokens=800, strategy="primary")
    
    print(f"\nCompression Results:")
    print(f"  Original length: {result['original_length']} characters")
    print(f"  Compressed length: {result['compressed_length']} characters")
    print(f"  Compression ratio: {result['compression_ratio']:.2f}")
    print(f"  Estimated tokens saved: {result['tokens_saved']}")
    print(f"  Compression time: {result['compression_time']:.4f} seconds")
    print(f"  Method: {result['method_used']}")
    
    print(f"\nOriginal context had {len(large_context)} documents, compressed to {len(result['compressed_context'])} segments")
    
    print(f"\nFirst compressed segment preview:")
    if result['compressed_context']:
        preview = result['compressed_context'][0][:200] + "..." if len(result['compressed_context'][0]) > 200 else result['compressed_context'][0]
        print(f"  '{preview}'")


if __name__ == "__main__":
    main()