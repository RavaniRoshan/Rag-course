"""
Module 7: Multi-Query Expansion
Implementation Examples

This module demonstrates multi-query expansion techniques to improve
retrieval coverage and accuracy in RAG systems.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import openai
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import uuid
from itertools import combinations


class QueryGeneratorBase:
    """Base class for query generation"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_queries(self, original_query: str, n_queries: int = 3) -> List[str]:
        """Generate multiple queries from an original query"""
        raise NotImplementedError


class LLMQueryGenerator(QueryGeneratorBase):
    """Generate queries using language models"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        super().__init__()
        
        try:
            # Initialize transformer model for query generation
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            self.use_llm = True
        except:
            # Fallback to simple rule-based generation
            self.use_llm = False
            print("LLM model failed to load, using rule-based generation")
    
    def generate_queries(self, original_query: str, n_queries: int = 3) -> List[str]:
        """Generate queries using LLM"""
        if self.use_llm:
            # For query expansion, we'll use a prompt to generate variations
            prompt = f"Generate 3 different ways to ask this question: {original_query}"
            
            try:
                results = self.generator(
                    prompt,
                    max_length=100,
                    num_return_sequences=n_queries,
                    temperature=0.7,
                    do_sample=True
                )
                
                queries = []
                for result in results:
                    # Extract the generated text
                    generated_text = result['generated_text']
                    # This is a simplified extraction - in practice, you'd have a more sophisticated way to parse
                    queries.append(generated_text)
                
                return queries[:n_queries]
            except:
                # If LLM generation fails, fall back to rule-based
                return self._rule_based_generation(original_query, n_queries)
        else:
            return self._rule_based_generation(original_query, n_queries)
    
    def _rule_based_generation(self, original_query: str, n_queries: int = 3) -> List[str]:
        """Rule-based fallback for query generation"""
        queries = [original_query]
        
        # Add simple variations
        variations = [
            f"what is {original_query.lower()}",
            f"explain {original_query.lower()}",
            f"how does {original_query.lower()} work",
            f"{original_query.lower()} definition",
            f"{original_query.lower()} tutorial"
        ]
        
        # Add some synonyms-based variations (simplified)
        synonyms = {
            "machine learning": ["ML", "algorithmic learning", "statistical learning"],
            "neural network": ["NN", "artificial neural network", "deep learning model"],
            "artificial intelligence": ["AI", "machine intelligence", "intelligent systems"]
        }
        
        for term, syns in synonyms.items():
            if term.lower() in original_query.lower():
                for syn in syns[:2]:  # Use first 2 synonyms
                    new_query = original_query.lower().replace(term.lower(), syn)
                    variations.append(new_query)
        
        # Combine with original and return requested number
        all_queries = queries + variations
        return list(set(all_queries))[:n_queries]


class SemanticQueryExpander(QueryGeneratorBase):
    """Expand queries using semantic similarity"""
    
    def __init__(self):
        super().__init__()
        
        # Predefined terms related to common concepts in our domain
        self.semantic_related_terms = {
            "machine learning": ["artificial intelligence", "ML", "algorithm", "model", "supervised learning", "unsupervised learning"],
            "neural network": ["deep learning", "artificial neural network", "backpropagation", "activation function"],
            "natural language processing": ["NLP", "language model", "text processing", "sentiment analysis", "tokenization"],
            "data science": ["data analysis", "statistics", "machine learning", "big data", "data mining"],
            "vector database": ["embedding", "similarity search", "high-dimensional space", "indexing"],
            "retrieval augmented generation": ["RAG", "information retrieval", "document retrieval", "context augmentation"]
        }
    
    def generate_queries(self, original_query: str, n_queries: int = 3) -> List[str]:
        """Generate queries using semantic expansion"""
        queries = [original_query]
        
        # Find related terms based on semantic similarity
        query_lower = original_query.lower()
        found_related = False
        
        for concept, related_terms in self.semantic_related_terms.items():
            if concept in query_lower:
                # Add variations with related terms
                for related_term in related_terms[:2]:  # Limit to first 2 related terms
                    new_query = query_lower.replace(concept, related_term)
                    queries.append(new_query)
                found_related = True
                break
        
        # If no specific concepts found, use embedding-based similarity
        if not found_related:
            # Get embedding for original query
            query_embedding = self.embedder.encode([original_query])
            
            # Use some predefined related terms based on common query patterns
            related_terms = [
                "definition of " + original_query,
                "tutorial about " + original_query,
                "example of " + original_query,
                "how to " + original_query,
                "what is " + original_query
            ]
            
            # Check semantic similarity
            additional_terms = []
            for term in related_terms:
                term_embedding = self.embedder.encode([term])
                similarity = cosine_similarity(query_embedding, term_embedding)[0][0]
                if similarity > 0.3:  # Threshold for similarity
                    additional_terms.append(term)
            
            queries.extend(additional_terms)
        
        return list(set(queries))[:n_queries]


class ParaphraseQueryGenerator(QueryGeneratorBase):
    """Generate queries through paraphrasing techniques"""
    
    def __init__(self):
        super().__init__()
        
        # Predefined paraphrasing patterns and synonyms
        self.paraphrase_patterns = [
            ("what is the", "definition of"),
            ("what is the", "explanation of"),
            ("how to", "steps for"),
            ("how to", "guide for"),
            ("what are", "list of"),
            ("tell me about", "explain"),
            ("explain", "describe"),
            ("describe", "outline"),
            ("steps to", "how to")
        ]
    
    def generate_queries(self, original_query: str, n_queries: int = 3) -> List[str]:
        """Generate queries using paraphrasing techniques"""
        queries = [original_query]
        
        # Apply paraphrasing patterns
        for old_pattern, new_pattern in self.paraphrase_patterns:
            if old_pattern in original_query.lower():
                new_query = original_query.lower().replace(old_pattern, new_pattern, 1)
                queries.append(new_query)
                if len(queries) >= n_queries:
                    break
        
        # Add synonyms for common terms
        synonym_map = {
            "build": ["create", "develop", "construct"],
            "learn": ["study", "understand", "master"],
            "use": ["utilize", "apply", "employ"],
            "implement": ["execute", "deploy", "apply"],
            "find": ["locate", "search", "discover"]
        }
        
        original_words = original_query.split()
        for i, word in enumerate(original_words):
            if word.lower() in synonym_map:
                for synonym in synonym_map[word.lower()]:
                    new_words = original_words.copy()
                    new_words[i] = synonym
                    new_query = " ".join(new_words)
                    queries.append(new_query)
                    if len(queries) >= n_queries:
                        return list(set(queries))
        
        return list(set(queries))[:n_queries]


class MultiQueryRetriever:
    """Class to handle multi-query retrieval and result combination"""
    
    def __init__(self, query_generator: QueryGeneratorBase = None):
        self.query_generator = query_generator or LLMQueryGenerator()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve_with_expansion(self, original_query: str, documents: List[str], 
                               n_expanded_queries: int = 3, top_k_per_query: int = 5) -> Dict[str, Any]:
        """Retrieve documents using multi-query expansion"""
        
        # Generate expanded queries
        expanded_queries = self.query_generator.generate_queries(original_query, n_expanded_queries)
        
        # Retrieve documents for each query
        all_results = []
        query_times = []
        
        for query in expanded_queries:
            start_time = time.time()
            query_results = self._retrieve_for_single_query(query, documents, top_k_per_query)
            retrieval_time = time.time() - start_time
            query_times.append(retrieval_time)
            
            all_results.append({
                'query': query,
                'results': query_results,
                'time': retrieval_time
            })
        
        # Combine results from all queries
        combined_results = self._combine_results(all_results, documents)
        
        return {
            'original_query': original_query,
            'expanded_queries': expanded_queries,
            'individual_results': all_results,
            'combined_results': combined_results,
            'avg_query_time': sum(query_times) / len(query_times),
            'total_time': sum(query_times)
        }
    
    def _retrieve_for_single_query(self, query: str, documents: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Retrieve documents for a single query using embedding similarity"""
        # Encode query and documents
        query_embedding = self.embedder.encode([query])
        doc_embeddings = self.embedder.encode(documents)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'doc_id': int(idx),
                'document': documents[idx],
                'similarity': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
    
    def _combine_results(self, all_results: List[Dict], documents: List[str]) -> List[Dict[str, Any]]:
        """Combine results from multiple queries"""
        # Create a mapping of document to total score
        doc_scores = {}
        doc_rank_sums = {}
        
        for result_set in all_results:
            for result in result_set['results']:
                doc_id = result['doc_id']
                similarity = result['similarity']
                rank = result['rank']
                
                # Accumulate similarity scores
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_rank_sums[doc_id] = 0
                
                doc_scores[doc_id] += similarity
                doc_rank_sums[doc_id] += rank
        
        # Calculate combined scores using RRF (Reciprocal Rank Fusion)
        final_scores = {}
        for doc_id in doc_scores:
            # RRF formula: score = sum(1 / (k + rank)) where k is smoothing parameter (usually 60)
            rrf_score = 0.0
            for result_set in all_results:
                for result in result_set['results']:
                    if result['doc_id'] == doc_id:
                        rank = result['rank']
                        rrf_score += 1.0 / (60 + rank)
            
            # Combine RRF score with similarity score
            final_scores[doc_id] = {
                'rrf_score': rrf_score,
                'avg_similarity': doc_scores[doc_id] / len(all_results),
                'combined_score': rrf_score + doc_scores[doc_id] / len(all_results)
            }
        
        # Rank final results by combined score
        sorted_doc_ids = sorted(final_scores.keys(), 
                               key=lambda x: final_scores[x]['combined_score'], 
                               reverse=True)
        
        final_results = []
        for i, doc_id in enumerate(sorted_doc_ids):
            final_results.append({
                'doc_id': doc_id,
                'document': documents[doc_id],
                'rrf_score': final_scores[doc_id]['rrf_score'],
                'avg_similarity': final_scores[doc_id]['avg_similarity'],
                'combined_score': final_scores[doc_id]['combined_score'],
                'rank': i + 1
            })
        
        return final_results


class MultiQueryEvaluator:
    """Evaluator for multi-query expansion techniques"""
    
    def __init__(self):
        self.sample_queries = [
            "machine learning algorithms",
            "neural network training",
            "natural language processing",
            "data science techniques",
            "vector database optimization"
        ]
        
        self.sample_docs = [
            "Machine learning algorithms are methods to train models using data",
            "Deep learning neural networks require significant computational resources",
            "Natural language processing uses machine learning to understand text",
            "Data science combines statistics, programming, and domain expertise",
            "Vector databases optimize similarity search for high-dimensional vectors",
            "Neural networks learn patterns from large datasets",
            "Supervised learning requires labeled training examples",
            "Unsupervised learning discovers patterns in unlabeled data",
            "Transformer models revolutionized natural language processing",
            "Reinforcement learning uses rewards to train agents"
        ]
    
    def compare_generators(self, n_queries: int = 3) -> Dict[str, Any]:
        """Compare different query generation methods"""
        generators = {
            'LLM': LLMQueryGenerator(),
            'Semantic': SemanticQueryExpander(),
            'Paraphrase': ParaphraseQueryGenerator()
        }
        
        comparison_results = {}
        
        for name, generator in generators.items():
            total_expansion_time = 0
            all_generated_queries = []
            
            for query in self.sample_queries:
                start_time = time.time()
                generated = generator.generate_queries(query, n_queries)
                expansion_time = time.time() - start_time
                
                total_expansion_time += expansion_time
                all_generated_queries.extend(generated)
            
            comparison_results[name] = {
                'avg_generation_time': total_expansion_time / len(self.sample_queries),
                'total_generated_queries': len(all_generated_queries),
                'queries_per_original': len(all_generated_queries) / len(self.sample_queries)
            }
        
        return comparison_results
    
    def evaluate_retrieval_improvement(self) -> Dict[str, Any]:
        """Evaluate retrieval improvement with multi-query expansion"""
        # Single query retrieval
        single_retriever = MultiQueryRetriever()
        # We'll simulate single query by using just the original
        single_query_time = 0
        
        # Multi-query retrieval
        llm_generator = LLMQueryGenerator()
        multi_retriever = MultiQueryRetriever(llm_generator)
        
        multi_query_times = []
        
        for query in self.sample_queries:
            start_time = time.time()
            multi_results = multi_retriever.retrieve_with_expansion(
                query, self.sample_docs, n_expanded_queries=3
            )
            multi_time = time.time() - start_time
            multi_query_times.append(multi_time)
        
        return {
            'avg_multi_retrieval_time': sum(multi_query_times) / len(multi_query_times),
            'total_multi_retrieval_time': sum(multi_query_times),
            'num_queries_tested': len(self.sample_queries)
        }


def demonstrate_multi_query_expansion():
    """Demonstrate multi-query expansion techniques"""
    print("=== Multi-Query Expansion Demonstration ===\n")
    
    # Sample query and documents
    original_query = "machine learning algorithms"
    sample_docs = [
        "Machine learning algorithms are methods to train models using data",
        "Deep learning neural networks require significant computational resources", 
        "Natural language processing uses machine learning to understand text",
        "Data science combines statistics, programming, and domain expertise",
        "Vector databases optimize similarity search for high-dimensional vectors",
        "Neural networks learn patterns from large datasets",
        "Supervised learning requires labeled training examples",
        "Unsupervised learning discovers patterns in unlabeled data",
        "Transformer models revolutionized natural language processing",
        "Reinforcement learning uses rewards to train agents"
    ]
    
    print(f"Original query: '{original_query}'\n")
    
    # Test different generators
    generators = {
        'LLM-based': LLMQueryGenerator(),
        'Semantic': SemanticQueryExpander(),
        'Paraphrase': ParaphraseQueryGenerator()
    }
    
    for name, generator in generators.items():
        print(f"{name} generator:")
        try:
            queries = generator.generate_queries(original_query, n_queries=4)
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")
        except Exception as e:
            print(f"  Error: {e}")
        print()
    
    # Demonstrate multi-query retrieval
    print("Multi-query retrieval example:")
    multi_retriever = MultiQueryRetriever(LLMQueryGenerator())
    results = multi_retriever.retrieve_with_expansion(original_query, sample_docs)
    
    print(f"Expanded queries: {results['expanded_queries']}")
    print(f"Combined results (top 3):")
    for i, result in enumerate(results['combined_results'][:3], 1):
        print(f"  {i}. Score: {result['combined_score']:.3f} - {result['document']}")
    print()


def performance_comparison():
    """Compare performance of different multi-query approaches"""
    print("=== Performance Comparison ===\n")
    
    evaluator = MultiQueryEvaluator()
    
    # Compare generators
    gen_results = evaluator.compare_generators()
    
    print("Query Generation Performance:")
    print(f"{'Method':<12} {'Avg Time (s)':<12} {'Queries Generated':<15} {'Queries/Original':<15}")
    print("-" * 60)
    for method, metrics in gen_results.items():
        print(f"{method:<12} {metrics['avg_generation_time']:<12.4f} "
              f"{metrics['total_generated_queries']:<15} {metrics['queries_per_original']:<15.1f}")
    print()
    
    # Evaluate retrieval improvement
    retrieval_results = evaluator.evaluate_retrieval_improvement()
    print("Retrieval Performance:")
    print(f"Average multi-query retrieval time: {retrieval_results['avg_multi_retrieval_time']:.4f}s")
    print(f"Total retrieval time for {retrieval_results['num_queries_tested']} queries: {retrieval_results['total_multi_retrieval_time']:.4f}s")


def main():
    """Main function to demonstrate multi-query expansion implementations"""
    print("Module 7: Multi-Query Expansion")
    print("=" * 50)
    
    # Demonstrate multi-query expansion
    demonstrate_multi_query_expansion()
    
    # Show performance comparison
    performance_comparison()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of detailed processing
    original_query = "neural network training"
    sample_docs = [
        "Neural networks learn patterns from large datasets",
        "Training neural networks requires labeled examples", 
        "Deep learning models need significant computational resources",
        "Machine learning algorithms process data to find patterns",
        "Supervised learning uses labeled training examples",
        "Backpropagation is an algorithm for training neural networks",
        "Gradient descent optimizes neural network weights",
        "Convolutional neural networks are used for image processing"
    ]
    
    print(f"Processing query: '{original_query}'")
    
    # Use semantic expander
    semantic_expander = SemanticQueryExpander()
    expanded_queries = semantic_expander.generate_queries(original_query, n_queries=3)
    print(f"Expanded queries: {expanded_queries}")
    
    # Use multi-query retriever
    retriever = MultiQueryRetriever(semantic_expander)
    results = retriever.retrieve_with_expansion(original_query, sample_docs, n_expanded_queries=2)
    
    print(f"Individual Query Results:")
    for i, query_result in enumerate(results['individual_results']):
        print(f"  Query {i+1} ('{query_result['query']}'): {len(query_result['results'])} results in {query_result['time']:.4f}s")
    
    print(f"\nCombined Results (top 4):")
    for i, result in enumerate(results['combined_results'][:4], 1):
        print(f"  {i}. Score: {result['combined_score']:.3f} - {result['document']}")
    
    print(f"\nTotal time: {results['total_time']:.4f}s")
    print(f"Avg query time: {results['avg_query_time']:.4f}s")


if __name__ == "__main__":
    main()