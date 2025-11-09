"""
Module 9: Evaluation Metrics
Implementation Examples

This module demonstrates evaluation metrics for RAG systems,
including retrieval metrics, generation metrics, and end-to-end evaluation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import string
import re
import time
import uuid
from scipy import stats
from collections import Counter
import statistics


class RetrievalEvaluator:
    """Evaluator for retrieval component of RAG systems"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def calculate_precision_at_k(self, retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """Calculate precision at k"""
        if k == 0:
            return 0.0
        
        retrieved_at_k = retrieved_docs[:k]
        if len(retrieved_at_k) == 0:
            return 0.0
        
        relevant_retrieved = set(retrieved_at_k) & set(relevant_docs)
        return len(relevant_retrieved) / len(retrieved_at_k)
    
    def calculate_recall_at_k(self, retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """Calculate recall at k"""
        if len(relevant_docs) == 0:
            return 1.0  # If no relevant docs exist, recall is 1 if nothing retrieved
        
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = set(retrieved_at_k) & set(relevant_docs)
        return len(relevant_retrieved) / len(relevant_docs)
    
    def calculate_f1_at_k(self, retrieved_docs: List[int], relevant_docs: List[int], k: int) -> float:
        """Calculate F1 score at k"""
        precision = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_mrr(self, retrieved_docs_list: List[List[int]], relevant_docs_list: List[List[int]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        rr_scores = []
        
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant:
                    rr_scores.append(1.0 / (i + 1))
                    break
            else:
                rr_scores.append(0.0)  # No relevant doc found
        
        return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0
    
    def calculate_ndcg(self, retrieved_docs: List[int], relevant_docs: List[int], 
                      relevance_scores: Dict[int, int], k: int = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if k is None:
            k = len(retrieved_docs)
        
        retrieved_at_k = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_at_k):
            rank = i + 1
            relevance = relevance_scores.get(doc_id, 0)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / np.log2(rank)
        
        # Calculate IDCG (ideal DCG)
        sorted_relevance = sorted([relevance_scores.get(doc_id, 0) for doc_id in relevant_docs], reverse=True)
        idcg = 0.0
        for i, rel in enumerate(sorted_relevance[:min(k, len(sorted_relevance))]):
            rank = i + 1
            if i == 0:
                idcg += rel
            else:
                idcg += rel / np.log2(rank)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_map(self, retrieved_docs_list: List[List[int]], relevant_docs_list: List[List[int]]) -> float:
        """Calculate Mean Average Precision"""
        average_precisions = []
        
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            if len(relevant) == 0:
                average_precisions.append(1.0 if len(retrieved) == 0 else 0.0)
                continue
            
            relevant_set = set(relevant)
            precisions_at_k = []
            
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant_set:
                    precision_at_k = len([d for d in retrieved[:i+1] if d in relevant_set]) / (i+1)
                    precisions_at_k.append(precision_at_k)
            
            if not precisions_at_k:
                average_precisions.append(0.0)
            else:
                average_precision = sum(precisions_at_k) / len(relevant)
                average_precisions.append(average_precision)
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    def evaluate_retrieval(self, retrieved_docs_list: List[List[int]], 
                          relevant_docs_list: List[List[int]], 
                          relevance_scores_list: List[Dict[int, int]] = None,
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Comprehensive retrieval evaluation"""
        if relevance_scores_list is None:
            relevance_scores_list = [{} for _ in relevant_docs_list]
        
        results = {
            'precision_at_k': {},
            'recall_at_k': {},
            'f1_at_k': {},
            'mrr': self.calculate_mrr(retrieved_docs_list, relevant_docs_list),
            'map': self.calculate_map(retrieved_docs_list, relevant_docs_list),
            'ndcg_at_k': {}
        }
        
        # Calculate metrics for each k value
        for k in k_values:
            precisions = []
            recalls = []
            f1_scores = []
            ndcg_scores = []
            
            for retrieved, relevant, relevance_scores in zip(retrieved_docs_list, relevant_docs_list, relevance_scores_list):
                precisions.append(self.calculate_precision_at_k(retrieved, relevant, k))
                recalls.append(self.calculate_recall_at_k(retrieved, relevant, k))
                f1_scores.append(self.calculate_f1_at_k(retrieved, relevant, k))
                ndcg_scores.append(self.calculate_ndcg(retrieved, relevant, relevance_scores, k))
            
            results['precision_at_k'][f'P@{k}'] = sum(precisions) / len(precisions) if precisions else 0.0
            results['recall_at_k'][f'R@{k}'] = sum(recalls) / len(recalls) if recalls else 0.0
            results['f1_at_k'][f'F1@{k}'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            results['ndcg_at_k'][f'nDCG@{k}'] = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        
        return results


class GenerationEvaluator:
    """Evaluator for generation component of RAG systems"""
    
    def __init__(self):
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            # Initialize ROUGE scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        except:
            # If NLTK downloads fail, use simplified versions
            self.rouge_scorer = None
    
    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if self.rouge_scorer is None:
            # Simplified implementation if rouge_scorer is not available
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_bleu(self, generated: str, reference: str) -> float:
        """Calculate BLEU score"""
        try:
            # Tokenize the sentences
            gen_tokens = word_tokenize(generated.lower())
            ref_tokens = word_tokenize(reference.lower())
            
            # Calculate BLEU score
            bleu_score = sentence_bleu([ref_tokens], gen_tokens)
            return bleu_score
        except:
            # Fallback if NLTK is not working
            return 0.0
    
    def calculate_exact_match(self, generated: str, reference: str) -> float:
        """Calculate exact match score"""
        # Normalize both strings
        gen_norm = self._normalize_text(generated)
        ref_norm = self._normalize_text(reference)
        return 1.0 if gen_norm == ref_norm else 0.0
    
    def calculate_f1_score(self, generated: str, reference: str) -> float:
        """Calculate token-level F1 score"""
        gen_tokens = set(self._normalize_text(generated).split())
        ref_tokens = set(self._normalize_text(reference).split())
        
        if not gen_tokens and not ref_tokens:
            return 1.0
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        common_tokens = gen_tokens.intersection(ref_tokens)
        precision = len(common_tokens) / len(gen_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_semantic_similarity(self, generated: str, reference: str, embedder) -> float:
        """Calculate semantic similarity using embeddings"""
        gen_embedding = embedder.encode([generated])
        ref_embedding = embedder.encode([reference])
        
        similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
        return float(similarity)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove punctuation and convert to lowercase
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def evaluate_generation(self, generated_list: List[str], 
                           reference_list: List[str], 
                           embedder) -> Dict[str, Any]:
        """Comprehensive generation evaluation"""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        exact_match_scores = []
        f1_scores = []
        semantic_similarities = []
        
        for gen, ref in zip(generated_list, reference_list):
            # ROUGE scores
            rouge_scores = self.calculate_rouge(gen, ref)
            rouge1_scores.append(rouge_scores['rouge1'])
            rouge2_scores.append(rouge_scores['rouge2'])
            rougeL_scores.append(rouge_scores['rougeL'])
            
            # BLEU score
            bleu_scores.append(self.calculate_bleu(gen, ref))
            
            # Exact match
            exact_match_scores.append(self.calculate_exact_match(gen, ref))
            
            # F1 score
            f1_scores.append(self.calculate_f1_score(gen, ref))
            
            # Semantic similarity
            semantic_similarities.append(self.calculate_semantic_similarity(gen, ref, embedder))
        
        return {
            'avg_rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            'avg_rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
            'avg_rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
            'avg_bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            'avg_exact_match': sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0,
            'avg_f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            'avg_semantic_similarity': sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0.0
        }


class RAGEvaluator:
    """Comprehensive evaluator for end-to-end RAG systems"""
    
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_rag_system(self, queries: List[str],
                           retrieved_docs_list: List[List[int]],
                           relevant_docs_list: List[List[int]],
                           generated_answers: List[str], 
                           reference_answers: List[str],
                           relevance_scores_list: List[Dict[int, int]] = None) -> Dict[str, Any]:
        """Evaluate complete RAG system with retrieval and generation metrics"""
        
        # Evaluate retrieval component
        retrieval_results = self.retrieval_evaluator.evaluate_retrieval(
            retrieved_docs_list, relevant_docs_list, relevance_scores_list
        )
        
        # Evaluate generation component
        generation_results = self.generation_evaluator.evaluate_generation(
            generated_answers, reference_answers, self.embedder
        )
        
        # Combine results
        results = {
            'retrieval_metrics': retrieval_results,
            'generation_metrics': generation_results,
            'end_to_end_metrics': {}
        }
        
        # Calculate overall end-to-end metrics
        # For example, combine retrieval and generation scores
        avg_retrieval_score = (retrieval_results['precision_at_k'].get('P@5', 0) + 
                              retrieval_results['recall_at_k'].get('R@5', 0)) / 2
        avg_generation_score = generation_results['avg_semantic_similarity']
        
        results['end_to_end_metrics']['retrieval_generation_balance'] = (
            0.5 * avg_retrieval_score + 0.5 * avg_generation_score
        )
        
        return results
    
    def calculate_efficiency_metrics(self, 
                                   query_times: List[float],
                                   token_counts: List[int] = None,
                                   processing_times: List[float] = None) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        results = {
            'avg_query_time': sum(query_times) / len(query_times) if query_times else 0.0,
            'std_query_time': statistics.stdev(query_times) if len(query_times) > 1 else 0.0,
            'max_query_time': max(query_times) if query_times else 0.0,
            'min_query_time': min(query_times) if query_times else 0.0,
            'p95_query_time': np.percentile(query_times, 95) if query_times else 0.0
        }
        
        if token_counts:
            results['avg_tokens_per_query'] = sum(token_counts) / len(token_counts)
            results['total_tokens'] = sum(token_counts)
        
        if processing_times:
            results['avg_processing_time'] = sum(processing_times) / len(processing_times)
        
        return results
    
    def calculate_cost_metrics(self,
                              api_calls: List[Dict[str, int]],  # List of call info with tokens
                              model_costs: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate cost metrics based on API usage"""
        if model_costs is None:
            # Default cost model (example values)
            model_costs = {
                'embedding': 0.0001,  # $0.0001 per 1k tokens for embeddings
                'generation': 0.0015,  # $0.0015 per 1k tokens for generation
                'retrieval': 0.00005  # $0.00005 per call for retrieval
            }
        
        total_cost = 0.0
        
        for call in api_calls:
            if 'model_type' in call and call['model_type'] in model_costs:
                cost_per_thousand = model_costs[call['model_type']]
                tokens = call.get('tokens', call.get('input_tokens', 0) + call.get('output_tokens', 0))
                call_cost = (tokens / 1000) * cost_per_thousand
                total_cost += call_cost
        
        return {
            'estimated_cost': total_cost,
            'total_api_calls': len(api_calls),
            'cost_per_query': total_cost / len(api_calls) if api_calls else 0.0
        }


class StatisticalSignificanceTester:
    """Class to perform statistical significance testing"""
    
    def __init__(self):
        pass
    
    def perform_ttest(self, sample1: List[float], sample2: List[float], 
                     alpha: float = 0.05) -> Dict[str, Any]:
        """Perform t-test between two samples"""
        if len(sample1) < 2 or len(sample2) < 2:
            return {
                'p_value': None,
                'significant': False,
                't_statistic': None,
                'message': 'Sample size too small for t-test'
            }
        
        t_stat, p_value = stats.ttest_ind(sample1, sample2)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'alpha': alpha,
            'message': 'Significant difference' if p_value < alpha else 'No significant difference'
        }
    
    def calculate_confidence_interval(self, sample: List[float], 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a sample"""
        if len(sample) < 2:
            return 0.0, 0.0
        
        mean = statistics.mean(sample)
        stdev = statistics.stdev(sample)
        n = len(sample)
        
        # Calculate t-value for confidence level
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        # Calculate margin of error
        margin_error = t_value * (stdev / (n ** 0.5))
        
        return mean - margin_error, mean + margin_error


def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics in action"""
    print("=== Evaluation Metrics Demonstration ===\n")
    
    # Sample data for evaluation
    queries = ["What is machine learning?", "Explain neural networks", "How does NLP work?"]
    
    # Simulated retrieved documents (represented as IDs)
    retrieved_docs = [
        [1, 3, 5, 7, 9],      # Retrieved for query 1
        [2, 4, 6, 8, 10],     # Retrieved for query 2
        [1, 2, 5, 8, 11]      # Retrieved for query 3
    ]
    
    # Simulated relevant documents (represented as IDs)
    relevant_docs = [
        [1, 5, 9],            # Relevant for query 1
        [2, 6, 10],           # Relevant for query 2
        [1, 2, 8]             # Relevant for query 3
    ]
    
    # Simulated relevance scores (for nDCG)
    relevance_scores = [
        {1: 3, 3: 1, 5: 3, 7: 0, 9: 2},    # Scores for query 1
        {2: 2, 4: 0, 6: 3, 8: 1, 10: 2},   # Scores for query 2
        {1: 1, 2: 2, 5: 0, 8: 3, 11: 1}    # Scores for query 3
    ]
    
    # Simulated generated and reference answers
    generated_answers = [
        "Machine learning is a type of AI that enables computers to learn",
        "Neural networks are computing systems inspired by the human brain",
        "NLP helps computers understand and process human language"
    ]
    
    reference_answers = [
        "Machine learning is a method of teaching computers to recognize patterns in data",
        "Neural networks are a series of algorithms that mimic the operations of a human brain",
        "Natural language processing enables computers to understand, interpret and manipulate human language"
    ]
    
    # Evaluate the system
    evaluator = RAGEvaluator()
    
    print("Evaluating RAG system...")
    results = evaluator.evaluate_rag_system(
        queries=queries,
        retrieved_docs_list=retrieved_docs,
        relevant_docs_list=relevant_docs,
        generated_answers=generated_answers,
        reference_answers=reference_answers,
        relevance_scores_list=relevance_scores
    )
    
    # Print retrieval metrics
    retrieval_metrics = results['retrieval_metrics']
    print("\nRetrieval Metrics:")
    print(f"  MRR: {retrieval_metrics['mrr']:.3f}")
    print(f"  MAP: {retrieval_metrics['map']:.3f}")
    print(f"  P@5: {retrieval_metrics['precision_at_k'].get('P@5', 0):.3f}")
    print(f"  R@5: {retrieval_metrics['recall_at_k'].get('R@5', 0):.3f}")
    
    # Print generation metrics
    generation_metrics = results['generation_metrics']
    print(f"\nGeneration Metrics:")
    print(f"  Avg ROUGE-1: {generation_metrics['avg_rouge1']:.3f}")
    print(f"  Avg BLEU: {generation_metrics['avg_bleu']:.3f}")
    print(f"  Avg Semantic Similarity: {generation_metrics['avg_semantic_similarity']:.3f}")
    
    # Calculate efficiency metrics
    query_times = [0.25, 0.32, 0.28]  # Example query times in seconds
    efficiency_metrics = evaluator.calculate_efficiency_metrics(query_times)
    
    print(f"\nEfficiency Metrics:")
    print(f"  Avg Query Time: {efficiency_metrics['avg_query_time']:.3f}s")
    print(f"  P95 Query Time: {efficiency_metrics['p95_query_time']:.3f}s")
    
    # Calculate cost metrics
    api_calls = [
        {'model_type': 'embedding', 'tokens': 500},
        {'model_type': 'generation', 'tokens': 1000},
        {'model_type': 'retrieval', 'tokens': 0}
    ]
    cost_metrics = evaluator.calculate_cost_metrics(api_calls)
    
    print(f"\nCost Metrics:")
    print(f"  Estimated Total Cost: ${cost_metrics['estimated_cost']:.4f}")
    print(f"  Avg Cost per Query: ${cost_metrics['cost_per_query']:.4f}")


def performance_comparison():
    """Compare performance of different evaluation approaches"""
    print("\n=== Performance Comparison ===\n")
    
    # Create mock data for comparison
    large_retrieved = [[i for i in range(100)]] * 50  # 50 queries, 100 docs each
    large_relevant = [[i*2 for i in range(10)] for _ in range(50)]  # 10 relevant docs per query
    
    evaluator = RetrievalEvaluator()
    
    # Time the evaluation
    start_time = time.time()
    large_results = evaluator.evaluate_retrieval(large_retrieved, large_relevant)
    elapsed_time = time.time() - start_time
    
    print(f"Evaluated 50 queries with 100 retrieved docs each in {elapsed_time:.3f} seconds")
    print(f"Metrics calculated: {len(large_results)} top-level metrics")


def statistical_significance_example():
    """Demonstrate statistical significance testing"""
    print("\n=== Statistical Significance Example ===\n")
    
    # Simulated performance scores for two RAG systems
    system_a_scores = [0.72, 0.75, 0.68, 0.74, 0.71, 0.76, 0.69, 0.73, 0.70, 0.75]
    system_b_scores = [0.78, 0.79, 0.76, 0.80, 0.77, 0.81, 0.75, 0.78, 0.79, 0.80]
    
    tester = StatisticalSignificanceTester()
    
    # Perform t-test
    ttest_result = tester.perform_ttest(system_a_scores, system_b_scores)
    
    print(f"System A Average Score: {statistics.mean(system_a_scores):.3f}")
    print(f"System B Average Score: {statistics.mean(system_b_scores):.3f}")
    print(f"p-value: {ttest_result['p_value']:.3f}")
    print(f"Significant difference: {ttest_result['significant']}")
    
    # Calculate confidence intervals
    ci_a = tester.calculate_confidence_interval(system_a_scores)
    ci_b = tester.calculate_confidence_interval(system_b_scores)
    
    print(f"System A 95% CI: [{ci_a[0]:.3f}, {ci_a[1]:.3f}]")
    print(f"System B 95% CI: [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")


def main():
    """Main function to demonstrate evaluation metrics implementations"""
    print("Module 9: Evaluation Metrics")
    print("=" * 50)
    
    # Demonstrate evaluation metrics
    demonstrate_evaluation_metrics()
    
    # Show performance comparison
    performance_comparison()
    
    # Show statistical significance example
    statistical_significance_example()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of detailed retrieval evaluation
    print("Detailed Retrieval Evaluation:")
    retrieval_eval = RetrievalEvaluator()
    
    # Example with single query
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 3, 5]
    
    p_at_3 = retrieval_eval.calculate_precision_at_k(retrieved, relevant, 3)
    r_at_3 = retrieval_eval.calculate_recall_at_k(retrieved, relevant, 3)
    f1_at_3 = retrieval_eval.calculate_f1_at_k(retrieved, relevant, 3)
    
    print(f"  P@3: {p_at_3:.3f}")
    print(f"  R@3: {r_at_3:.3f}")
    print(f"  F1@3: {f1_at_3:.3f}")
    
    print(f"\nModule 9 completed - Evaluation metrics implemented and demonstrated")


if __name__ == "__main__":
    main()