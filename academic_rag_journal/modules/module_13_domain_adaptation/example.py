"""
Module 13: Domain Adaptation
Implementation Examples

This module demonstrates domain adaptation techniques for RAG systems,
including domain-specific embeddings, retrieval, and generation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import chromadb
from chromadb.config import Settings
import time
import uuid
import json
import re
from datetime import datetime
from dataclasses import dataclass
import pickle
import hashlib
import requests
from sklearn.metrics.pairwise import cosine_similarity
import torch


class DomainVocabulary:
    """Class to manage domain-specific vocabulary and terminology"""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.terms = {}
        self.abbreviations = {}
        self.expansions = {}
        self.domain_expertise_level = 0  # 0=general, 1=intermediate, 2=expert
    
    def add_term(self, term: str, definition: str, category: str = "general"):
        """Add a domain-specific term with its definition"""
        self.terms[term.lower()] = {
            'definition': definition,
            'category': category,
            'added_at': datetime.now()
        }
    
    def add_abbreviation(self, abbr: str, full_form: str):
        """Add an abbreviation and its full form"""
        self.abbreviations[abbr.lower()] = full_form
        self.expansions[full_form.lower()] = abbr.upper()
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations in text"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word).lower()
            if clean_word in self.abbreviations:
                expanded_words.append(self.abbreviations[clean_word])
            else:
                expanded_words.append(word)
        
        return " ".join(expanded_words)
    
    def get_related_terms(self, term: str) -> List[str]:
        """Get terms related to the given term"""
        # For this example, return terms from the same category
        target_term = term.lower()
        if target_term in self.terms:
            category = self.terms[target_term]['category']
            related = [t for t, data in self.terms.items() if data['category'] == category and t != target_term]
            return related
        return []


class DomainEmbeddingAdapter:
    """Adapter for domain-specific embedding customization"""
    
    def __init__(self, base_model_name: str = 'all-MiniLM-L6-v2'):
        self.base_model_name = base_model_name
        self.base_embedder = SentenceTransformer(base_model_name)
        self.domain_embeddings = {}  # Domain-specific embeddings
        self.domain_weights = {}  # Weight adjustments for domain terms
    
    def adapt_for_domain(self, domain_vocab: DomainVocabulary, 
                        domain_texts: List[str] = None) -> 'DomainEmbeddingAdapter':
        """Adapt embeddings for a specific domain"""
        # Create domain-specific embeddings for important terms
        if domain_vocab.terms:
            domain_terms = list(domain_vocab.terms.keys())
            domain_embeddings = self.base_embedder.encode(domain_terms)
            
            for term, embedding in zip(domain_terms, domain_embeddings):
                self.domain_embeddings[term] = embedding
        
        # Calculate domain weights based on term importance
        if domain_texts:
            self._calculate_domain_weights(domain_texts)
        
        return self
    
    def _calculate_domain_weights(self, domain_texts: List[str]):
        """Calculate weights for domain-specific terms"""
        # Simple frequency-based approach
        all_text = " ".join(domain_texts).lower()
        for term in self.domain_embeddings:
            if term in all_text:
                # Weight based on frequency
                frequency = all_text.count(term)
                self.domain_weights[term] = 1.0 + (frequency * 0.1)
    
    def encode_with_domain_knowledge(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts with domain-specific knowledge"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Process text with domain knowledge
            processed_text = self._apply_domain_processing(text)
            embedding = self.base_embedder.encode([processed_text])[0]
            
            # Adjust embedding based on domain weights
            weighted_embedding = self._apply_domain_weights(embedding, text)
            embeddings.append(weighted_embedding)
        
        return np.array(embeddings)
    
    def _apply_domain_processing(self, text: str) -> str:
        """Apply domain-specific processing to text"""
        # This is a simplified implementation
        # In practice, this would involve more sophisticated NLP
        return text
    
    def _apply_domain_weights(self, embedding: np.ndarray, text: str) -> np.ndarray:
        """Apply domain weights to embedding"""
        # Simple approach: boost embedding if domain terms are present
        text_lower = text.lower()
        boost_factor = 1.0
        
        for term, weight in self.domain_weights.items():
            if term in text_lower:
                boost_factor *= weight
        
        return embedding * boost_factor


class DomainSpecificRetriever:
    """Retriever with domain-specific optimizations"""
    
    def __init__(self, domain_adapter: DomainEmbeddingAdapter, 
                 domain_vocab: DomainVocabulary,
                 collection_name: str = "domain_specific_rag"):
        self.domain_adapter = domain_adapter
        self.domain_vocab = domain_vocab
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None, doc_id: str = None) -> str:
        """Add a document with domain-specific processing"""
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}
        metadata['id'] = doc_id
        
        # Process text with domain knowledge
        processed_text = self._preprocess_for_domain(text)
        
        # Generate domain-aware embedding
        embedding = self.domain_adapter.encode_with_domain_knowledge([processed_text])
        
        # Add to collection
        self.collection.add(
            embeddings=embedding.tolist(),
            documents=[text],  # Store original text
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def _preprocess_for_domain(self, text: str) -> str:
        """Preprocess text using domain knowledge"""
        # Expand abbreviations
        expanded_text = self.domain_vocab.expand_abbreviations(text)
        
        # Add related terms (simplified approach)
        words = expanded_text.split()
        for term in self.domain_vocab.terms:
            if term in expanded_text.lower():
                related = self.domain_vocab.get_related_terms(term)
                # Add related terms to enhance retrieval
                for related_term in related[:2]:  # Limit to 2 related terms
                    if related_term not in expanded_text.lower():
                        expanded_text += f" {related_term}"
        
        return expanded_text
    
    def retrieve_with_domain_knowledge(self, query: str, top_k: int = 5,
                                     domain_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve documents with domain-specific enhancements"""
        # Process query with domain knowledge
        processed_query = self._preprocess_for_domain(query)
        
        # Generate domain-aware embedding for query
        query_embedding = self.domain_adapter.encode_with_domain_knowledge([processed_query])
        
        # Perform retrieval
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=domain_filters,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1.0 - results['distances'][0][i],
                'domain_relevance': self._calculate_domain_relevance(
                    query, results['documents'][0][i]
                )
            })
        
        # Sort by combined domain relevance and similarity
        formatted_results.sort(
            key=lambda x: x['domain_relevance'] * 0.3 + x['similarity'] * 0.7,
            reverse=True
        )
        
        return formatted_results
    
    def _calculate_domain_relevance(self, query: str, document: str) -> float:
        """Calculate domain-specific relevance score"""
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Count matches of domain-specific terms
        domain_term_matches = 0
        total_domain_terms = len(self.domain_vocab.terms)
        
        for term in self.domain_vocab.terms:
            if term in query_lower or term in doc_lower:
                domain_term_matches += 1
        
        if total_domain_terms == 0:
            return 0.5  # Neutral score if no domain terms
        
        return min(domain_term_matches / total_domain_terms * 2, 1.0)  # Cap at 1.0


class DomainAwareGenerator:
    """Generator with domain-aware capabilities"""
    
    def __init__(self, domain_vocab: DomainVocabulary):
        self.domain_vocab = domain_vocab
        try:
            # Use a smaller model for local generation
            self.generator = pipeline(
                "text-generation", 
                model="gpt2",  # Using GPT-2 as a lighter alternative
                tokenizer="gpt2",
                pad_token_id=50256  # GPT-2 pad token
            )
        except:
            # Fallback to simple template-based generation
            self.generator = None
    
    def generate_with_domain_knowledge(self, query: str, context: str = "", 
                                     max_length: int = 100) -> str:
        """Generate response with domain knowledge integration"""
        if self.generator is None:
            # Fallback to template-based response
            return self._template_based_generation(query, context)
        
        # Enhance query with domain context
        enhanced_prompt = self._enhance_prompt_with_domain(query, context)
        
        try:
            result = self.generator(
                enhanced_prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            # Extract the generated text (remove the prompt part)
            generated = result[0]['generated_text'][len(enhanced_prompt):]
            
            # Post-process to ensure domain consistency
            return self._post_process_with_domain(generated)
        
        except Exception as e:
            # Fallback if generation fails
            return f"Could not generate response: {str(e)}"
    
    def _enhance_prompt_with_domain(self, query: str, context: str) -> str:
        """Enhance prompt with domain-specific context"""
        # Add domain-specific instructions
        domain_instruction = f"Please provide an answer using {self.domain_vocab.domain_name} terminology and concepts. "
        
        # Extract relevant domain terms from query
        relevant_terms = []
        query_lower = query.lower()
        for term in self.domain_vocab.terms:
            if term in query_lower:
                relevant_terms.append(f"{term}: {self.domain_vocab.terms[term]['definition']}")
        
        if relevant_terms:
            domain_context = "Domain terms and definitions: " + "; ".join(relevant_terms) + ". "
        else:
            domain_context = ""
        
        enhanced_prompt = f"{domain_instruction}{domain_context}\nContext: {context}\nQuery: {query}\nAnswer:"
        return enhanced_prompt
    
    def _post_process_with_domain(self, generated_text: str) -> str:
        """Post-process generated text to ensure domain consistency"""
        # Expand any abbreviations that might have been used
        result = self.domain_vocab.expand_abbreviations(generated_text)
        
        # Clean up the text
        result = result.strip()
        
        return result
    
    def _template_based_generation(self, query: str, context: str) -> str:
        """Template-based generation as fallback"""
        return f"Based on the context and within the domain of {self.domain_vocab.domain_name}, here is the response to your query about '{query[:50]}...' (context: {context[:100]}...)."


class DomainAdaptedRAG:
    """Domain-adapted RAG system"""
    
    def __init__(self, domain_name: str, 
                 base_model: str = 'all-MiniLM-L6-v2'):
        self.domain_name = domain_name
        self.domain_vocab = DomainVocabulary(domain_name)
        
        # Initialize domain adapter
        self.domain_adapter = DomainEmbeddingAdapter(base_model)
        
        # Initialize components
        self.retriever = DomainSpecificRetriever(
            self.domain_adapter, 
            self.domain_vocab,
            collection_name=f"domain_{domain_name.replace(' ', '_')}_rag"
        )
        self.generator = DomainAwareGenerator(self.domain_vocab)
        
        # Performance metrics
        self.metrics = {
            'domain_queries': 0,
            'avg_retrieval_time': 0.0,
            'avg_generation_time': 0.0
        }
    
    def add_domain_knowledge(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add domain-specific knowledge to the system"""
        if metadata is None:
            metadata = [{} for _ in texts]
        
        doc_ids = []
        for text, meta in zip(texts, metadata):
            doc_id = self.retriever.add_document(text, meta)
            doc_ids.append(doc_id)
        
        # Adapt embeddings to the domain
        self.domain_adapter.adapt_for_domain(self.domain_vocab, texts)
        
        return doc_ids
    
    def query(self, user_query: str, top_k: int = 5, 
              include_generation: bool = True) -> Dict[str, Any]:
        """Process a query using the domain-adapted system"""
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_start = time.time()
        results = self.retriever.retrieve_with_domain_knowledge(user_query, top_k)
        retrieval_time = time.time() - retrieval_start
        
        response = {
            'query': user_query,
            'retrieval_results': results,
            'retrieval_time': retrieval_time
        }
        
        # Generate response if requested
        if include_generation and results:
            generation_start = time.time()
            context = " ".join([r['content'] for r in results[:3]])  # Use top 3 results
            generated_response = self.generator.generate_with_domain_knowledge(
                user_query, context
            )
            generation_time = time.time() - generation_start
            
            response['generated_response'] = generated_response
            response['generation_time'] = generation_time
        
        total_time = time.time() - start_time
        response['total_time'] = total_time
        
        # Update metrics
        self.metrics['domain_queries'] += 1
        avg_retrieval = self.metrics['avg_retrieval_time']
        avg_generation = self.metrics['avg_generation_time']
        
        self.metrics['avg_retrieval_time'] = (
            (avg_retrieval * (self.metrics['domain_queries'] - 1) + retrieval_time) / 
            self.metrics['domain_queries']
        )
        self.metrics['avg_generation_time'] = (
            (avg_generation * (self.metrics['domain_queries'] - 1) + 
             response.get('generation_time', 0)) / 
            self.metrics['domain_queries']
        )
        
        return response
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the domain-adapted system"""
        return self.metrics.copy()
    
    def add_domain_term(self, term: str, definition: str, category: str = "general"):
        """Add a domain-specific term"""
        self.domain_vocab.add_term(term, definition, category)
    
    def add_domain_abbreviation(self, abbr: str, full_form: str):
        """Add a domain-specific abbreviation"""
        self.domain_vocab.add_abbreviation(abbr, full_form)


class DomainEvaluator:
    """Evaluator for domain-adapted RAG systems"""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.domain_queries = []
        self.domain_responses = []
    
    def evaluate_domain_relevance(self, query: str, response: str, 
                                expected_domain_terms: List[str] = None) -> float:
        """Evaluate how well the response addresses domain-specific concepts"""
        if expected_domain_terms is None:
            expected_domain_terms = []
        
        response_lower = response.lower()
        
        # Count occurrences of expected domain terms
        matches = sum(1 for term in expected_domain_terms if term.lower() in response_lower)
        
        if expected_domain_terms:
            relevance_score = matches / len(expected_domain_terms)
        else:
            # If no expected terms provided, use a simple heuristic
            domain_keywords = ['method', 'approach', 'technique', 'process', 'framework', 'model']
            matches = sum(1 for keyword in domain_keywords if keyword in response_lower)
            relevance_score = min(matches * 0.2, 1.0)
        
        return relevance_score
    
    def evaluate_terminology_accuracy(self, response: str, domain_vocab: DomainVocabulary) -> float:
        """Evaluate the accuracy of domain terminology usage"""
        response_lower = response.lower()
        correct_terms = 0
        total_terms = len(domain_vocab.terms)
        
        for term in domain_vocab.terms:
            if term in response_lower:
                correct_terms += 1
        
        if total_terms == 0:
            return 0.5  # Neutral score if no domain terms defined
        
        return correct_terms / total_terms
    
    def run_domain_evaluation(self, rag_system: DomainAdaptedRAG, 
                             test_queries: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
        """Run comprehensive domain evaluation"""
        results = {
            'average_domain_relevance': 0.0,
            'average_terminology_accuracy': 0.0,
            'total_queries': len(test_queries),
            'detailed_results': []
        }
        
        relevance_scores = []
        terminology_scores = []
        
        for query, expected_terms in test_queries:
            response = rag_system.query(query, include_generation=True)
            
            # Evaluate domain relevance
            domain_relevance = self.evaluate_domain_relevance(
                query, response.get('generated_response', ''), expected_terms
            )
            relevance_scores.append(domain_relevance)
            
            # Evaluate terminology accuracy
            term_accuracy = self.evaluate_terminology_accuracy(
                response.get('generated_response', ''), rag_system.domain_vocab
            )
            terminology_scores.append(term_accuracy)
            
            results['detailed_results'].append({
                'query': query,
                'domain_relevance': domain_relevance,
                'terminology_accuracy': term_accuracy,
                'response_length': len(response.get('generated_response', ''))
            })
        
        results['average_domain_relevance'] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        results['average_terminology_accuracy'] = sum(terminology_scores) / len(terminology_scores) if terminology_scores else 0
        
        return results


def demonstrate_domain_adaptation():
    """Demonstrate domain adaptation techniques"""
    print("=== Domain Adaptation Demonstration ===\n")
    
    # Create a domain-adapted RAG system for medical domain
    medical_rag = DomainAdaptedRAG(domain_name="Medical")
    
    # Add domain-specific terminology
    medical_terms = [
        ("myocardial infarction", "Commonly known as a heart attack, occurs when blood flow to part of the heart is blocked"),
        ("hypertension", "High blood pressure, a condition where the force of blood against artery walls is too high"),
        ("diabetes mellitus", "A group of diseases that affect how your body uses blood sugar (glucose)"),
        ("MRI", "Magnetic Resonance Imaging, a medical imaging technique"),
        ("CT scan", "Computed Tomography scan, using X-rays to create detailed images")
    ]
    
    for term, definition in medical_terms:
        medical_rag.add_domain_term(term, definition)
        if "(" in definition and ")" in definition:
            # Extract abbreviation from definition
            import re
            abbrev_match = re.search(r'\(([^)]+)\)', definition)
            if abbrev_match:
                abbrev = abbrev_match.group(1)
                medical_rag.add_domain_abbreviation(abbrev, term)
    
    # Add sample medical documents
    medical_docs = [
        "Myocardial infarction occurs when blood flow to the heart is severely reduced or blocked. The blockage is usually due to a buildup of plaque.",
        "Hypertension is often called the silent killer because it typically has no symptoms. It can lead to serious health problems if left untreated.",
        "Diabetes mellitus affects how your body regulates blood sugar. There are two main types: Type 1 and Type 2 diabetes.",
        "MRI uses powerful magnets and radio waves to create detailed images. It's particularly useful for imaging soft tissues.",
        "CT scans combine X-ray images taken from different angles to create cross-sectional images of bones, blood vessels, and soft tissues."
    ]
    
    print(f"Adding {len(medical_docs)} medical documents to the system...")
    medical_rag.add_domain_knowledge(medical_docs)
    
    # Test domain-specific queries
    test_queries = [
        "Explain heart attack",
        "What is high blood pressure?",
        "Describe MRI imaging",
        "How does diabetes affect the body?"
    ]
    
    print("\nTesting domain-adapted retrieval and generation:\n")
    for query in test_queries:
        print(f"Query: '{query}'")
        
        response = medical_rag.query(query, top_k=3, include_generation=True)
        
        print(f"Retrieval time: {response['retrieval_time']:.4f}s")
        if 'generated_response' in response:
            print(f"Generated response: {response['generated_response'][:200]}...")
            print(f"Generation time: {response['generation_time']:.4f}s")
        print()
    
    # Show performance metrics
    metrics = medical_rag.get_performance_metrics()
    print("Performance Metrics:")
    print(f"  Domain queries processed: {metrics['domain_queries']}")
    print(f"  Average retrieval time: {metrics['avg_retrieval_time']:.4f}s")
    print(f"  Average generation time: {metrics['avg_generation_time']:.4f}s")


def performance_comparison():
    """Compare performance of domain-adapted vs general system"""
    print("\n=== Performance Comparison ===\n")
    
    # This would normally compare a domain-adapted system with a general one
    # For this example, we'll just show the domain adaptation process
    
    # Create domain vocabularies
    tech_vocab = DomainVocabulary("Technology")
    tech_vocab.add_term("API", "Application Programming Interface, a set of rules for building software applications")
    tech_vocab.add_term("machine learning", "A type of artificial intelligence that enables computers to learn from data")
    tech_vocab.add_term("cloud computing", "The delivery of computing services over the internet")
    
    print("Domain vocabulary created with terminology and definitions")
    print(f"Technology domain has {len(tech_vocab.terms)} terms")
    
    # Show how domain terms can be used for expansion
    sample_query = "API and machine learning"
    expanded = tech_vocab.expand_abbreviations(sample_query)
    print(f"Original query: '{sample_query}'")
    print(f"Expanded (if applicable): '{expanded}'")


def main():
    """Main function to demonstrate domain adaptation implementations"""
    print("Module 13: Domain Adaptation")
    print("=" * 50)
    
    # Demonstrate domain adaptation
    demonstrate_domain_adaptation()
    
    # Show performance comparison
    performance_comparison()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of domain evaluation
    print("Domain Evaluation Example:")
    
    legal_rag = DomainAdaptedRAG(domain_name="Legal")
    
    # Add legal terms
    legal_terms = [
        ("contract", "A legally binding agreement between parties"),
        ("liability", "Legal responsibility for one's acts or omissions"),
        ("jurisdiction", "The authority of a court to hear and decide cases")
    ]
    
    for term, definition in legal_terms:
        legal_rag.add_domain_term(term, definition)
    
    # Add legal documents
    legal_docs = [
        "A contract is a legally binding agreement that creates obligations between parties.",
        "Liability refers to legal responsibility that can arise from actions or omissions.",
        "Jurisdiction determines which court has the authority to hear a particular case."
    ]
    
    legal_rag.add_domain_knowledge(legal_docs)
    
    # Run domain evaluation
    evaluator = DomainEvaluator("Legal")
    test_queries = [
        ("What is a contract?", ["contract", "agreement", "legally binding"]),
        ("Explain liability", ["liability", "responsibility", "legal"])
    ]
    
    eval_results = evaluator.run_domain_evaluation(legal_rag, test_queries)
    print(f"Evaluation Results:")
    print(f"  Average domain relevance: {eval_results['average_domain_relevance']:.2f}")
    print(f"  Average terminology accuracy: {eval_results['average_terminology_accuracy']:.2f}")
    print(f"  Total queries evaluated: {eval_results['total_queries']}")
    
    print(f"\nModule 13 completed - Domain adaptation techniques implemented and demonstrated")


if __name__ == "__main__":
    main()