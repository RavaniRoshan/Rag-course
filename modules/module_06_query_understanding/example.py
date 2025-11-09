"""
Module 6: Query Understanding
Implementation Examples

This module demonstrates query understanding techniques including preprocessing,
intent classification, query expansion, and reformulation for RAG systems.
"""

import re
import string
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import spacy
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid


class QueryPreprocessor:
    """Class for query preprocessing and normalization"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load("en_core_web_sm") if spacy.util.is_package("en_core_web_sm") else None
    
    def preprocess(self, query: str) -> Dict[str, Any]:
        """Complete preprocessing pipeline for a query"""
        original_query = query
        
        # Step 1: Clean and normalize
        query = self.clean_text(query)
        
        # Step 2: Tokenize
        tokens = self.tokenize(query)
        
        # Step 3: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 4: Lemmatize
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        
        # Step 5: Extract named entities if spaCy is available
        entities = self.extract_entities(original_query) if self.nlp else []
        
        return {
            'original_query': original_query,
            'cleaned_query': query,
            'tokens': tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'entities': entities
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using spaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities


class IntentClassifier:
    """Class for classifying query intent"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        # Use a pre-trained model for sequence classification
        # For demonstration, we'll use a simple approach with predefined patterns
        # In practice, you'd train a model on your specific intent dataset
        self.intent_patterns = {
            'informational': [
                r'what is', r'what are', r'tell me about', r'explain', 
                r'define', r'meaning of', r'describe', r'how does', r'why is'
            ],
            'navigational': [
                r'go to', r'find', r'show me', r'where is', 
                r'locate', r'find the', r'looking for'
            ],
            'comparative': [
                r'compare', r'vs', r'versus', r'difference between',
                r'better than', r'pros and cons', r'similarities'
            ],
            'instructional': [
                r'how to', r'guide', r'tutorial', r'steps to',
                r'process', r'procedure', r'way to'
            ]
        }
        
        # Initialize a transformer pipeline for more sophisticated classification
        try:
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",  # Placeholder - in practice use a proper classification model
                return_all_scores=True
            )
            self.use_transformer = True
        except:
            self.use_transformer = False
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify the intent of a query"""
        query_lower = query.lower()
        
        # Simple pattern matching approach
        detected_intents = []
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents.append({
                        'intent': intent,
                        'confidence': 1.0,  # Placeholder confidence
                        'pattern': pattern
                    })
                    break  # Don't match the same intent multiple times
        
        # If no patterns matched, use default
        if not detected_intents:
            detected_intents.append({
                'intent': 'informational',
                'confidence': 0.5,  # Lower confidence for default
                'pattern': 'default'
            })
        
        # For now, return the first detected intent
        return detected_intents[0] if detected_intents else {
            'intent': 'unknown',
            'confidence': 0.0,
            'pattern': 'none'
        }


class QueryExpander:
    """Class for query expansion techniques"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.synonyms_cache = {}
    
    def expand_with_synonyms(self, query: str, max_expansions: int = 5) -> List[str]:
        """Expand query with synonyms using WordNet"""
        tokens = word_tokenize(query.lower())
        expanded_queries = []
        
        for token in tokens:
            synonyms = self._get_synonyms(token)
            if synonyms:
                for synonym in synonyms[:max_expansions]:
                    new_query = query.replace(token, f"{token} {synonym}")
                    expanded_queries.append(new_query)
        
        return expanded_queries
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet"""
        if word in self.synonyms_cache:
            return self.synonyms_cache[word]
        
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        synonyms_list = list(synonyms)[:10]  # Limit to 10 synonyms
        self.synonyms_cache[word] = synonyms_list
        return synonyms_list
    
    def expand_with_embeddings(self, query: str, candidate_terms: List[str], top_k: int = 5) -> List[str]:
        """Expand query with semantically similar terms using embeddings"""
        # Encode the original query
        query_embedding = self.embedder.encode([query])
        
        # Encode candidate terms
        candidate_embeddings = self.embedder.encode(candidate_terms)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top-k similar terms
        top_indices = similarities.argsort()[-top_k:][::-1]
        similar_terms = [candidate_terms[i] for i in top_indices if similarities[i] > 0.3]  # Threshold
        
        # Create expanded query
        expanded_query = query + " " + " ".join(similar_terms)
        
        return [expanded_query]
    
    def expand_with_entities(self, query: str, entities: List[Dict]) -> str:
        """Expand query with related entities"""
        if not entities:
            return query
        
        # For this example, we'll just append entity types to the query
        # In practice, you'd use knowledge bases to find related entities
        entity_types = [entity['label'] for entity in entities]
        entity_types_string = " ".join(set(entity_types))  # Remove duplicates
        
        return f"{query} {entity_types_string}"


class QueryReformulator:
    """Class for query reformulation"""
    
    def __init__(self):
        self.spell_checker = self._initialize_spell_checker()
    
    def _initialize_spell_checker(self):
        """Initialize spell checker (using a simple approach)"""
        try:
            from textblob import TextBlob
            return TextBlob
        except ImportError:
            return None
    
    def correct_spelling(self, query: str) -> str:
        """Correct spelling errors in the query"""
        if not self.spell_checker:
            return query  # Return original if no spell checker available
        
        try:
            blob = self.spell_checker(query)
            corrected_query = blob.correct()
            return str(corrected_query)
        except:
            return query  # Return original if correction fails
    
    def rewrite_query(self, query: str, intent: str) -> str:
        """Rewrite query based on detected intent"""
        if intent == 'informational':
            # Add common informational phrases
            if not any(phrase in query.lower() for phrase in ['what is', 'what are', 'how', 'why', 'explain']):
                # This is a simple rule, in practice you'd have more sophisticated rewrite rules
                pass
        elif intent == 'instructional':
            # Ensure query has instructional phrasing
            if not any(phrase in query.lower() for phrase in ['how to', 'steps to', 'guide', 'tutorial']):
                pass
        
        return query
    
    def simplify_query(self, query: str) -> str:
        """Simplify complex queries"""
        # Remove redundant phrases
        simplified = re.sub(r'\b(please|kindly|could you)\b', '', query, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        simplified = re.sub(r'[!?.]{2,}', '.', simplified)
        
        # Clean up extra spaces
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        
        return simplified


class QueryUnderstandingPipeline:
    """Complete pipeline for query understanding"""
    
    def __init__(self):
        self.preprocessor = QueryPreprocessor()
        self.intent_classifier = IntentClassifier()
        self.expander = QueryExpander()
        self.reformulator = QueryReformulator()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the entire pipeline"""
        # Step 1: Preprocess
        preprocessed = self.preprocessor.preprocess(query)
        
        # Step 2: Intent Classification
        intent = self.intent_classifier.classify_intent(query)
        
        # Step 3: Query Reformulation
        reformulated_query = self.reformulator.rewrite_query(query, intent['intent'])
        spell_corrected = self.reformulator.correct_spelling(reformulated_query)
        simplified = self.reformulator.simplify_query(spell_corrected)
        
        # Step 4: Query Expansion
        synonym_expansions = self.expander.expand_with_synonyms(simplified)
        entity_expanded = self.expander.expand_with_entities(simplified, preprocessed['entities'])
        
        # Combine results
        result = {
            'original_query': query,
            'preprocessed': preprocessed,
            'intent': intent,
            'reformulated_query': simplified,
            'synonym_expansions': synonym_expansions,
            'entity_expanded_query': entity_expanded,
            'all_expansions': [simplified] + synonym_expansions + [entity_expanded]
        }
        
        return result


class QueryUnderstandingEvaluator:
    """Evaluator for query understanding components"""
    
    def __init__(self):
        self.sample_queries = [
            "How does machine learning work?",
            "What is the capital of France?",
            "Compare neural networks vs decision trees",
            "Find documentation for Python requests library",
            "Tutorial on building a chatbot",
            "Explain quantum computing"
        ]
        
        self.pipeline = QueryUnderstandingPipeline()
    
    def evaluate_preprocessing(self) -> Dict[str, Any]:
        """Evaluate preprocessing components"""
        results = []
        for query in self.sample_queries:
            preprocessed = self.pipeline.preprocessor.preprocess(query)
            results.append({
                'original': query,
                'processed': preprocessed['cleaned_query'],
                'tokens': preprocessed['lemmatized_tokens'],
                'entities': preprocessed['entities']
            })
        
        return {
            'method': 'preprocessing',
            'total_queries': len(self.sample_queries),
            'results': results
        }
    
    def evaluate_intent_classification(self) -> Dict[str, Any]:
        """Evaluate intent classification"""
        intent_counts = {}
        for query in self.sample_queries:
            intent = self.pipeline.intent_classifier.classify_intent(query)
            intent_name = intent['intent']
            if intent_name in intent_counts:
                intent_counts[intent_name] += 1
            else:
                intent_counts[intent_name] = 1
        
        return {
            'method': 'intent_classification',
            'intent_distribution': intent_counts,
            'total_queries': len(self.sample_queries)
        }
    
    def evaluate_query_expansion(self) -> Dict[str, Any]:
        """Evaluate query expansion"""
        expansion_stats = []
        for query in self.sample_queries:
            processed = self.pipeline.process_query(query)
            original_len = len(query.split())
            expansions = processed['all_expansions']
            
            avg_expansion_len = np.mean([len(exp.split()) for exp in expansions])
            expansion_stats.append({
                'original': query,
                'original_length': original_len,
                'avg_expansion_length': avg_expansion_len,
                'num_expansions': len(expansions)
            })
        
        return {
            'method': 'query_expansion',
            'expansion_stats': expansion_stats,
            'total_queries': len(self.sample_queries)
        }
    
    def run_full_evaluation(self) -> List[Dict[str, Any]]:
        """Run full evaluation of all components"""
        results = []
        results.append(self.evaluate_preprocessing())
        results.append(self.evaluate_intent_classification())
        results.append(self.evaluate_query_expansion())
        return results


def demonstrate_query_understanding():
    """Demonstrate query understanding techniques"""
    print("=== Query Understanding Demonstration ===\n")
    
    # Initialize pipeline
    pipeline = QueryUnderstandingPipeline()
    
    # Sample queries
    sample_queries = [
        "How does neural network training work?",
        "What is Python programming language?",
        "Compare SVM vs Random Forest",
        "Find docs for React hooks"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"Query {i}: '{query}'")
        
        # Process the query
        result = pipeline.process_query(query)
        
        print(f"  Intent: {result['intent']['intent']} (confidence: {result['intent']['confidence']:.2f})")
        print(f"  Reformulated: '{result['reformulated_query']}'")
        print(f"  Preprocessed tokens: {result['preprocessed']['lemmatized_tokens']}")
        print(f"  Named entities: {[e['text'] for e in result['preprocessed']['entities']]}")
        print(f"  Synonym expansions: {result['synonym_expansions'][:2] if result['synonym_expansions'] else 'None'}")
        print(f"  Entity expanded: '{result['entity_expanded_query']}'")
        print()


def performance_comparison():
    """Compare performance of different query understanding techniques"""
    import time
    
    print("=== Performance Comparison ===\n")
    
    pipeline = QueryUnderstandingPipeline()
    evaluator = QueryUnderstandingEvaluator()
    
    # Time the full pipeline
    start_time = time.time()
    results = evaluator.run_full_evaluation()
    total_time = time.time() - start_time
    
    print(f"Processed {len(evaluator.sample_queries)} queries in {total_time:.4f} seconds")
    print(f"Average time per query: {total_time / len(evaluator.sample_queries):.4f} seconds")
    
    # Show results summary
    for result in results:
        print(f"{result['method']}: Processed {result['total_queries']} queries")


def main():
    """Main function to demonstrate query understanding implementations"""
    print("Module 6: Query Understanding")
    print("=" * 50)
    
    # Demonstrate query understanding techniques
    demonstrate_query_understanding()
    
    # Show performance comparison
    performance_comparison()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of detailed processing
    pipeline = QueryUnderstandingPipeline()
    query = "How to build a machine learning model in Python?"
    
    print(f"Processing query: '{query}'")
    result = pipeline.process_query(query)
    
    print("\nDetailed processing steps:")
    print(f"Original: {result['original_query']}")
    print(f"Intent: {result['intent']['intent']}")
    print(f"Entities: {[e['text'] for e in result['preprocessed']['entities']]}")
    print(f"Final processed query: {result['reformulated_query']}")
    print(f"Synonym expansions: {result['synonym_expansions'][:3]}")
    
    print("\n" + "=" * 50)
    print("Query understanding pipeline demonstrates:")
    print("- Text preprocessing and normalization")
    print("- Intent classification")
    print("- Query expansion techniques")
    print("- Query reformulation strategies")


if __name__ == "__main__":
    main()