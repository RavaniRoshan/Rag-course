"""
RAG Course - Module 3: Embedding Models Implementation

This file contains the implementation of the AdvancedEmbeddingModel class
and related functionality as described in the module.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import time
from sklearn.metrics.pairwise import cosine_similarity
import re


@dataclass
class EmbeddingResult:
    """Represents an embedding with metadata"""
    vector: np.ndarray
    tokens_used: int
    processing_time: float
    model_name: str
    text: str


class AdvancedEmbeddingModel:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 pooling_strategy: str = "mean",
                 normalize_embeddings: bool = True,
                 max_length: int = 512):
        """
        Advanced embedding model with configurable options
        """
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max_length
        
        # Initialize the model based on type
        if "sentence-transformers" in model_name or any(name in model_name for name in [
            "all-MiniLM", "all-mpnet", "all-distilroberta", "multi-qa", "paraphrase"
        ]):
            self.model = SentenceTransformer(model_name)
            self.tokenizer = self.model.tokenizer
        else:
            # For generic transformers models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress_bar: bool = False) -> Union[np.ndarray, List[EmbeddingResult]]:
        """
        Encode text(s) to embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        start_time = time.time()
        
        # Use SentenceTransformer if available
        if hasattr(self, 'model') and isinstance(self.model, SentenceTransformer):
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self.normalize_embeddings
            )
        else:
            # For generic transformers models
            embeddings = self._encode_transformers(texts, batch_size)
        
        # Normalize if required
        if self.normalize_embeddings and not hasattr(self, 'model') or not isinstance(self.model, SentenceTransformer):
            embeddings = self._normalize_embeddings(embeddings)
        
        processing_time = time.time() - start_time
        
        # Return embedding results with metadata
        results = []
        for i, text in enumerate(texts):
            result = EmbeddingResult(
                vector=embeddings[i],
                tokens_used=len(self.tokenizer.encode(text)),
                processing_time=processing_time / len(texts),  # Avg per text
                model_name=self.model_name,
                text=text
            )
            results.append(result)
        
        return results
    
    def _encode_transformers(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using generic transformers model"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                if self.pooling_strategy == "mean":
                    # Mean pooling
                    mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                elif self.pooling_strategy == "cls":
                    # Use [CLS] token
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooling_strategy == "max":
                    # Max pooling
                    mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    masked_embeddings = outputs.last_hidden_state * mask
                    embeddings = torch.max(masked_embeddings, 1)[0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length"""
        if embeddings.ndim == 1:
            embeddings = embeddings / np.linalg.norm(embeddings)
        else:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def similarity(self, 
                   embeddings1: Union[np.ndarray, List[np.ndarray]], 
                   embeddings2: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Calculate similarity between embeddings"""
        if isinstance(embeddings1, list):
            embeddings1 = np.array(embeddings1)
        if isinstance(embeddings2, list):
            embeddings2 = np.array(embeddings2)
        
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        return cosine_similarity(embeddings1, embeddings2)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        # Get a sample embedding to determine dimension
        sample_embedding = self.encode(["sample text"])
        return len(sample_embedding[0].vector)


class MultiModelEmbeddingEnsemble:
    """Combines multiple embedding models for improved performance"""
    
    def __init__(self, 
                 model_names: List[str],
                 weights: Optional[List[float]] = None):
        """
        Initialize ensemble of embedding models
        """
        self.models = [AdvancedEmbeddingModel(name) for name in model_names]
        self.model_names = model_names
        
        if weights is None:
            self.weights = [1.0 / len(model_names)] * len(model_names)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def encode(self, 
               texts: Union[str, List[str]], 
               aggregation_method: str = "weighted_average") -> Union[np.ndarray, List[EmbeddingResult]]:
        """
        Encode using ensemble of models
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Get embeddings from all models
        all_embeddings = []
        for model in self.models:
            model_embeddings = model.encode(texts)
            all_embeddings.append([emb.vector for emb in model_embeddings])
        
        # Convert to numpy arrays
        all_embeddings = np.array(all_embeddings)  # Shape: (n_models, n_texts, embedding_dim)
        
        if aggregation_method == "weighted_average":
            # Weighted average of embeddings
            weighted_embeddings = np.zeros_like(all_embeddings[0])
            for i, weight in enumerate(self.weights):
                weighted_embeddings += weight * all_embeddings[i]
            
            # Return results with combined metadata
            results = []
            for j, text in enumerate(texts):
                combined_vector = weighted_embeddings[j]
                avg_tokens = int(np.mean([model.encode(text)[0].tokens_used for model in self.models]))
                avg_time = np.mean([model.encode(text)[0].processing_time for model in self.models])
                
                result = EmbeddingResult(
                    vector=combined_vector,
                    tokens_used=avg_tokens,
                    processing_time=avg_time,
                    model_name=f"ensemble({','.join(self.model_names)})",
                    text=text
                )
                results.append(result)
                
            return results
        
        elif aggregation_method == "concatenation":
            # Concatenate embeddings from different models
            concatenated_embeddings = np.concatenate(all_embeddings, axis=-1)
            
            results = []
            for j, text in enumerate(texts):
                combined_vector = concatenated_embeddings[j]
                avg_tokens = int(np.mean([model.encode(text)[0].tokens_used for model in self.models]))
                avg_time = np.mean([model.encode(text)[0].processing_time for model in self.models])
                
                result = EmbeddingResult(
                    vector=combined_vector,
                    tokens_used=avg_tokens,
                    processing_time=avg_time,
                    model_name=f"concatenated_ensemble({','.join(self.model_names)})",
                    text=text
                )
                results.append(result)
                
            return results
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def get_combined_dimension(self) -> int:
        """Get the dimension of combined embeddings"""
        if hasattr(self, '_combined_dim'):
            return self._combined_dim
        
        # Calculate based on aggregation method
        dims = [model.get_embedding_dimension() for model in self.models]
        self._combined_dim = sum(dims)  # For concatenation
        return self._combined_dim


class DomainSpecificEmbedder:
    """Embedder specialized for specific domains with custom preprocessing"""
    
    def __init__(self, 
                 base_model: AdvancedEmbeddingModel,
                 domain_preprocessor: Optional[callable] = None):
        self.base_model = base_model
        self.domain_preprocessor = domain_preprocessor or self._default_preprocessor
        self.domain_keywords = set()
    
    def _default_preprocessor(self, text: str) -> str:
        """Default preprocessing for domain-specific text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text
    
    def add_domain_keywords(self, keywords: List[str]) -> None:
        """Add important domain keywords for special handling"""
        self.domain_keywords.update(keywords)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for domain-specific embedding"""
        processed_text = self.domain_preprocessor(text)
        
        # Enhance text with domain knowledge if needed
        if self.domain_keywords:
            # Add context for domain-specific terms
            for keyword in self.domain_keywords:
                if keyword.lower() in processed_text.lower():
                    # Add semantic context (simplified approach)
                    processed_text = processed_text.replace(keyword, f"[DOMAIN] {keyword} [DOMAIN]")
        
        return processed_text
    
    def encode(self, 
               texts: Union[str, List[str]], 
               preprocess: bool = True) -> Union[np.ndarray, List[EmbeddingResult]]:
        """Encode text with domain-specific preprocessing"""
        if isinstance(texts, str):
            texts = [texts]
        
        if preprocess:
            processed_texts = [self.preprocess_text(text) for text in texts]
        else:
            processed_texts = texts
        
        return self.base_model.encode(processed_texts)


class HealthcareEmbedder(DomainSpecificEmbedder):
    """Specialized embedder for healthcare and medical text"""
    
    def __init__(self, base_model: AdvancedEmbeddingModel):
        super().__init__(base_model, self._medical_preprocessor)
        
        # Medical domain keywords and abbreviations
        self.medical_keywords = {
            'conditions': ['hypertension', 'diabetes', 'cancer', 'cardiac', 'stroke', 'asthma'],
            'procedures': ['surgery', 'biopsy', 'chemotherapy', 'radiation', 'transplant'],
            'medications': ['aspirin', 'insulin', 'antibiotic', 'anticoagulant'],
            'anatomy': ['heart', 'lung', 'kidney', 'liver', 'brain', 'spine'],
            'abbreviations': ['EKG', 'MRI', 'CT', 'BP', 'HR', 'ECG', 'CBC', 'RBC', 'WBC']
        }
        
        # Add all keywords to domain set
        all_keywords = set()
        for category, keywords in self.medical_keywords.items():
            all_keywords.update(keywords)
        self.add_domain_keywords(list(all_keywords))
    
    def _medical_preprocessor(self, text: str) -> str:
        """Preprocess medical text with domain knowledge"""
        # Expand common medical abbreviations
        text = self._expand_medical_abbreviations(text)
        
        # Normalize medical terminology
        text = self._normalize_medical_terms(text)
        
        # Remove or standardize medical jargon that might confuse general models
        text = self._standardize_medical_jargon(text)
        
        # Ensure text is cleaned
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text
    
    def _expand_medical_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations"""
        abbrev_expansions = {
            'EKG': 'electrocardiogram',
            'ECG': 'electrocardiogram', 
            'MRI': 'magnetic resonance imaging',
            'CT': 'computed tomography',
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'CBC': 'complete blood count',
            'RBC': 'red blood cell',
            'WBC': 'white blood cell',
            'pt': 'patient',  # Also expand "pt" which could mean "part" or "patient"
            'y/o': 'years old'
        }
        
        for abbrev, expansion in abbrev_expansions.items():
            text = re.sub(rf'\b{abbrev}\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_medical_terms(self, text: str) -> str:
        """Normalize medical terminology to standard forms"""
        # Convert plural forms to singular where appropriate
        text = re.sub(r'\bpatients\b', 'patient', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsurgeries\b', 'surgery', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdiagnoses\b', 'diagnosis', text, flags=re.IGNORECASE)
        
        return text
    
    def _standardize_medical_jargon(self, text: str) -> str:
        """Standardize medical jargon for better embedding"""
        # Medical jargon that should be explained
        jargon_replacements = {
            'cc': 'chief complaint',
            'hpi': 'history of present illness', 
            'pmh': 'past medical history',
            'fh': 'family history',
            'sh': 'social history',
            'ros': 'review of systems'
        }
        
        for jargon, replacement in jargon_replacements.items():
            text = re.sub(rf'\b{jargon}\b', replacement, text, flags=re.IGNORECASE)
        
        return text


class IntelligentEmbeddingSelector:
    """An intelligent system that selects the best embedding model based on text characteristics"""
    
    def __init__(self):
        self.model_performance = {
            # Performance characteristics of different models
            "all-MiniLM-L6-v2": {
                "speed": 0.9,  # High speed
                "accuracy": 0.7,  # Medium accuracy
                "memory": 0.3,  # Low memory usage
                "multilingual": 0.4,  # Limited multilingual
                "domain": "general"
            },
            "all-mpnet-base-v2": {
                "speed": 0.6,
                "accuracy": 0.9,  # High accuracy
                "memory": 0.7,  # High memory usage
                "multilingual": 0.3,
                "domain": "general"
            },
            "paraphrase-multilingual-MiniLM-L12-v2": {
                "speed": 0.7,
                "accuracy": 0.8,
                "memory": 0.5,
                "multilingual": 0.9,  # Excellent multilingual
                "domain": "multilingual"
            },
            "multi-qa-MiniLM-L6-cos-v1": {
                "speed": 0.8,
                "accuracy": 0.8,
                "memory": 0.4,
                "multilingual": 0.4,
                "domain": "qa"
            }
        }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text characteristics to inform model selection"""
        characteristics = {}
        
        # Language detection (simplified)
        words = text.lower().split()
        non_english_words = sum(1 for word in words if self._is_non_english_word(word))
        characteristics["non_english_ratio"] = non_english_words / len(words) if words else 0
        
        # Length characteristics
        characteristics["length"] = len(text)
        characteristics["word_count"] = len(words)
        characteristics["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        
        # Character diversity
        char_diversity = len(set(text)) / len(text) if text else 0
        characteristics["char_diversity"] = char_diversity
        
        # Special characters ratio
        special_chars = len([c for c in text if not c.isalnum() and not c.isspace()])
        characteristics["special_char_ratio"] = special_chars / len(text) if text else 0
        
        return characteristics
    
    def _is_non_english_word(self, word: str) -> bool:
        """Simple check if a word might be non-English (this is very simplified)"""
        # In reality, use a proper language detection library
        non_english_indicators = ['le', 'la', 'el', 'der', 'die', 'das', 'un', 'une', 'il', 'lo']
        return word.lower() in non_english_indicators
    
    def select_best_model(self, 
                         text: str, 
                         requirements: Dict[str, float] = None) -> str:
        """
        Select the best model based on text characteristics and requirements
        """
        if requirements is None:
            requirements = {
                "speed_importance": 0.5,
                "accuracy_importance": 0.5,
                "multilingual_importance": 0.3
            }
        
        # Analyze text
        text_characteristics = self.analyze_text(text)
        
        # Calculate scores for each model
        model_scores = {}
        
        for model_name, perf in self.model_performance.items():
            score = 0
            
            # Calculate weighted score based on requirements
            score += requirements["speed_importance"] * perf["speed"]
            score += requirements["accuracy_importance"] * perf["accuracy"]
            
            # Add multilingual bonus if needed
            if text_characteristics["non_english_ratio"] > 0.3:
                score += requirements["multilingual_importance"] * perf["multilingual"]
            
            # Apply text-specific adjustments
            if text_characteristics["length"] > 500:  # Long text
                if perf["memory"] > 0.6:  # High memory models handle long texts better
                    score += 0.1
            
            model_scores[model_name] = score
        
        # Return the model with highest score
        best_model = max(model_scores.items(), key=lambda x: x[1])
        return best_model[0]
    
    def encode_with_intelligence(self, 
                                text: str, 
                                requirements: Dict[str, float] = None) -> EmbeddingResult:
        """Encode text using intelligently selected model"""
        best_model_name = self.select_best_model(text, requirements)
        
        # Initialize and use the selected model
        model = AdvancedEmbeddingModel(best_model_name)
        result = model.encode(text)[0]
        
        # Add metadata about model selection
        result.metadata = {
            "selected_by": "intelligent_selector",
            "model_reasoning": self.select_best_model(text, requirements),
            "text_characteristics": self.analyze_text(text)
        }
        
        return result


def main():
    """Example usage of the embedding models"""
    print("RAG Course - Module 3: Embedding Models Example")
    print("=" * 55)
    
    # Example 1: Basic embedding
    print("\n1. Basic Embedding Example:")
    model = AdvancedEmbeddingModel("all-MiniLM-L6-v2")
    
    sample_texts = [
        "This is a sample sentence for embedding.",
        "Another example sentence to encode.",
        "A completely different topic altogether."
    ]
    
    results = model.encode(sample_texts)
    for i, result in enumerate(results):
        print(f"  Text {i+1}: '{result.text[:30]}...'")
        print(f"    Embedding shape: {result.vector.shape}")
        print(f"    Tokens used: {result.tokens_used}")
        print(f"    Model: {result.model_name}")
    
    # Example 2: Similarity calculation
    print("\n2. Similarity Calculation:")
    emb1 = model.encode("The weather is beautiful today")[0].vector
    emb2 = model.encode("Today's weather is nice")[0].vector
    emb3 = model.encode("Completely different topic")[0].vector
    
    sim12 = cosine_similarity([emb1], [emb2])[0][0]
    sim13 = cosine_similarity([emb1], [emb3])[0][0]
    
    print(f"  Similarity between similar sentences: {sim12:.3f}")
    print(f"  Similarity between different sentences: {sim13:.3f}")
    
    # Example 3: Multi-model ensemble
    print("\n3. Multi-Model Ensemble:")
    ensemble = MultiModelEmbeddingEnsemble([
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L3-v2"
    ])
    
    ensemble_results = ensemble.encode(sample_texts[:2])
    for i, result in enumerate(ensemble_results):
        print(f"  Ensemble result {i+1}: shape {result.vector.shape}, model {result.model_name}")
    
    # Example 4: Healthcare embedder
    print("\n4. Healthcare Domain Embedder:")
    health_model = AdvancedEmbeddingModel("all-MiniLM-L6-v2")
    health_embedder = HealthcareEmbedder(health_model)
    
    medical_text = "Pt. has HTN and DM. Scheduled for EKG next week."
    health_result = health_embedder.encode(medical_text)[0]
    print(f"  Medical text: '{medical_text}'")
    print(f"  Processed: '{health_embedder.preprocess_text(medical_text)}'")
    print(f"  Embedding shape: {health_result.vector.shape}")
    
    # Example 5: Intelligent model selection
    print("\n5. Intelligent Model Selection:")
    selector = IntelligentEmbeddingSelector()
    
    test_texts = [
        "This is a simple English sentence.",
        "Le Lorem ipsum dolor sit amet.",
        "This is a longer text that might benefit from a more accurate model for better semantic understanding and context preservation."
    ]
    
    for text in test_texts:
        best_model = selector.select_best_model(text)
        characteristics = selector.analyze_text(text)
        print(f"  Text: '{text[:40]}...'")
        print(f"    Best model: {best_model}")
        print(f"    Length: {characteristics['length']}, Non-English ratio: {characteristics['non_english_ratio']:.2f}")


if __name__ == "__main__":
    main()