# **Deep Dive: Module 3 - Embedding Models (Selection, Evaluation, and Optimization)**

## **1. Advanced Theoretical Foundations**

### **The Mathematical Basis of Embedding Models**

Embedding models transform text into high-dimensional vector spaces where semantic similarity corresponds to geometric proximity. The fundamental mathematical principle involves:

```
f(text) → embedding_vector ∈ ℝ^d

Where:
- f is the embedding function (neural network encoder)
- d is the dimensionality (typically 384, 512, 768, 1024, or 3840)
- Similarity(x, y) = cosine_similarity(emb(x), emb(y))
```

The mathematical foundation relies on:

- **Vector Space Model**: Text represented as dense vectors in high-dimensional space
- **Distributional Semantics**: Words appearing in similar contexts have similar meanings
- **Attention Mechanisms**: Contextual understanding of word relationships
- **Contrastive Learning**: Training to pull similar texts together and push dissimilar ones apart

### **Embedding Model Architectures**

- **Dense Embeddings**: Sentence Transformers, BERT, RoBERTa (meaningful representations)
- **Sparse Embeddings**: BM25, TF-IDF (traditional keyword-based)
- **Hybrid Embeddings**: Combining dense and sparse representations
- **Multimodal Embeddings**: Text-image, text-audio representations

### **Key Mathematical Concepts**

- **Cosine Similarity**: Σ(xi * yi) / (√(Σxi²) * √(Σyi²))
- **Euclidean Distance**: √(Σ(xi - yi)²)
- **Dot Product**: Σ(xi * yi) (for normalized embeddings)
- **Contrastive Loss**: L = -log(exp(sim_pos)/(exp(sim_pos) + Σexp(sim_neg)))

## **2. Extended Technical Implementation**

### **Advanced Embedding Model Implementation with Multiple Options**

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import time
import json
from sklearn.metrics.pairwise import cosine_similarity
import faiss


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


class EmbeddingSelector:
    """Selects the best embedding model for a specific use case"""
    
    def __init__(self):
        self.model_performance = {}
        self.recommendations = {
            "general_purpose": "all-MiniLM-L6-v2",
            "semantic_search": "multi-qa-MiniLM-L6-cos-v1",
            "sentence_similarity": "all-MiniLM-L12-v2",
            "paraphrase_detection": "paraphrase-MiniLM-L6-v2",
            "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
            "long_context": "all-mpnet-base-v2",
            "retrieval": "msmarco-distilbert-base-v4"
        }
    
    def select_model(self, 
                     use_case: str, 
                     language: str = "english",
                     performance_requirements: Dict[str, float] = None) -> str:
        """
        Select the best embedding model based on use case and requirements
        """
        if performance_requirements is None:
            performance_requirements = {"speed": 0.5, "accuracy": 0.5}
        
        # Base recommendation
        base_model = self.recommendations.get(use_case, "all-MiniLM-L6-v2")
        
        # Adjust based on language
        if language != "english" and use_case != "multilingual":
            base_model = self.recommendations.get("multilingual", base_model)
        
        # Adjust based on performance requirements
        if performance_requirements.get("speed", 0.5) > 0.7:
            # Prioritize faster models
            fast_models = {
                "all-MiniLM-L6-v2": "Fast and good for general purpose",
                "paraphrase-MiniLM-L3-v2": "Fastest option, good for similarity tasks"
            }
            return list(fast_models.keys())[0]  # Return the fastest option
        
        if performance_requirements.get("accuracy", 0.5) > 0.8:
            # Prioritize more accurate models
            accurate_models = {
                "all-mpnet-base-v2": "Most accurate for general purpose",
                "multi-qa-mpnet-base-dot-v1": "Best for question answering",
                "msmarco-bert-base-dot-v5": "Best for retrieval tasks"
            }
            return list(accurate_models.keys())[0]  # Return the most accurate
        
        return base_model
    
    def evaluate_model_performance(self, 
                                   model_name: str, 
                                   test_dataset: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """
        Evaluate model performance on a test dataset
        """
        model = AdvancedEmbeddingModel(model_name)
        
        predictions = []
        ground_truth = []
        
        for text1, text2, similarity_score in test_dataset:
            emb1 = model.encode(text1)[0].vector
            emb2 = model.encode(text2)[0].vector
            
            pred_similarity = cosine_similarity([emb1], [emb2])[0][0]
            
            predictions.append(pred_similarity)
            ground_truth.append(similarity_score)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - ground_truth))
        pearson_corr = np.corrcoef(predictions, ground_truth)[0, 1]
        
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Pearson_Correlation": pearson_corr,
            "model_name": model_name
        }


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
```

### **Embedding Model Fine-Tuning and Custom Training**

```python
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset


class EmbeddingModelTrainer:
    """Class for fine-tuning embedding models on domain-specific data"""
    
    def __init__(self, base_model_name: str = "all-MiniLM-L6-v2"):
        self.base_model_name = base_model_name
        self.model = SentenceTransformer(base_model_name)
    
    def prepare_training_data(self, 
                             text_pairs: List[Tuple[str, str, float]], 
                             output_path: str) -> List[InputExample]:
        """
        Prepare training data from text pairs with similarity scores
        """
        examples = []
        for text1, text2, score in text_pairs:
            # Convert similarity score (0-1) to label for training
            # For MultipleNegativesRankingLoss, we use 1 for positive pairs
            examples.append(InputExample(texts=[text1, text2], label=score))
        
        # Save to file if needed
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(f"{example.texts[0]} ||| {example.texts[1]} ||| {example.label}\n")
        
        return examples
    
    def train_model(self, 
                   training_examples: List[InputExample],
                   output_path: str,
                   epochs: int = 1,
                   batch_size: int = 16,
                   warmup_steps: int = 100) -> None:
        """
        Fine-tune the embedding model
        """
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
        
        # Define the training loss
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )
    
    def train_with_triplets(self, 
                           triplets: List[Tuple[str, str, str]],  # anchor, positive, negative
                           output_path: str,
                           epochs: int = 1,
                           batch_size: int = 16) -> None:
        """
        Train model using triplet loss (anchor, positive, negative)
        """
        train_examples = []
        for anchor, positive, negative in triplets:
            # For triplet loss, we create pairs for MultipleNegativesRankingLoss
            train_examples.append(InputExample(texts=[anchor, positive]))
            train_examples.append(InputExample(texts=[anchor, negative]))
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Use MultipleNegativesRankingLoss for triplet-like training
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            output_path=output_path,
            show_progress_bar=True
        )
    
    def evaluate_model(self, 
                      model_path: str, 
                      test_pairs: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """
        Evaluate the trained model
        """
        # Load the fine-tuned model
        model = SentenceTransformer(model_path)
        
        predictions = []
        ground_truth = []
        
        for text1, text2, similarity_score in test_pairs:
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)
            
            pred_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            predictions.append(float(pred_similarity))
            ground_truth.append(similarity_score)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - ground_truth))
        pearson_corr = np.corrcoef(predictions, ground_truth)[0, 1]
        
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Pearson_Correlation": pearson_corr
        }


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
    
    def encode_with_context(self, 
                           primary_text: str, 
                           context_texts: List[str],
                           weight: float = 0.7) -> EmbeddingResult:
        """Encode text with additional context from related documents"""
        # Combine primary text with context
        context_str = " ".join(context_texts[:3])  # Use first 3 context texts
        combined_text = f"{primary_text} [CONTEXT] {context_str}"
        
        # Encode the combined text
        result = self.base_model.encode(combined_text)[0]
        
        # Modify result to indicate context was used
        result.text = primary_text
        result.metadata = {"context_used": True, "context_count": len(context_texts)}
        
        return result


class AdaptiveEmbeddingSelector:
    """Selects embedding model adaptively based on input characteristics"""
    
    def __init__(self, 
                 model_catalog: Dict[str, AdvancedEmbeddingModel] = None):
        if model_catalog is None:
            # Default catalog of models
            self.model_catalog = {
                "fast_general": AdvancedEmbeddingModel("all-MiniLM-L6-v2"),
                "accurate_general": AdvancedEmbeddingModel("all-mpnet-base-v2"),
                "semantic_search": AdvancedEmbeddingModel("multi-qa-MiniLM-L6-cos-v1"),
                "multilingual": AdvancedEmbeddingModel("paraphrase-multilingual-MiniLM-L12-v2")
            }
        else:
            self.model_catalog = model_catalog
        
        self.performance_cache = {}
    
    def analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze text to determine optimal model"""
        characteristics = {
            "length": len(text),
            "word_count": len(text.split()),
            "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
            "language_indicators": self._detect_language_indicators(text),
            "special_chars": len([c for c in text if not c.isalnum() and c != ' ']),
            "numerical_content": len([c for c in text if c.isdigit()]),
            "casing_variety": self._analyze_casing_variety(text)
        }
        
        return characteristics
    
    def _detect_language_indicators(self, text: str) -> Dict[str, float]:
        """Detect potential language based on character patterns"""
        # Simplified language detection
        text_lower = text.lower()
        
        # Common words in different languages
        language_indicators = {
            "english": ["the", "and", "for", "are", "but", "not", "you", "all", "can", "had"],
            "spanish": ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"],
            "french": ["le", "de", "et", "à", "un", "il", "être", "et", "avoir", "je"],
            "german": ["der", "die", "und", "in", "den", "von", "zu", "das", "nicht", "sie"]
        }
        
        indicators = {}
        for lang, words in language_indicators.items():
            count = sum(1 for word in words if word in text_lower)
            indicators[lang] = count / len(words)  # Normalize by number of indicator words
        
        return indicators
    
    def _analyze_casing_variety(self, text: str) -> float:
        """Analyze the variety of casing patterns in text"""
        upper_count = sum(1 for c in text if c.isupper())
        lower_count = sum(1 for c in text if c.islower())
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return 0.0
        
        # Calculate variety index (how mixed the case is)
        upper_ratio = upper_count / total_alpha
        lower_ratio = lower_count / total_alpha
        return abs(upper_ratio - lower_ratio)  # 0 = all same case, 1 = balanced
    
    def select_model_for_text(self, text: str) -> str:
        """Select the best model for a specific text"""
        characteristics = self.analyze_text_characteristics(text)
        
        # Define model selection rules based on characteristics
        if characteristics["language_indicators"]["english"] < 0.1:  # Likely non-English
            return "multilingual"
        elif characteristics["word_count"] > 200:  # Long text
            return "accurate_general"
        elif characteristics["special_chars"] > characteristics["word_count"] * 0.2:  # High special character density
            # Probably contains special formatting, use robust model
            return "accurate_general"
        else:
            # Default to fast model for general purpose
            return "fast_general"
    
    def adaptive_encode(self, 
                       texts: Union[str, List[str]], 
                       selection_strategy: str = "per_text") -> List[EmbeddingResult]:
        """Encode using adaptive model selection"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        if selection_strategy == "per_text":
            # Select model for each text individually
            for text in texts:
                model_name = self.select_model_for_text(text)
                model = self.model_catalog[model_name]
                result = model.encode(text)[0]
                result.model_name = f"adaptive_{model_name}"
                results.append(result)
        
        elif selection_strategy == "batch_optimal":
            # Analyze all texts to select one optimal model for batch
            all_characteristics = [self.analyze_text_characteristics(text) for text in texts]
            
            # For simplicity, use the most common recommendation
            model_recommendations = [self.select_model_for_text(text) for text in texts]
            most_common_model = max(set(model_recommendations), key=model_recommendations.count)
            
            # Use this model for all texts
            model = self.model_catalog[most_common_model]
            batch_results = model.encode(texts)
            
            for i, result in enumerate(batch_results):
                result.model_name = f"adaptive_{most_common_model}_batch"
                results.append(result)
        
        return results
```

## **3. Advanced Real-World Applications**

### **Healthcare Domain Embedder**

```python
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
    
    def encode_medical_document(self, document: Dict[str, str]) -> EmbeddingResult:
        """Encode a complete medical document with section awareness"""
        # Combine different sections with appropriate weighting
        sections = []
        
        # Add sections in order of importance
        for section_name in ['chief_complaint', 'history', 'findings', 'impression', 'plan']:
            if section_name in document:
                sections.append(f"[{section_name.upper()}] {document[section_name]}")
        
        combined_text = " ".join(sections)
        return self.encode(combined_text, preprocess=True)[0]
    
    def calculate_medical_similarity(self, 
                                   text1: str, 
                                   text2: str,
                                   context_aware: bool = True) -> float:
        """Calculate similarity with medical context awareness"""
        if context_aware:
            # Enhance with medical preprocessing
            emb1 = self.encode(text1, preprocess=True)[0].vector
            emb2 = self.encode(text2, preprocess=True)[0].vector
        else:
            # Use basic preprocessing
            emb1 = self.base_model.encode(text1)[0].vector
            emb2 = self.base_model.encode(text2)[0].vector
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)


import re  # Need to import re for the above class


class LegalDocumentEmbedder(DomainSpecificEmbedder):
    """Specialized embedder for legal documents"""
    
    def __init__(self, base_model: AdvancedEmbeddingModel):
        super().__init__(base_model, self._legal_preprocessor)
        
        # Legal domain keywords
        self.legal_keywords = [
            'court', 'case', 'statute', 'regulation', 'section', 'subsection',
            'defendant', 'plaintiff', 'petitioner', 'respondent', 'counsel',
            'motion', 'brief', 'opinion', 'precedent', 'jurisdiction',
            'contract', 'agreement', 'clause', 'provision', 'enforceable'
        ]
        self.add_domain_keywords(self.legal_keywords)
    
    def _legal_preprocessor(self, text: str) -> str:
        """Preprocess legal text"""
        # Remove case citations and legal formatting that might confuse embeddings
        text = re.sub(r'\d+\s+\w+\s+\d+', '', text)  # Remove citations like "123 F.3d 456"
        text = re.sub(r'[IVX]+\.?\s', '', text, flags=re.IGNORECASE)  # Remove Roman numerals at start of lines
        
        # Normalize legal terminology
        text = re.sub(r'\b(?:v\.|v |versus)\b', 'versus', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(?:§|section)\b', 'section', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def encode_case_document(self, 
                           case_title: str, 
                           case_text: str,
                           include_title: bool = True) -> EmbeddingResult:
        """Encode legal case document with special handling"""
        if include_title:
            full_text = f"CASE TITLE: {case_title}. CONTENT: {case_text}"
        else:
            full_text = case_text
        
        return self.encode(full_text, preprocess=True)[0]
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from text"""
        entities = {
            'cases': re.findall(r'[A-Z][A-Z\s\-.]+v\.\s*[A-Z][A-Z\s\-.]+', text),
            'statutes': re.findall(r'[A-Z][A-Z\s-]+\s+\d+', text),
            'sections': re.findall(r'§\s*\d+[a-z]*(?:\([a-z]\))*', text),
            'parties': []
        }
        
        # Extract potential party names
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            # Look for capitalized words that might be party names
            words = sentence.split()
            potential_parties = [word.strip(',.()') for word in words if word[0].isupper() and len(word) > 2]
            entities['parties'].extend(potential_parties)
        
        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return entities


class CodeEmbedder(DomainSpecificEmbedder):
    """Specialized embedder for source code"""
    
    def __init__(self, base_model: AdvancedEmbeddingModel):
        super().__init__(base_model, self._code_preprocessor)
        
        # Code-related keywords
        self.code_keywords = [
            'function', 'class', 'method', 'variable', 'parameter', 'return',
            'import', 'include', 'library', 'module', 'package', 'framework'
        ]
        self.add_domain_keywords(self.code_keywords)
    
    def _code_preprocessor(self, text: str) -> str:
        """Preprocess code text"""
        # Remove comments
        text = re.sub(r'#.*', '', text)  # Single-line comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # Multi-line comments
        
        # Remove code-specific characters that don't add semantic meaning
        text = re.sub(r'[{}[\];(),.:]', ' ', text)  # Brackets, punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def encode_code_snippet(self, 
                           code: str, 
                           language: str = "generic") -> EmbeddingResult:
        """Encode code snippet with language awareness"""
        # Add language context
        if language != "generic":
            code_with_context = f"LANGUAGE: {language}. CODE: {code}"
        else:
            code_with_context = code
        
        return self.encode(code_with_context, preprocess=True)[0]
    
    def encode_function_signature(self, 
                                 name: str, 
                                 parameters: str,
                                 return_type: str = "",
                                 description: str = "") -> EmbeddingResult:
        """Encode function signature with documentation"""
        signature = f"FUNCTION: {name}({parameters})"
        if return_type:
            signature += f" -> {return_type}"
        if description:
            signature += f". DESCRIPTION: {description}"
        
        return self.encode(signature, preprocess=True)[0]
```

## **4. Performance Optimization Strategies**

### **Efficient Embedding Computation**

```python
import asyncio
import concurrent.futures
from functools import lru_cache
import pickle
import hashlib


class OptimizedEmbeddingModel(AdvancedEmbeddingModel):
    """Optimized embedding model with caching and batching"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 use_cache: bool = True,
                 cache_size: int = 10000,
                 max_batch_size: int = 64):
        super().__init__(model_name)
        self.use_cache = use_cache
        self.max_batch_size = max_batch_size
        
        if use_cache:
            # Custom cache implementation for embeddings
            self._cache = {}
            self._cache_order = []  # Track insertion order for LRU
            self._cache_size = cache_size
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cache_get(self, text: str) -> Optional[EmbeddingResult]:
        """Get embedding from cache"""
        if not self.use_cache:
            return None
        
        key = self._get_cache_key(text)
        if key in self._cache:
            # Move to end to mark as recently used
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]
        
        return None
    
    def _cache_put(self, text: str, result: EmbeddingResult) -> None:
        """Put embedding in cache"""
        if not self.use_cache:
            return
        
        key = self._get_cache_key(text)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self._cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = result
        self._cache_order.append(key)
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 32,
               use_cache: bool = True) -> Union[np.ndarray, List[EmbeddingResult]]:
        """
        Optimized encode with caching and intelligent batching
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache first
        uncached_texts = []
        cached_results = {}
        
        for i, text in enumerate(texts):
            if use_cache:
                cached = self._cache_get(text)
                if cached:
                    cached_results[i] = cached
                else:
                    uncached_texts.append((i, text))
            else:
                uncached_texts.append((i, text))
        
        # Process uncached texts
        uncached_results = []
        if uncached_texts:
            uncached_batch = [text for i, text in uncached_texts]
            
            # Use parent's encode method for uncached items
            parent_results = super().encode(uncached_batch, min(batch_size, self.max_batch_size))
            
            # Cache the results
            for j, (i, text) in enumerate(uncached_texts):
                result = parent_results[j]
                uncached_results.append((i, result))
                
                # Cache the result
                if use_cache:
                    self._cache_put(text, result)
        
        # Combine cached and uncached results in original order
        all_results = [None] * len(texts)
        
        # Fill in cached results
        for i, cached_result in cached_results.items():
            all_results[i] = cached_result
        
        # Fill in uncached results
        for i, result in uncached_results:
            all_results[i] = result
        
        return all_results


class QuantizedEmbeddingModel:
    """Embedding model with quantization for memory efficiency"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 quantization_bits: int = 8):
        self.model_name = model_name
        self.quantization_bits = quantization_bits
        self.original_model = SentenceTransformer(model_name)
        self.quantization_scale = 2 ** quantization_bits - 1
        self.model_dimension = self.original_model.get_sentence_embedding_dimension()
    
    def quantize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Quantize embedding to lower precision"""
        # Normalize embedding to [0, 1]
        normalized = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-8)
        # Scale to quantization range
        quantized = (normalized * self.quantization_scale).astype(np.uint8)
        return quantized
    
    def dequantize_embedding(self, quantized_embedding: np.ndarray) -> np.ndarray:
        """Restore quantized embedding to original precision"""
        # Convert back to float and denormalize
        dequantized = quantized_embedding.astype(np.float32) / self.quantization_scale
        # Restore original range (this is an approximation)
        original_range = self.original_model.encode(["test"])[0]  # Get a sample to determine range
        return dequantized * (original_range.max() - original_range.min()) + original_range.min()
    
    def encode(self, texts: Union[str, List[str]]) -> List[EmbeddingResult]:
        """Encode with quantized embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Get original embeddings
        original_embeddings = self.original_model.encode(texts, convert_to_numpy=True)
        
        results = []
        for i, text in enumerate(texts):
            if original_embeddings.ndim == 1:
                original_emb = original_embeddings
            else:
                original_emb = original_embeddings[i]
            
            # Quantize
            quantized_emb = self.quantize_embedding(original_emb)
            
            result = EmbeddingResult(
                vector=quantized_emb,
                tokens_used=len(self.original_model.tokenizer.encode(text)),
                processing_time=0.0,  # Not measured in this example
                model_name=f"quantized_{self.model_name}_{self.quantization_bits}bit",
                text=text
            )
            results.append(result)
        
        return results
    
    def similarity(self, 
                   quantized_emb1: np.ndarray, 
                   quantized_emb2: np.ndarray) -> float:
        """Calculate similarity between quantized embeddings"""
        # Dequantize for similarity calculation
        emb1 = self.dequantize_embedding(quantized_emb1)
        emb2 = self.dequantize_embedding(quantized_emb2)
        return cosine_similarity([emb1], [emb2])[0][0]


class BatchOptimizedEmbedder:
    """Embedder optimized for batch processing"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 max_concurrent_batches: int = 4):
        self.model = SentenceTransformer(model_name)
        self.max_concurrent_batches = max_concurrent_batches
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def encode_batch_optimized(self, 
                              texts: List[str], 
                              batch_size: int = 32,
                              num_workers: int = 4) -> List[EmbeddingResult]:
        """Optimized batch encoding using multiprocessing"""
        if len(texts) <= batch_size:
            # If small enough, encode directly
            embeddings = self.model.encode(texts, batch_size=batch_size)
            return [EmbeddingResult(
                vector=emb,
                tokens_used=len(self.model.tokenizer.encode(text)),
                processing_time=0.0,
                model_name=self.model._modules["0"].auto_model.__class__.__name__,
                text=text
            ) for emb, text in zip(embeddings, texts)]
        
        # Split into batches
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches in parallel
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_results = future.result()
                all_results.extend(batch_results)
        
        return all_results
    
    def _process_batch(self, batch_texts: List[str]) -> List[EmbeddingResult]:
        """Process a single batch of texts"""
        embeddings = self.model.encode(batch_texts)
        return [EmbeddingResult(
            vector=emb,
            tokens_used=len(self.model.tokenizer.encode(text)),
            processing_time=0.0,
            model_name=self.model._modules["0"].auto_model.__class__.__name__,
            text=text
        ) for emb, text in zip(embeddings, batch_texts)]
    
    def adaptive_batch_size(self, 
                           texts: List[str], 
                           target_memory_gb: float = 2.0) -> int:
        """Determine optimal batch size based on memory constraints"""
        # Estimate memory usage per text (rough approximation)
        sample_text = texts[0] if texts else "sample text"
        sample_tokens = len(self.model.tokenizer.encode(sample_text))
        estimated_memory_per_item = sample_tokens * 4 * 4  # 4 bytes per float * 4x overhead
        
        # Calculate max batch size
        max_items = int((target_memory_gb * 1024 * 1024 * 1024) / estimated_memory_per_item)
        
        # Use conservative estimate
        return min(max_items, 64)  # Cap at 64 for safety


class MemoryEfficientEmbedder:
    """Embedder designed for minimal memory usage"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 max_memory_embeddings: int = 1000):
        self.model_name = model_name
        self.max_memory_embeddings = max_memory_embeddings
        self.embeddings = {}  # Cache of embeddings
        self.access_order = []  # LRU tracking
    
    def encode(self, texts: Union[str, List[str]]) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """Memory-efficient encoding with LRU cache"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Use SentenceTransformer model directly for single encoding
        # This minimizes memory by not storing intermediate results unnecessarily
        results = []
        
        for text in texts:
            # Generate embedding directly
            model = SentenceTransformer(self.model_name)
            embedding = model.encode(text, convert_to_numpy=True)
            
            result = EmbeddingResult(
                vector=embedding,
                tokens_used=len(model.tokenizer.encode(text)),
                processing_time=0.0,
                model_name=self.model_name,
                text=text
            )
            
            results.append(result)
            
            # Clean up model reference to free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results if len(results) > 1 else results[0]
    
    def encode_streaming(self, 
                        text_iterator, 
                        output_callback: callable) -> None:
        """Encode text in streaming fashion to minimize memory"""
        model = SentenceTransformer(self.model_name)
        
        for i, text in enumerate(text_iterator):
            embedding = model.encode(text, convert_to_numpy=True)
            
            result = EmbeddingResult(
                vector=embedding,
                tokens_used=len(model.tokenizer.encode(text)),
                processing_time=0.0,
                model_name=self.model_name,
                text=text
            )
            
            output_callback(result)
            
            # Periodically clear cache
            if i % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Final cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
```

## **5. Evaluation Framework for Embedding Models**

```python
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class EmbeddingEvaluator:
    """Comprehensive evaluation framework for embedding models"""
    
    def __init__(self):
        self.metrics = {
            'similarity_accuracy': [],
            'clustering_quality': [],
            'dimensionality_reduction': [],
            'downstream_task_performance': []
        }
    
    def evaluate_model(self, 
                      model: AdvancedEmbeddingModel, 
                      test_datasets: Dict[str, List]) -> Dict[str, Dict[str, float]]:
        """Evaluate embedding model across multiple dimensions"""
        results = {}
        
        for dataset_name, dataset in test_datasets.items():
            if dataset_name == "similarity":
                results[dataset_name] = self._evaluate_similarity(model, dataset)
            elif dataset_name == "clustering":
                results[dataset_name] = self._evaluate_clustering(model, dataset)
            elif dataset_name == "classification":
                results[dataset_name] = self._evaluate_classification(model, dataset)
        
        return results
    
    def _evaluate_similarity(self, model: AdvancedEmbeddingModel, dataset: List) -> Dict[str, float]:
        """Evaluate similarity prediction accuracy"""
        predictions = []
        ground_truth = []
        
        for text1, text2, similarity_score in dataset:
            emb1 = model.encode(text1)[0].vector
            emb2 = model.encode(text2)[0].vector
            
            pred_similarity = cosine_similarity([emb1], [emb2])[0][0]
            
            predictions.append(pred_similarity)
            ground_truth.append(similarity_score)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - ground_truth))
        pearson_corr, _ = scipy.stats.pearsonr(predictions, ground_truth)
        spearman_corr, _ = scipy.stats.spearmanr(predictions, ground_truth)
        
        return {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "Pearson_Correlation": float(pearson_corr),
            "Spearman_Correlation": float(spearman_corr)
        }
    
    def _evaluate_clustering(self, model: AdvancedEmbeddingModel, dataset: List) -> Dict[str, float]:
        """Evaluate clustering quality of embeddings"""
        texts, labels = zip(*dataset)  # Assuming dataset is [(text, label), ...]
        
        # Generate embeddings
        embedding_results = model.encode(list(texts))
        embeddings = np.array([result.vector for result in embedding_results])
        
        # Perform clustering
        n_clusters = len(set(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate clustering metrics
        from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
        
        ari_score = adjusted_rand_score(labels, cluster_labels)
        silhouette = silhouette_score(embeddings, cluster_labels)
        calinski = calinski_harabasz_score(embeddings, cluster_labels)
        
        return {
            "Adjusted_Rand_Index": float(ari_score),
            "Silhouette_Score": float(silhouette),
            "Calinski_Harabasz_Index": float(calinski)
        }
    
    def _evaluate_classification(self, model: AdvancedEmbeddingModel, dataset: List) -> Dict[str, float]:
        """Evaluate embeddings for classification tasks"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
        
        texts, labels = zip(*dataset)
        
        # Generate embeddings
        embedding_results = model.encode(list(texts))
        embeddings = np.array([result.vector for result in embedding_results])
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        # Train classifier and evaluate
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(classifier, embeddings, encoded_labels, cv=5)
        
        return {
            "Cross_Val_Accuracy_Mean": float(cv_scores.mean()),
            "Cross_Val_Accuracy_Std": float(cv_scores.std()),
            "Max_CV_Score": float(cv_scores.max()),
            "Min_CV_Score": float(cv_scores.min())
        }
    
    def compare_models(self, 
                      models: Dict[str, AdvancedEmbeddingModel], 
                      test_datasets: Dict[str, List]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compare multiple embedding models"""
        comparison_results = {}
        
        for model_name, model in models.items():
            comparison_results[model_name] = self.evaluate_model(model, test_datasets)
        
        return comparison_results
    
    def visualize_embeddings(self, 
                           model: AdvancedEmbeddingModel, 
                           texts: List[str], 
                           labels: Optional[List[str]] = None,
                           output_path: Optional[str] = None) -> None:
        """Visualize embeddings in 2D space using PCA"""
        # Generate embeddings
        embedding_results = model.encode(texts)
        embeddings = np.array([result.vector for result in embedding_results])
        
        # Reduce dimensionality to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if labels:
            # Color by labels if provided
            unique_labels = list(set(labels))
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(
                    embeddings_2d[mask, 0], 
                    embeddings_2d[mask, 1], 
                    c=[colors[i]], 
                    label=label, 
                    alpha=0.7
                )
            plt.legend()
        else:
            # Plot all as same color
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        plt.title(f"Embedding Visualization: {model.model_name}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        
        if output_path:
            plt.savefig(output_path)
        plt.show()
    
    def analyze_dimensionality(self, 
                              model: AdvancedEmbeddingModel, 
                              texts: List[str]) -> Dict[str, float]:
        """Analyze the effective dimensionality of embeddings"""
        # Generate embeddings
        embedding_results = model.encode(texts)
        embeddings = np.array([result.vector for result in embedding_results])
        
        # Calculate PCA to determine effective dimensions
        pca = PCA()
        pca.fit(embeddings)
        
        # Find number of components that explain 95% of variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
        n_components_99 = np.argmax(cumsum_var >= 0.99) + 1
        
        return {
            "total_dimensions": embeddings.shape[1],
            "components_for_95_percent_variance": int(n_components_95),
            "components_for_99_percent_variance": int(n_components_99),
            "explained_variance_ratio_first_3": pca.explained_variance_ratio_[:3].tolist()
        }


class ModelSelectionFramework:
    """Framework for selecting optimal embedding models"""
    
    def __init__(self):
        self.evaluator = EmbeddingEvaluator()
        self.model_catalog = {
            "efficient": "all-MiniLM-L6-v2",
            "balanced": "all-mpnet-base-v2", 
            "accurate": "all-roberta-large-v1",
            "semantic_search": "multi-qa-MiniLM-L6-cos-v1",
            "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"
        }
    
    def recommend_model(self, 
                       task_requirements: Dict[str, any],
                       evaluation_data: Dict[str, List]) -> Dict[str, any]:
        """Recommend best model based on task requirements"""
        
        # Initialize models to compare
        models_to_evaluate = {}
        
        # Select models based on requirements
        if task_requirements.get("multilingual", False):
            models_to_evaluate["multilingual"] = AdvancedEmbeddingModel(
                self.model_catalog["multilingual"]
            )
        else:
            # Add models based on performance needs
            performance_level = task_requirements.get("performance_level", "balanced")
            models_to_evaluate[performance_level] = AdvancedEmbeddingModel(
                self.model_catalog[performance_level]
            )
            
            # For comprehensive evaluation, add multiple models
            if task_requirements.get("comprehensive_evaluation", False):
                for name, model_name in self.model_catalog.items():
                    if name != "multilingual" or task_requirements.get("multilingual", False):
                        models_to_evaluate[name] = AdvancedEmbeddingModel(model_name)
        
        # Evaluate models
        evaluation_results = self.evaluator.compare_models(models_to_evaluate, evaluation_data)
        
        # Rank models based on requirements
        model_scores = {}
        for model_name, results in evaluation_results.items():
            score = 0
            
            # Apply weights based on task requirements
            similarity_weight = task_requirements.get("similarity_weight", 0.5)
            clustering_weight = task_requirements.get("clustering_weight", 0.3)
            classification_weight = task_requirements.get("classification_weight", 0.2)
            
            if "similarity" in results:
                sim_score = results["similarity"].get("Pearson_Correlation", 0)
                score += similarity_weight * sim_score
            
            if "clustering" in results:
                cluster_score = results["clustering"].get("Silhouette_Score", 0)
                score += clustering_weight * cluster_score
                
            if "classification" in results:
                class_score = results["classification"].get("Cross_Val_Accuracy_Mean", 0)
                score += classification_weight * class_score
            
            model_scores[model_name] = {
                "overall_score": score,
                "detailed_results": results
            }
        
        # Find best model
        best_model = max(model_scores.items(), key=lambda x: x[1]["overall_score"])
        
        return {
            "recommended_model": best_model[0],
            "score": best_model[1]["overall_score"],
            "all_model_scores": model_scores,
            "task_requirements": task_requirements
        }
```

## **6. Production Deployment Considerations**

### **Scalable Embedding Service**

```python
import asyncio
import aiohttp
from typing import AsyncGenerator
import redis
import pickle
from contextlib import asynccontextmanager


class ScalableEmbeddingService:
    """Production-ready scalable embedding service"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 max_workers: int = 8):
        self.model_name = model_name
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.max_workers = max_workers
        
        # Initialize model
        self.model = AdvancedEmbeddingModel(model_name)
        
        # Request queue for batch processing
        self.request_queue = asyncio.Queue()
        self.results = {}
    
    async def encode_async(self, texts: List[str], request_id: str) -> List[EmbeddingResult]:
        """Asynchronous encoding with caching and batching"""
        # Check cache first
        uncached_texts = []
        cached_results = {}
        
        for i, text in enumerate(texts):
            cache_key = f"embed:{self.model_name}:{hash(text)}"
            cached = self.redis_client.get(cache_key)
            
            if cached:
                cached_emb = pickle.loads(cached)
                cached_results[i] = EmbeddingResult(
                    vector=cached_emb,
                    tokens_used=len(self.model.tokenizer.encode(text)),
                    processing_time=0.0,
                    model_name=self.model_name,
                    text=text
                )
            else:
                uncached_texts.append((i, text))
        
        # Process uncached texts
        uncached_results = {}
        if uncached_texts:
            uncached_batch = [text for i, text in uncached_texts]
            uncached_embeddings = self.model.encode(uncached_batch)
            
            # Cache results
            for j, (i, text) in enumerate(uncached_texts):
                result = uncached_embeddings[j]
                uncached_results[i] = result
                
                # Cache the embedding
                cache_key = f"embed:{self.model_name}:{hash(text)}"
                self.redis_client.setex(
                    cache_key, 
                    3600,  # Cache for 1 hour
                    pickle.dumps(result.vector)
                )
        
        # Combine results in original order
        all_results = [None] * len(texts)
        
        for i, cached_result in cached_results.items():
            all_results[i] = cached_result
            
        for i, uncached_result in uncached_results.items():
            all_results[i] = uncached_result
        
        return all_results
    
    async def batch_encode_async(self, 
                                texts: List[str], 
                                batch_size: int = 32) -> List[EmbeddingResult]:
        """Batch encoding for large lists of texts"""
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await self.encode_async(batch, f"batch_{i}")
            all_results.extend(batch_results)
        
        return all_results
    
    async def stream_encode(self, 
                           text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[EmbeddingResult, None]:
        """Stream encoding for continuous text processing"""
        batch = []
        batch_size = 10  # Adjust based on requirements
        
        async for text in text_stream:
            batch.append(text)
            
            if len(batch) >= batch_size:
                results = await self.encode_async(batch, f"stream_{hash(str(batch))}")
                for result in results:
                    yield result
                batch = []
        
        # Process remaining items
        if batch:
            results = await self.encode_async(batch, f"stream_remaining_{hash(str(batch))}")
            for result in results:
                yield result


class EmbeddingModelManager:
    """Manages multiple embedding models with load balancing"""
    
    def __init__(self):
        self.models = {}
        self.model_loads = {}
        self.model_performance = {}
    
    def register_model(self, 
                      name: str, 
                      model: AdvancedEmbeddingModel,
                      weight: float = 1.0) -> None:
        """Register a new embedding model"""
        self.models[name] = model
        self.model_loads[name] = 0
        self.model_performance[name] = {"usage_count": 0, "avg_response_time": 0.0}
    
    def get_model(self, name: str) -> AdvancedEmbeddingModel:
        """Get a registered model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")
        return self.models[name]
    
    def select_model_with_load_balancing(self) -> str:
        """Select model based on load balancing"""
        # Simple round-robin selection
        available_models = list(self.models.keys())
        if not available_models:
            raise ValueError("No models registered")
        
        # Return least loaded model
        min_load_model = min(available_models, key=lambda m: self.model_loads[m])
        self.model_loads[min_load_model] += 1
        self.model_performance[min_load_model]["usage_count"] += 1
        return min_load_model
    
    def encode_with_optimal_model(self, 
                                 texts: Union[str, List[str]],
                                 strategy: str = "load_balance") -> List[EmbeddingResult]:
        """Encode using optimally selected model"""
        if strategy == "load_balance":
            model_name = self.select_model_with_load_balancing()
        elif strategy == "first_registered":
            model_name = next(iter(self.models.keys()))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        start_time = time.time()
        model = self.get_model(model_name)
        results = model.encode(texts)
        response_time = time.time() - start_time
        
        # Update performance metrics
        perf = self.model_performance[model_name]
        perf["avg_response_time"] = (
            (perf["avg_response_time"] * (perf["usage_count"] - 1) + response_time) / 
            perf["usage_count"]
        )
        
        return results, model_name


class EmbeddingAPI:
    """REST API wrapper for embedding models"""
    
    def __init__(self, model_manager: EmbeddingModelManager):
        self.model_manager = model_manager
    
    async def encode_endpoint(self, request_data: Dict) -> Dict:
        """API endpoint for encoding text to embeddings"""
        texts = request_data.get("texts", [])
        model_name = request_data.get("model", "default")
        normalize = request_data.get("normalize", True)
        
        if isinstance(texts, str):
            texts = [texts]
        
        if model_name == "default":
            # Use load balancing
            results, selected_model = self.model_manager.encode_with_optimal_model(texts)
        else:
            # Use specific model
            model = self.model_manager.get_model(model_name)
            results = model.encode(texts)
            selected_model = model_name
        
        # Format response
        response = {
            "embeddings": [result.vector.tolist() for result in results],
            "model_used": selected_model,
            "text_count": len(texts),
            "tokens_used": sum(result.tokens_used for result in results),
            "processing_times": [result.processing_time for result in results]
        }
        
        return response
    
    async def similarity_endpoint(self, request_data: Dict) -> Dict:
        """API endpoint for calculating similarity between embeddings"""
        emb1 = np.array(request_data["embedding1"])
        emb2 = np.array(request_data["embedding2"])
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return {
            "similarity": float(similarity),
            "method": "cosine"
        }


# Configuration and factory for production deployment
class EmbeddingServiceFactory:
    """Factory for creating production embedding services"""
    
    @staticmethod
    def create_optimized_service(config: Dict) -> ScalableEmbeddingService:
        """Create an optimized embedding service based on configuration"""
        
        service = ScalableEmbeddingService(
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
            redis_host=config.get("redis_host", "localhost"),
            redis_port=config.get("redis_port", 6379),
            max_workers=config.get("max_workers", 8)
        )
        
        return service
    
    @staticmethod
    def create_multi_model_service(model_configs: List[Dict]) -> EmbeddingModelManager:
        """Create a service managing multiple embedding models"""
        
        manager = EmbeddingModelManager()
        
        for config in model_configs:
            model = AdvancedEmbeddingModel(config["name"])
            manager.register_model(config["identifier"], model, config.get("weight", 1.0))
        
        return manager
```

## **7. Testing and Validation**

### **Unit Tests for Embedding Models Module**

```python
import unittest
from unittest.mock import Mock, patch


class TestEmbeddingModelsModule(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = AdvancedEmbeddingModel("all-MiniLM-L6-v2")
        self.sample_texts = [
            "This is the first sentence.",
            "This is another sentence for testing.",
            "A completely different topic altogether."
        ]
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.tokenizer)
        self.assertEqual(self.model.model_name, "all-MiniLM-L6-v2")
    
    def test_single_text_encoding(self):
        """Test encoding of a single text"""
        result = self.model.encode("Test sentence")
        
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0].vector, np.ndarray)
        self.assertGreater(len(result[0].vector), 0)
        self.assertEqual(result[0].text, "Test sentence")
    
    def test_multiple_text_encoding(self):
        """Test encoding of multiple texts"""
        results = self.model.encode(self.sample_texts)
        
        self.assertEqual(len(results), len(self.sample_texts))
        for result in results:
            self.assertIsInstance(result.vector, np.ndarray)
            self.assertGreater(len(result.vector), 0)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between embeddings"""
        results = self.model.encode(self.sample_texts)
        
        emb1 = results[0].vector
        emb2 = results[1].vector
        
        similarity_matrix = self.model.similarity([emb1], [emb2])
        self.assertEqual(similarity_matrix.shape, (1, 1))
        self.assertGreaterEqual(similarity_matrix[0, 0], -1.0)
        self.assertLessEqual(similarity_matrix[0, 0], 1.0)
    
    def test_embedding_dimension(self):
        """Test that embedding dimension is correct"""
        dim = self.model.get_embedding_dimension()
        self.assertGreater(dim, 0)
        
        # Test with a known model dimension
        # all-MiniLM-L6-v2 should have 384 dimensions
        self.assertEqual(dim, 384)  # This might vary by model
    
    def test_ensemble_encoding(self):
        """Test multi-model ensemble"""
        ensemble = MultiModelEmbeddingEnsemble([
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L3-v2"
        ])
        
        results = ensemble.encode(self.sample_texts, aggregation_method="weighted_average")
        
        self.assertEqual(len(results), len(self.sample_texts))
        for result in results:
            self.assertIsInstance(result.vector, np.ndarray)
            # Combined dimension should be same as single model for weighted average
            self.assertGreater(len(result.vector), 0)


class TestDomainSpecificEmbedders(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for domain-specific embedders."""
        base_model = AdvancedEmbeddingModel("all-MiniLM-L6-v2")
        self.health_embedder = HealthcareEmbedder(base_model)
        self.legal_embedder = LegalDocumentEmbedder(base_model)
        self.code_embedder = CodeEmbedder(base_model)
    
    def test_healthcare_preprocessing(self):
        """Test healthcare-specific preprocessing"""
        original_text = "Pt. has HTN and DM."
        processed = self.health_embedder._medical_preprocessor(original_text)
        
        # Should expand abbreviations
        self.assertIn("patient", processed.lower())
        self.assertIn("hypertension", processed.lower())
        self.assertIn("diabetes", processed.lower())
    
    def test_legal_preprocessing(self):
        """Test legal-specific preprocessing"""
        original_text = "Case: Smith v. Jones"
        processed = self.legal_embedder._legal_preprocessor(original_text)
        
        self.assertIn("versus", processed.lower())
    
    def test_code_preprocessing(self):
        """Test code-specific preprocessing"""
        original_code = "def hello_world(): # This is a comment\n    return 'Hello'"
        processed = self.code_embedder._code_preprocessor(original_code)
        
        # Comment should be removed
        self.assertNotIn("comment", processed.lower())
    
    def test_healthcare_encoding(self):
        """Test healthcare-specific encoding"""
        result = self.health_embedder.encode("Patient has hypertension and diabetes")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0].vector, np.ndarray)
    
    def test_medical_similarity_calculation(self):
        """Test medical similarity calculation"""
        similarity = self.health_embedder.calculate_medical_similarity(
            "patient has high blood pressure", 
            "individual diagnosed with hypertension"
        )
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


class TestOptimizedEmbedders(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for optimized embedders."""
        self.optimized_model = OptimizedEmbeddingModel(
            "all-MiniLM-L6-v2", 
            use_cache=True, 
            cache_size=100
        )
    
    def test_caching_functionality(self):
        """Test that caching works correctly"""
        # Encode the same text twice
        result1 = self.optimized_model.encode("test text for caching")
        result2 = self.optimized_model.encode("test text for caching")
        
        # Both should return valid results
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        
        # Results should be very similar (if not identical due to caching)
        similarity = cosine_similarity(
            [result1[0].vector], 
            [result2[0].vector]
        )[0][0]
        
        self.assertGreaterEqual(similarity, 0.99)  # Very high similarity if cached
    
    def test_batch_encoding(self):
        """Test batch encoding functionality"""
        texts = ["text 1", "text 2", "text 3"]
        results = self.optimized_model.encode(texts)
        
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIsInstance(result.vector, np.ndarray)


def run_tests():
    """Run all tests in the module"""
    print("Running tests for Module 3: Embedding Models")
    print("=" * 50)
    
    # Create test suites
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestEmbeddingModelsModule)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestDomainSpecificEmbedders)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestOptimizedEmbedders)
    
    # Combine suites
    full_suite = unittest.TestSuite([suite1, suite2, suite3])
    
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

### **Build a Custom Embedding Model Selector**

```python
# Exercise: Create an intelligent embedding model selector
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


# Example usage and testing
def main():
    """Example usage of the Intelligent Embedding Selector"""
    print("RAG Course - Module 3: Embedding Models Example")
    print("=" * 55)
    
    # Create the intelligent selector
    selector = IntelligentEmbeddingSelector()
    
    # Test with different types of text
    test_texts = [
        "This is a simple English sentence for testing.",
        "Le Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "这是一个中文句子用于测试。",  # Chinese text
        "The quick brown fox jumps over the lazy dog. This is a longer text with more content to analyze for better model selection.",
        "def calculate_similarity(text1, text2): # Function to calculate similarity between two texts\n    return cosine_similarity(text1, text2)"
    ]
    
    print("Testing intelligent model selection:")
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: '{text[:50]}...'")
        
        # Select model based on default requirements
        best_model = selector.select_best_model(text)
        print(f"  Recommended model: {best_model}")
        
        # Analyze text characteristics
        characteristics = selector.analyze_text(text)
        print(f"  Text characteristics: {dict(list(characteristics.items())[:3])}...")  # Show first 3
        
        # Encode with intelligent selection
        try:
            result = selector.encode_with_intelligence(text)
            print(f"  Embedding dimensions: {len(result.vector)}")
            print(f"  Model used: {result.model_name}")
        except Exception as e:
            print(f"  Error during encoding: {str(e)}")
    
    # Test with specific requirements
    print(f"\nTesting with specific requirements:")
    requirements = {
        "speed_importance": 0.8,
        "accuracy_importance": 0.2,
        "multilingual_importance": 0.5
    }
    
    multilingual_text = "Hello world. Hola mundo. Bonjour le monde."
    best_model = selector.select_best_model(multilingual_text, requirements)
    print(f"Multilingual text: '{multilingual_text}'")
    print(f"Recommended model with speed requirement: {best_model}")


if __name__ == "__main__":
    main()
```

This deep dive into embedding models provides comprehensive coverage from theoretical foundations to production-ready implementations. The module includes:

- **Advanced theoretical concepts** with mathematical foundations
- **Multiple embedding model implementations** with different architectures
- **Model selection and evaluation frameworks** for choosing optimal models
- **Domain-specific embeddings** for healthcare, legal, and code
- **Performance optimization** techniques for efficient computation
- **Evaluation methodologies** for measuring embedding quality
- **Production deployment** considerations with scalable services
- **Testing strategies** for validation
- **Hands-on exercises** for practical learning