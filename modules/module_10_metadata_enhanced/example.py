"""
Module 10: Metadata-Enhanced Retrieval
Implementation Examples

This module demonstrates metadata-enhanced retrieval techniques
to improve document retrieval accuracy and relevance in RAG systems.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pandas as pd
import json
import time
import uuid
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import re
import hashlib


class MetadataDocument:
    """Class to represent a document with metadata"""
    
    def __init__(self, content: str, doc_id: str = None, metadata: Dict[str, Any] = None):
        self.doc_id = doc_id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.metadata['id'] = self.doc_id
        self.metadata['content_length'] = len(content)
        self.metadata['created_at'] = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format"""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata
        }


class MetadataIndexer:
    """Class to handle indexing documents with metadata"""
    
    def __init__(self, collection_name: str = "metadata_rag"):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Create collection with metadata support
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, doc: MetadataDocument) -> str:
        """Add a document with metadata to the index"""
        # Generate embedding for content
        embedding = self.embedder.encode([doc.content]).tolist()[0]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[doc.content],
            metadatas=[doc.metadata],
            ids=[doc.doc_id]
        )
        
        return doc.doc_id
    
    def add_documents(self, docs: List[MetadataDocument]) -> List[str]:
        """Add multiple documents with metadata to the index"""
        if not docs:
            return []
        
        # Generate embeddings for all content
        contents = [doc.content for doc in docs]
        embeddings = self.embedder.encode(contents).tolist()
        
        # Prepare metadata and IDs
        metadatas = [doc.metadata for doc in docs]
        ids = [doc.doc_id for doc in docs]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def get_count(self) -> int:
        """Get the total number of documents in the collection"""
        return self.collection.count()


class MetadataFilter:
    """Class to handle metadata-based filtering"""
    
    def __init__(self):
        pass
    
    def build_filter_condition(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build a filter condition for ChromaDB from filters dict"""
        if not filters:
            return {}
        
        # ChromaDB uses a specific filter format
        return filters
    
    def apply_exact_filters(self, docs: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply exact match filters to documents"""
        if not filters:
            return docs
        
        filtered_docs = []
        for doc in docs:
            include = True
            metadata = doc.get('metadata', {})
            
            for key, value in filters.items():
                if isinstance(value, list):
                    # Check if metadata value is in the list
                    if metadata.get(key) not in value:
                        include = False
                        break
                else:
                    # Exact match
                    if metadata.get(key) != value:
                        include = False
                        break
            
            if include:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def apply_range_filters(self, docs: List[Dict[str, Any]], range_filters: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Apply range filters to documents"""
        if not range_filters:
            return docs
        
        filtered_docs = []
        for doc in docs:
            include = True
            metadata = doc.get('metadata', {})
            
            for field, range_values in range_filters.items():
                doc_value = metadata.get(field)
                if doc_value is None:
                    include = False
                    break
                
                # Check if value is within range
                min_val = range_values.get('min')
                max_val = range_values.get('max')
                
                if min_val is not None and doc_value < min_val:
                    include = False
                    break
                if max_val is not None and doc_value > max_val:
                    include = False
                    break
            
            if include:
                filtered_docs.append(doc)
        
        return filtered_docs


class MetadataScorer:
    """Class to handle metadata-aware scoring"""
    
    def __init__(self):
        self.default_weights = {
            'metadata_relevance': 0.3,
            'content_similarity': 0.7
        }
    
    def calculate_metadata_score(self, query: str, metadata: Dict[str, Any]) -> float:
        """Calculate a score based on how well metadata matches the query"""
        score = 0.0
        
        # Check for matches in various metadata fields
        for key, value in metadata.items():
            if isinstance(value, str):
                # Simple string matching (could be enhanced with embedding similarity)
                if query.lower() in value.lower():
                    score += 0.3  # Boost for metadata matches
                elif any(term in value.lower() for term in query.lower().split()):
                    score += 0.1  # Partial match
        
        # Normalize score
        return min(score, 1.0)
    
    def combine_scores(self, content_score: float, metadata_score: float, 
                      weights: Dict[str, float] = None) -> float:
        """Combine content and metadata scores"""
        if weights is None:
            weights = self.default_weights
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            return content_score
        
        # Normalize weights
        norm_metadata_weight = weights.get('metadata_relevance', 0.3) / total_weight
        norm_content_weight = weights.get('content_similarity', 0.7) / total_weight
        
        combined_score = (norm_content_weight * content_score + 
                         norm_metadata_weight * metadata_score)
        
        return combined_score


class MetadataRetriever:
    """Main class for metadata-enhanced retrieval"""
    
    def __init__(self, indexer: MetadataIndexer, filter_engine: MetadataFilter = None, 
                 scorer: MetadataScorer = None):
        self.indexer = indexer
        self.filter_engine = filter_engine or MetadataFilter()
        self.scorer = scorer or MetadataScorer()
    
    def retrieve(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None, 
                range_filters: Dict[str, Dict[str, float]] = None, 
                metadata_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieve documents with metadata enhancement"""
        # Perform initial semantic search
        results = self.indexer.collection.query(
            query_texts=[query],
            n_results=top_k * 3,  # Get more results before applying metadata filters
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            doc = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1.0 - results['distances'][0][i]  # Convert distance to similarity
            }
            formatted_results.append(doc)
        
        # Apply filters if specified
        if filters:
            formatted_results = self.filter_engine.apply_exact_filters(formatted_results, filters)
        
        if range_filters:
            formatted_results = self.filter_engine.apply_range_filters(formatted_results, range_filters)
        
        # Recalculate scores with metadata enhancement
        enhanced_results = []
        for doc in formatted_results:
            metadata_score = self.scorer.calculate_metadata_score(query, doc['metadata'])
            content_similarity = doc['similarity']
            
            # Combine scores with specified weight
            weights = {
                'metadata_relevance': metadata_weight,
                'content_similarity': 1.0 - metadata_weight
            }
            enhanced_score = self.scorer.combine_scores(
                content_similarity, metadata_score, weights
            )
            
            doc['enhanced_score'] = enhanced_score
            doc['metadata_score'] = metadata_score
            enhanced_results.append(doc)
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
        
        # Return top_k results
        return enhanced_results[:top_k]
    
    def retrieve_with_intelligent_filtering(self, query: str, top_k: int = 5, 
                                          query_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using intelligent query-based metadata filtering"""
        # Extract potential metadata requirements from the query
        extracted_filters = self._extract_metadata_requirements(query)
        
        # Combine with any provided filters
        if query_filters:
            extracted_filters.update(query_filters)
        
        # Perform retrieval with filters
        return self.retrieve(query, top_k, filters=extracted_filters)
    
    def _extract_metadata_requirements(self, query: str) -> Dict[str, Any]:
        """Extract metadata requirements from query text"""
        filters = {}
        
        # Look for common patterns in the query
        query_lower = query.lower()
        
        # Check for temporal requirements
        if 'recent' in query_lower or 'latest' in query_lower or 'new' in query_lower:
            # Add filter for recent documents (last 6 months)
            six_months_ago = datetime.now() - timedelta(days=180)
            filters['created_at'] = {'$gte': six_months_ago.isoformat()}
        
        # Check for source requirements
        if 'wikipedia' in query_lower:
            filters['source'] = 'wikipedia'
        elif 'arxiv' in query_lower:
            filters['source'] = 'arxiv'
        elif 'blog' in query_lower:
            filters['source_type'] = 'blog'
        
        # Check for quality requirements
        if 'high quality' in query_lower or 'reliable' in query_lower or 'authoritative' in query_lower:
            filters['quality_score'] = {'$gte': 0.8}
        
        # Check for format requirements
        if 'tutorial' in query_lower or 'guide' in query_lower:
            filters['content_type'] = 'tutorial'
        
        return filters


class MetadataRetrievalEvaluator:
    """Evaluator for metadata-enhanced retrieval"""
    
    def __init__(self):
        self.sample_docs = [
            MetadataDocument(
                "Machine learning is a subset of artificial intelligence that focuses on algorithms",
                metadata={
                    'author': 'Dr. Smith',
                    'source': 'journal',
                    'category': 'AI',
                    'quality_score': 0.9,
                    'content_type': 'research_paper',
                    'year': 2023
                }
            ),
            MetadataDocument(
                "Deep learning neural networks have multiple layers that process information",
                metadata={
                    'author': 'Dr. Johnson',
                    'source': 'arxiv',
                    'category': 'ML',
                    'quality_score': 0.8,
                    'content_type': 'research_paper',
                    'year': 2022
                }
            ),
            MetadataDocument(
                "Natural language processing helps computers understand human language",
                metadata={
                    'author': 'Dr. Williams',
                    'source': 'blog',
                    'category': 'NLP',
                    'quality_score': 0.7,
                    'content_type': 'tutorial',
                    'year': 2023
                }
            ),
            MetadataDocument(
                "Data science combines statistics, programming, and domain expertise",
                metadata={
                    'author': 'Dr. Brown',
                    'source': 'journal',
                    'category': 'Data Science',
                    'quality_score': 0.95,
                    'content_type': 'research_paper',
                    'year': 2021
                }
            ),
            MetadataDocument(
                "Vector databases optimize similarity search for high-dimensional embeddings",
                metadata={
                    'author': 'Dr. Davis',
                    'source': 'blog',
                    'category': 'Databases',
                    'quality_score': 0.85,
                    'content_type': 'tutorial',
                    'year': 2023
                }
            )
        ]
    
    def evaluate_retrieval_methods(self) -> Dict[str, Any]:
        """Evaluate different retrieval methods"""
        # Initialize indexer and populate with sample docs
        indexer = MetadataIndexer()
        indexer.add_documents(self.sample_docs)
        
        # Initialize retriever
        retriever = MetadataRetriever(indexer)
        
        # Test queries
        queries = [
            "machine learning algorithms",
            "recent AI research papers",
            "high quality NLP tutorials"
        ]
        
        results = {
            'basic_retrieval': [],
            'metadata_enhanced': [],
            'intelligent_filtering': []
        }
        
        for query in queries:
            # Basic retrieval
            basic_results = retriever.retrieve(query, top_k=3)
            results['basic_retrieval'].append({
                'query': query,
                'results': basic_results
            })
            
            # Metadata-enhanced retrieval
            if 'recent' in query.lower():
                metadata_results = retriever.retrieve(
                    query, top_k=3, 
                    range_filters={'year': {'min': 2022}}
                )
            elif 'high quality' in query.lower():
                metadata_results = retriever.retrieve(
                    query, top_k=3,
                    range_filters={'quality_score': {'min': 0.8}}
                )
            else:
                metadata_results = retriever.retrieve(query, top_k=3, metadata_weight=0.4)
            
            results['metadata_enhanced'].append({
                'query': query,
                'results': metadata_results
            })
            
            # Intelligent filtering
            intelligent_results = retriever.retrieve_with_intelligent_filtering(query, top_k=3)
            results['intelligent_filtering'].append({
                'query': query,
                'results': intelligent_results
            })
        
        return results
    
    def compare_performance(self) -> Dict[str, float]:
        """Compare performance of different approaches"""
        indexer = MetadataIndexer()
        indexer.add_documents(self.sample_docs * 10)  # Multiply to have more docs
        
        retriever = MetadataRetriever(indexer)
        
        # Time the different approaches
        query = "machine learning concepts"
        
        # Basic approach
        start_time = time.time()
        for _ in range(5):  # Run multiple times to average
            basic_results = retriever.retrieve(query, top_k=3)
        basic_time = (time.time() - start_time) / 5
        
        # Metadata-enhanced approach
        start_time = time.time()
        for _ in range(5):
            metadata_results = retriever.retrieve(query, top_k=3, metadata_weight=0.3)
        metadata_time = (time.time() - start_time) / 5
        
        return {
            'basic_avg_time': basic_time,
            'metadata_enhanced_avg_time': metadata_time,
            'overhead': metadata_time - basic_time
        }


def demonstrate_metadata_enhanced_retrieval():
    """Demonstrate metadata-enhanced retrieval techniques"""
    print("=== Metadata-Enhanced Retrieval Demonstration ===\n")
    
    # Create sample documents with metadata
    sample_docs = [
        MetadataDocument(
            "Advanced machine learning techniques for predictive modeling",
            metadata={
                'author': 'Dr. Smith',
                'source': 'journal',
                'category': 'Machine Learning',
                'quality_score': 0.92,
                'content_type': 'research_paper',
                'year': 2023,
                'access_level': 'public'
            }
        ),
        MetadataDocument(
            "Deep learning neural networks explained in simple terms",
            metadata={
                'author': 'Dr. Johnson',
                'source': 'arxiv',
                'category': 'Deep Learning',
                'quality_score': 0.87,
                'content_type': 'research_paper',
                'year': 2022,
                'access_level': 'public'
            }
        ),
        MetadataDocument(
            "A beginner's guide to natural language processing",
            metadata={
                'author': 'Dr. Williams',
                'source': 'blog',
                'category': 'NLP',
                'quality_score': 0.75,
                'content_type': 'tutorial',
                'year': 2023,
                'access_level': 'public'
            }
        ),
        MetadataDocument(
            "Production-ready vector databases: A comprehensive comparison",
            metadata={
                'author': 'Dr. Brown',
                'source': 'blog',
                'category': 'Databases',
                'quality_score': 0.88,
                'content_type': 'tutorial',
                'year': 2023,
                'access_level': 'public'
            }
        ),
        MetadataDocument(
            "Data science methodologies and best practices",
            metadata={
                'author': 'Dr. Davis',
                'source': 'journal',
                'category': 'Data Science',
                'quality_score': 0.95,
                'content_type': 'research_paper',
                'year': 2021,
                'access_level': 'public'
            }
        )
    ]
    
    print(f"Created {len(sample_docs)} sample documents with metadata\n")
    
    # Initialize indexer and add documents
    indexer = MetadataIndexer()
    doc_ids = indexer.add_documents(sample_docs)
    print(f"Added documents to index with IDs: {doc_ids[:3]}...\n")
    
    # Initialize retriever
    retriever = MetadataRetriever(indexer)
    
    # Example 1: Basic retrieval
    print("1. Basic retrieval for 'machine learning':")
    basic_results = retriever.retrieve("machine learning", top_k=3)
    for i, result in enumerate(basic_results, 1):
        print(f"  {i}. Score: {result['similarity']:.3f} | {result['content'][:80]}...")
        print(f"     Author: {result['metadata'].get('author', 'Unknown')} | "
              f"Year: {result['metadata'].get('year', 'Unknown')} | "
              f"Quality: {result['metadata'].get('quality_score', 'Unknown')}")
    print()
    
    # Example 2: Metadata-enhanced retrieval
    print("2. Metadata-enhanced retrieval for 'NLP' with quality filter:")
    enhanced_results = retriever.retrieve(
        "natural language processing", 
        top_k=3, 
        range_filters={'quality_score': {'min': 0.8}},
        metadata_weight=0.4
    )
    for i, result in enumerate(enhanced_results, 1):
        print(f"  {i}. Enhanced Score: {result['enhanced_score']:.3f} | {result['content'][:80]}...")
        print(f"     Author: {result['metadata'].get('author', 'Unknown')} | "
              f"Quality: {result['metadata'].get('quality_score', 'Unknown')} | "
              f"Type: {result['metadata'].get('content_type', 'Unknown')}")
    print()
    
    # Example 3: Filtering by multiple criteria
    print("3. Retrieval with multiple metadata filters for 'recent ML research':")
    filtered_results = retriever.retrieve(
        "machine learning", 
        top_k=3,
        filters={'category': 'Machine Learning', 'content_type': 'research_paper'},
        range_filters={'year': {'min': 2022}}
    )
    for i, result in enumerate(filtered_results, 1):
        print(f"  {i}. Content Score: {result['similarity']:.3f} | {result['content'][:80]}...")
        print(f"     Author: {result['metadata'].get('author', 'Unknown')} | "
              f"Year: {result['metadata'].get('year', 'Unknown')} | "
              f"Source: {result['metadata'].get('source', 'Unknown')}")
    print()


def performance_comparison():
    """Compare performance of different approaches"""
    print("=== Performance Comparison ===\n")
    
    evaluator = MetadataRetrieverEvaluator()
    perf_results = evaluator.compare_performance()
    
    print(f"Performance Results:")
    print(f"  Basic retrieval avg time: {perf_results['basic_avg_time']:.4f}s")
    print(f"  Metadata-enhanced avg time: {perf_results['metadata_enhanced_avg_time']:.4f}s")
    print(f"  Overhead: {perf_results['overhead']:.4f}s")
    
    print(f"\nMetadata enhancement adds {perf_results['overhead']/perf_results['basic_avg_time']*100:.1f}% overhead")
    print(f"But provides significantly better relevance through metadata awareness")


def main():
    """Main function to demonstrate metadata-enhanced retrieval implementations"""
    print("Module 10: Metadata-Enhanced Retrieval")
    print("=" * 60)
    
    # Demonstrate metadata-enhanced retrieval
    demonstrate_metadata_enhanced_retrieval()
    
    # Show performance comparison
    performance_comparison()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of intelligent query processing
    print("Intelligent query processing example:")
    
    # Create a small index for this example
    indexer = MetadataIndexer()
    
    docs = [
        MetadataDocument(
            "Latest advances in transformer architectures for NLP",
            metadata={
                'author': 'Dr. Lee',
                'source': 'arxiv',
                'category': 'NLP',
                'quality_score': 0.92,
                'content_type': 'research_paper',
                'year': 2023,
                'created_at': (datetime.now() - timedelta(days=30)).isoformat()
            }
        ),
        MetadataDocument(
            "Old techniques in machine learning from 2015",
            metadata={
                'author': 'Dr. Wilson',
                'source': 'journal',
                'category': 'ML',
                'quality_score': 0.68,
                'content_type': 'research_paper',
                'year': 2015,
                'created_at': (datetime.now() - timedelta(days=3000)).isoformat()
            }
        )
    ]
    
    indexer.add_documents(docs)
    retriever = MetadataRetriever(indexer)
    
    # Test intelligent filtering for "recent AI research"
    results = retriever.retrieve_with_intelligent_filtering("recent AI research", top_k=3)
    
    print(f"Query: 'recent AI research'")
    print(f"Detected {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['content'][:100]}...")
        print(f"     Year: {result['metadata'].get('year', 'Unknown')}, "
              f"Quality: {result['metadata'].get('quality_score', 'Unknown')}")
    
    print(f"\nModule 10 completed - Metadata-enhanced retrieval implemented and demonstrated")


if __name__ == "__main__":
    main()