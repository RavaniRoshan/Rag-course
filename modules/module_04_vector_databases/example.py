"""
Module 4: Vector Databases
Implementation Examples

This module demonstrates different vector database solutions including Chroma, FAISS,
and their usage patterns in RAG systems.
"""

import numpy as np
import time
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import faiss
import pickle
import os


class VectorDBBase:
    """Base class for vector database implementations"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_embeddings(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[str]:
        """Add embeddings to the database"""
        raise NotImplementedError
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        raise NotImplementedError
    
    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Perform batch search"""
        results = []
        for query in queries:
            results.append(self.search(query, k))
        return results


class ChromaDBWrapper(VectorDBBase):
    """Chroma vector database wrapper"""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        super().__init__(384)  # all-MiniLM-L6-v2 dimension
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_embeddings(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[str]:
        """Add embeddings to ChromaDB"""
        # Generate embeddings
        embeddings = self.embedder.encode(texts).tolist()
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Prepare metadata
        if metadata is None:
            # ChromaDB requires non-empty metadata dictionaries or None
            metadatas = None
        else:
            # Ensure metadata is properly formatted
            metadatas = metadata
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings in ChromaDB"""
        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            # Handle potential None metadata
            metadata = results['metadatas'][0][i] if results['metadatas'][0][i] is not None else {}
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': metadata,
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_count(self) -> int:
        """Get the total number of embeddings in the collection"""
        return self.collection.count()


class FAISSWrapper(VectorDBBase):
    """FAISS vector database wrapper"""
    
    def __init__(self, dimension: int = 384, index_type: str = "Flat"):
        super().__init__(dimension)
        self.index_type = index_type
        self.index = None
        self.texts = []
        self.metadata = []
        self.ids = []
        
        # Create FAISS index based on type
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (Cosine similarity with normalized vectors)
        elif index_type == "IVF":
            # More complex index with quantizer - will be reinitialized with proper nlist during training
            quantizer = faiss.IndexFlatIP(dimension)
            nlist = 1  # Placeholder, will be set properly during training
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.quantizer = quantizer
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
            
    def add_embeddings(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[str]:
        """Add embeddings to FAISS"""
        # Generate embeddings
        embeddings = self.embedder.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Store texts and metadata
        self.texts.extend(texts)
        if metadata is None:
            self.metadata.extend([{}] * len(texts))
        else:
            self.metadata.extend(metadata)
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        self.ids.extend(ids)
        
        # Add embeddings to index
        if self.index_type == "IVF":
            # Train the index if it hasn't been trained
            if not self.index.is_trained:
                # Calculate appropriate number of clusters (nlist)
                # Rule of thumb: nlist should be between sqrt(N) and N/10, where N is number of vectors
                n = len(self.texts)
                # Use a minimum of 1 and ensure it's not more than the number of vectors
                nlist = max(1, min(n, int(n ** 0.5)))
                
                # Reinitialize the index with the correct nlist
                quantizer = self.quantizer
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                
                # Train the index
                self.index.train(embeddings.astype('float32'))
        
        self.index.add(embeddings.astype('float32'))
        
        return ids
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings in FAISS"""
        # Generate query embedding
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Perform similarity search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        formatted_results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.texts):  # Valid index check
                formatted_results.append({
                    'id': self.ids[idx],
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'score': float(scores[0][i])  # Cosine similarity score
                })
        
        return formatted_results
    
    def save_index(self, filepath: str):
        """Save the FAISS index and associated data"""
        # Save the FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save texts, metadata, and ids
        data = {
            'texts': self.texts,
            'metadata': self.metadata,
            'ids': self.ids
        }
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load_index(self, filepath: str):
        """Load the FAISS index and associated data"""
        # Load the FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load texts, metadata, and ids
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.texts = data['texts']
        self.metadata = data['metadata']
        self.ids = data['ids']
        
        # Update dimension based on loaded index
        self.dimension = self.index.d


class VectorDBComparison:
    """Class to compare different vector database implementations"""
    
    def __init__(self):
        self.sample_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Natural language processing helps computers understand human language",
            "Vector databases store high-dimensional embeddings for similarity search",
            "Retrieval augmented generation combines retrieval and generation models",
            "Cosine similarity measures the angle between two vectors",
            "Embeddings represent text in high-dimensional vector space",
            "Chroma is an open-source vector database for AI applications",
            "FAISS is Facebook's library for efficient similarity search",
            "Pinecone is a managed vector database service",
            "Weaviate is an open-source vector database with GraphQL API",
            "Semantic search uses meaning rather than keywords for retrieval"
        ]
    
    def benchmark_db(self, db: VectorDBBase, name: str) -> Dict[str, float]:
        """Benchmark a vector database"""
        print(f"Benchmarking {name}...")
        
        # Timing for insertion
        start_time = time.time()
        ids = db.add_embeddings(self.sample_texts)
        insertion_time = time.time() - start_time
        
        # Timing for search
        start_time = time.time()
        results = db.search("artificial intelligence", k=5)
        search_time = time.time() - start_time
        
        # Timing for batch search
        start_time = time.time()
        batch_results = db.batch_search(["machine learning", "natural language processing", "vector search"], k=3)
        batch_search_time = time.time() - start_time
        
        return {
            'insertion_time': insertion_time,
            'search_time': search_time,
            'batch_search_time': batch_search_time,
            'results_count': len(results),
            'name': name
        }
    
    def compare_all_dbs(self) -> List[Dict[str, float]]:
        """Compare all vector database implementations"""
        results = []
        
        # Test ChromaDB
        chroma_db = ChromaDBWrapper(collection_name="comparison_test")
        chroma_results = self.benchmark_db(chroma_db, "ChromaDB")
        results.append(chroma_results)
        
        # Test FAISS Flat
        faiss_flat = FAISSWrapper(dimension=384, index_type="Flat")
        faiss_flat_results = self.benchmark_db(faiss_flat, "FAISS Flat")
        results.append(faiss_flat_results)
        
        # Test FAISS IVF
        faiss_ivf = FAISSWrapper(dimension=384, index_type="IVF")
        faiss_ivf_results = self.benchmark_db(faiss_ivf, "FAISS IVF")
        results.append(faiss_ivf_results)
        
        # Test FAISS HNSW
        faiss_hnsw = FAISSWrapper(dimension=384, index_type="HNSW")
        faiss_hnsw_results = self.benchmark_db(faiss_hnsw, "FAISS HNSW")
        results.append(faiss_hnsw_results)
        
        return results


def advanced_usage_examples():
    """Demonstrate advanced usage patterns"""
    print("=== Advanced Vector Database Usage Examples ===\n")
    
    # Example 1: ChromaDB with metadata filtering
    print("1. ChromaDB with metadata filtering:")
    chroma_db = ChromaDBWrapper(collection_name="advanced_test")
    
    # Add documents with metadata
    texts = [
        "Python is a high-level programming language",
        "JavaScript is primarily used for web development",
        "Rust is known for its memory safety",
        "Go is designed for simplicity and efficiency"
    ]
    
    metadata = [
        {"category": "language", "level": "high", "use_case": "general"},
        {"category": "language", "level": "high", "use_case": "web"},
        {"category": "language", "level": "system", "use_case": "systems"},
        {"category": "language", "level": "high", "use_case": "backend"}
    ]
    
    chroma_db.add_embeddings(texts, metadata)
    
    # Search with metadata filtering
    results = chroma_db.collection.query(
        query_texts=["programming"],
        n_results=3,
        where={"use_case": "web"}  # Metadata filter
    )
    
    print(f"Found {len(results['documents'][0])} documents with use_case='web'")
    for doc in results['documents'][0]:
        print(f"  - {doc}")
    print()
    
    # Example 2: FAISS with different distance metrics
    print("2. FAISS with different indexing strategies:")
    faiss_flat = FAISSWrapper(index_type="Flat")
    faiss_hnsw = FAISSWrapper(index_type="HNSW")
    
    # Add the same data to both
    sample_texts = [
        "The weather is beautiful today",
        "I love to go hiking in the mountains",
        "Machine learning algorithms are powerful",
        "Python has great libraries for data science"
    ]
    
    faiss_flat.add_embeddings(sample_texts)
    faiss_hnsw.add_embeddings(sample_texts)
    
    # Compare search results
    query = "outdoor activities"
    flat_results = faiss_flat.search(query, k=2)
    hnsw_results = faiss_hnsw.search(query, k=2)
    
    print(f"Query: '{query}'")
    print(f"FAISS Flat results: {len(flat_results)} results")
    for result in flat_results:
        print(f"  - {result['text']} (score: {result['score']:.3f})")
    
    print(f"FAISS HNSW results: {len(hnsw_results)} results")
    for result in hnsw_results:
        print(f"  - {result['text']} (score: {result['score']:.3f})")
    print()


def performance_optimization_tips():
    """Show performance optimization techniques"""
    print("=== Performance Optimization Tips ===\n")
    
    print("1. Index Selection:")
    print("   - Flat: Exact search, slower but accurate")
    print("   - IVF: Approximate search, good balance of speed/accuracy")
    print("   - HNSW: Fast search, good for high-dimensional data")
    print()
    
    print("2. Parameter Tuning:")
    print("   - FAISS IVF: Adjust nlist (number of clusters) based on data size")
    print("   - FAISS HNSW: Adjust M (max connections) and ef parameters")
    print("   - Use nprobe for IVF to balance speed/accuracy")
    print()
    
    print("3. Memory Management:")
    print("   - Use quantization to reduce memory usage")
    print("   - Consider IVF with quantization for large datasets")
    print("   - Monitor memory usage during operations")
    print()


def main():
    """Main function to demonstrate vector database implementations"""
    print("Module 4: Vector Databases")
    print("=" * 50)
    
    # Run comparison
    comparison = VectorDBComparison()
    results = comparison.compare_all_dbs()
    
    print("\n=== Performance Comparison ===")
    print(f"{'Database':<15} {'Insert(ms)':<12} {'Search(ms)':<12} {'Batch(ms)':<12} {'Results':<8}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['name']:<15} {result['insertion_time']*1000:<12.2f} "
              f"{result['search_time']*1000:<12.2f} {result['batch_search_time']*1000:<12.2f} "
              f"{result['results_count']:<8}")
    
    print()
    
    # Show advanced usage
    advanced_usage_examples()
    
    # Show optimization tips
    performance_optimization_tips()


if __name__ == "__main__":
    main()