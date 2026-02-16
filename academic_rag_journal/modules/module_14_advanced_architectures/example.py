"""
Module 14: Advanced Architectures
Implementation Examples

This module demonstrates advanced RAG architectures including
hierarchical retrieval, recursive processing, graph-based retrieval,
multi-modal processing, and multi-hop reasoning.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
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
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt


class DocumentNode:
    """Represents a document node in a hierarchical structure"""
    
    def __init__(self, id: str, content: str, level: int = 0, 
                 parent_id: str = None, metadata: Dict[str, Any] = None):
        self.id = id
        self.content = content
        self.level = level
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.children = []
        self.embedding = None
        self.summary = None  # For higher-level nodes


class HierarchicalRAG:
    """Hierarchical RAG system with multi-level document structure"""
    
    def __init__(self, levels: int = 3):
        self.levels = levels
        self.document_tree = {}  # id -> DocumentNode
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize vector stores for each level
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.level_collections = {}
        for level in range(levels):
            self.level_collections[level] = self.client.get_or_create_collection(
                name=f"level_{level}_rag",
                metadata={"hnsw:space": "cosine"}
            )
    
    def build_hierarchical_structure(self, documents: List[str]) -> str:
        """Build hierarchical structure from flat documents"""
        root_id = str(uuid.uuid4())
        root_node = DocumentNode(root_id, "Root", level=0)
        self.document_tree[root_id] = root_node
        
        for doc in documents:
            # Split document into chunks at each level
            self._add_document_recursive(doc, parent_node=root_node, current_level=1)
        
        return root_id
    
    def _add_document_recursive(self, content: str, parent_node: DocumentNode, 
                              current_level: int, max_chunk_size: int = 200):
        """Recursively add document content to hierarchical structure"""
        if current_level >= self.levels:
            # At lowest level, add the content directly
            node_id = str(uuid.uuid4())
            node = DocumentNode(node_id, content, level=current_level, parent_id=parent_node.id)
            node.embedding = self.embedder.encode([content])[0]
            
            # Add to level-specific collection
            self.level_collections[current_level].add(
                embeddings=[node.embedding.tolist()],
                documents=[content],
                metadatas=[node.metadata],
                ids=[node_id]
            )
            
            parent_node.children.append(node_id)
            self.document_tree[node_id] = node
            return
        
        # Split content into chunks for current level
        chunks = self._split_content(content, max_chunk_size)
        
        for chunk in chunks:
            chunk_node_id = str(uuid.uuid4())
            chunk_node = DocumentNode(
                chunk_node_id, chunk, level=current_level, parent_id=parent_node.id
            )
            
            # Add to level collection
            chunk_embedding = self.embedder.encode([chunk])[0]
            chunk_node.embedding = chunk_embedding
            
            self.level_collections[current_level].add(
                embeddings=[chunk_embedding.tolist()],
                documents=[chunk],
                metadatas=[chunk_node.metadata],
                ids=[chunk_node_id]
            )
            
            parent_node.children.append(chunk_node_id)
            self.document_tree[chunk_node_id] = chunk_node
            
            # Recursively process child content if needed
            self._add_document_recursive(
                chunk, parent_node=chunk_node, current_level=current_level + 1
            )
    
    def _split_content(self, content: str, chunk_size: int) -> List[str]:
        """Split content into chunks"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks or [content]  # Return original if it's shorter than chunk_size
    
    def retrieve_hierarchical(self, query: str, top_k_per_level: List[int]) -> Dict[int, List[Dict[str, Any]]]:
        """Retrieve from all levels hierarchically"""
        query_embedding = self.embedder.encode([query])[0]
        results = {}
        
        for level, top_k in enumerate(top_k_per_level):
            if level >= self.levels:
                continue
            
            collection = self.level_collections[level]
            level_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            for i in range(len(level_results['documents'][0])):
                formatted_results.append({
                    'id': level_results['ids'][0][i],
                    'content': level_results['documents'][0][i],
                    'metadata': level_results['metadatas'][0][i],
                    'similarity': 1.0 - level_results['distances'][0][i],
                    'level': level
                })
            
            results[level] = formatted_results
        
        return results
    
    def synthesize_hierarchical_results(self, hierarchical_results: Dict[int, List[Dict]]) -> str:
        """Synthesize results from different levels"""
        all_contents = []
        
        for level, results in sorted(hierarchical_results.items()):
            level_content = f"\nLevel {level} Results:\n"
            for result in results:
                level_content += f"- {result['content'][:100]}...\n"
            all_contents.append(level_content)
        
        return "\n".join(all_contents)


class GraphBasedRAG:
    """Graph-based RAG system using knowledge graphs"""
    
    def __init__(self):
        self.graph = nx.Graph()  # Knowledge graph
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="graph_rag",
            metadata={"hnsw:space": "cosine"}
        )
        self.entity_embeddings = {}  # entity -> embedding
        self.relation_embeddings = {}  # relation -> embedding
    
    def extract_entities_and_relations(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Extract entities and relations from text (simplified for example)"""
        # In a real implementation, you would use NER/RE models
        # Here we'll simulate entity extraction with a simple approach
        
        # Find capitalized words as potential entities
        entities = list(set(re.findall(r'\b[A-Z][a-z]{2,}\b', text)))
        
        # Create some relations between entities
        relations = []
        if len(entities) >= 2:
            for i in range(len(entities)-1):
                relations.append((entities[i], "related_to", entities[i+1]))
        
        return entities, relations
    
    def build_knowledge_graph(self, documents: List[str]):
        """Build knowledge graph from documents"""
        all_entities = []
        all_relations = []
        
        for doc in documents:
            entities, relations = self.extract_entities_and_relations(doc)
            all_entities.extend(entities)
            all_relations.extend(relations)
            
            # Add document to vector store
            doc_embedding = self.embedder.encode([doc])[0]
            doc_id = str(uuid.uuid4())
            
            self.collection.add(
                embeddings=[doc_embedding.tolist()],
                documents=[doc],
                metadatas=[{"id": doc_id, "entities": entities}],
                ids=[doc_id]
            )
        
        # Add entities and relations to graph
        unique_entities = list(set(all_entities))
        for entity in unique_entities:
            self.graph.add_node(entity, type="entity")
            # Create embedding for entity
            entity_embedding = self.embedder.encode([entity])[0]
            self.entity_embeddings[entity] = entity_embedding
        
        for rel in all_relations:
            self.graph.add_edge(rel[0], rel[2], relation=rel[1], type="relation")
            # Create embedding for relation
            rel_text = f"{rel[0]} {rel[1]} {rel[2]}"
            rel_embedding = self.embedder.encode([rel_text])[0]
            self.relation_embeddings[f"{rel[0]}_{rel[1]}_{rel[2]}"] = rel_embedding
    
    def retrieve_via_graph(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using graph-based approach"""
        query_embedding = self.embedder.encode([query])[0]
        
        # First, get relevant documents
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            doc_id = results['ids'][0][i]
            content = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            similarity = 1.0 - results['distances'][0][i]
            
            # Get related entities from the graph
            doc_entities = metadata.get("entities", [])
            related_info = []
            
            for entity in doc_entities:
                if self.graph.has_node(entity):
                    neighbors = list(self.graph.neighbors(entity))
                    related_info.extend(neighbors[:3])  # Limit to 3 neighbors
            
            formatted_results.append({
                'id': doc_id,
                'content': content,
                'metadata': metadata,
                'similarity': similarity,
                'related_entities': list(set(related_info))
            })
        
        return formatted_results
    
    def path_based_retrieval(self, query: str, entity1: str, entity2: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve information based on paths between entities in the graph"""
        try:
            # Find shortest path between entities
            if nx.has_path(self.graph, entity1, entity2):
                path = nx.shortest_path(self.graph, entity1, entity2)
                
                # Find documents that mention entities in the path
                path_entities = set(path)
                results = self.collection.query(
                    query_texts=[f"{entity1} {entity2} relations"],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                formatted_results = []
                for i in range(len(results['documents'][0])):
                    doc_entities = results['metadatas'][0][i].get("entities", [])
                    path_overlap = len(path_entities.intersection(set(doc_entities)))
                    
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'path_overlap': path_overlap,
                        'similarity': 1.0 - results['distances'][0][i]
                    })
                
                # Sort by path overlap and similarity
                formatted_results.sort(
                    key=lambda x: x['path_overlap'] * 0.7 + x['similarity'] * 0.3,
                    reverse=True
                )
                
                return formatted_results
            else:
                # No path exists, return regular retrieval
                return self.retrieve_via_graph(query, top_k)
        except:
            # If path finding fails, fall back to regular retrieval
            return self.retrieve_via_graph(query, top_k)


class RecursiveRAG:
    """Recursive RAG system that refines queries iteratively"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="recursive_rag",
            metadata={"hnsw:space": "cosine"}
        )
        self.max_recursion_depth = 3
        self.min_improvement_threshold = 0.1
    
    def add_documents(self, documents: List[str]):
        """Add documents to the collection"""
        embeddings = self.embedder.encode(documents)
        
        ids = [str(uuid.uuid4()) for _ in documents]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=[{"id": doc_id} for doc_id in ids],
            ids=ids
        )
    
    def retrieve_recursive(self, query: str, context: str = "", depth: int = 0) -> Dict[str, Any]:
        """Recursively retrieve and refine results"""
        if depth >= self.max_recursion_depth:
            # Base case: return final results
            query_embedding = self.embedder.encode([query + " " + context])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1.0 - results['distances'][0][i]
                })
            
            return {
                'query': query,
                'context': context,
                'results': formatted_results,
                'depth': depth,
                'final': True
            }
        
        # Get initial results based on query and context
        query_embedding = self.embedder.encode([query + " " + context])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Analyze results and generate sub-queries if needed
        sub_queries = self._generate_sub_queries(query, results)
        
        if sub_queries and depth < self.max_recursion_depth - 1:
            # Recursively process sub-queries
            refined_context = context + " " + " ".join([r['documents'][0] for r in [results] if r['documents']])
            
            for sub_query in sub_queries[:2]:  # Limit to 2 sub-queries to prevent explosion
                sub_result = self.retrieve_recursive(sub_query, refined_context, depth + 1)
                
                # Check if sub-result is significantly better
                if self._is_significantly_better(sub_result, results):
                    return sub_result
        
        # If no significant improvement, return current results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1.0 - results['distances'][0][i]
            })
        
        return {
            'query': query,
            'context': context,
            'results': formatted_results,
            'depth': depth,
            'final': depth == self.max_recursion_depth - 1
        }
    
    def _generate_sub_queries(self, original_query: str, current_results: Dict) -> List[str]:
        """Generate sub-queries based on current results"""
        # Simple sub-query generation based on entities in results
        # In practice, this would use more sophisticated reasoning
        sub_queries = []
        
        for doc in current_results['documents'][0][:2]:  # Look at top 2 documents
            # Extract potential entities or concepts (simplified)
            words = doc.split()
            potential_entities = [word for word in words if word.istitle() and len(word) > 3]
            
            # Create sub-queries with these entities
            for entity in potential_entities[:3]:  # Limit to 3 entities
                sub_queries.append(f"More about {entity} mentioned in context")
        
        return sub_queries
    
    def _is_significantly_better(self, new_result: Dict, old_results: Dict) -> bool:
        """Check if new result is significantly better than old results"""
        # Simplified comparison - in practice, you'd have more sophisticated measures
        if not new_result.get('results') or not old_results.get('documents'):
            return False
        
        return True  # Placeholder for more sophisticated comparison


class MultiModalRAG:
    """Multi-modal RAG system handling different input types"""
    
    def __init__(self):
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # In a real implementation, you would have image, audio, etc. embedders
        self.text_collection = None
        self.setup_collections()
    
    def setup_collections(self):
        """Set up collections for different modalities"""
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.text_collection = client.get_or_create_collection(
            name="multimodal_text",
            metadata={"hnsw:space": "cosine"}
        )
        # Additional collections for other modalities would go here
    
    def add_text_content(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add text content to the system"""
        if metadata is None:
            metadata = [{}] for _ in texts]
        
        embeddings = self.text_embedder.encode(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        
        self.text_collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
    
    def retrieve_multimodal(self, query: str, modality_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Retrieve using multi-modal approach"""
        if modality_weights is None:
            modality_weights = {"text": 1.0}  # Default to text only
        
        # For now, just do text retrieval (other modalities would be added in a full implementation)
        query_embedding = self.text_embedder.encode([query])[0]
        results = self.text_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1.0 - results['distances'][0][i],
                'modality': 'text'
            })
        
        return {
            'results': formatted_results,
            'query_modality': 'text',
            'modalities_used': ['text']
        }


class MultiHopRAG:
    """Multi-hop RAG system for complex reasoning that requires multiple steps"""
    
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="multihop_rag",
            metadata={"hnsw:space": "cosine"}
        )
        self.max_hops = 3
    
    def add_documents(self, documents: List[str]):
        """Add documents to the collection"""
        embeddings = self.embedder.encode(documents)
        ids = [str(uuid.uuid4()) for _ in documents]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=[{"id": doc_id, "step": 0} for doc_id in ids],
            ids=ids
        )
    
    def retrieve_multihop(self, query: str) -> Dict[str, Any]:
        """Perform multi-hop retrieval for complex queries"""
        hops = []
        current_query = query
        context = ""
        
        for hop in range(self.max_hops):
            # Retrieve based on current query and accumulated context
            query_with_context = current_query + " " + context
            query_embedding = self.embedder.encode([query_with_context])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )
            
            hop_results = []
            for i in range(len(results['documents'][0])):
                hop_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'similarity': 1.0 - results['distances'][0][i]
                })
            
            hops.append({
                'hop': hop + 1,
                'query': current_query,
                'results': hop_results
            })
            
            # Generate next hop query based on current results
            # This is a simplified example - in practice, you'd have more sophisticated logic
            if hop < self.max_hops - 1 and hop_results:
                context += " " + " ".join([r['content'] for r in hop_results])
                # Simplified query generation for next hop
                next_query = self._generate_next_query(current_query, hop_results)
                current_query = next_query
        
        return {
            'original_query': query,
            'hops': hops,
            'final_context': context
        }
    
    def _generate_next_query(self, original_query: str, current_results: List[Dict]) -> str:
        """Generate next hop query based on current results"""
        # Simple approach: identify entities from results and ask about their relationships
        # A real implementation would use more sophisticated reasoning
        first_result = current_results[0]['content']
        words = first_result.split()
        entities = [w for w in words if w.istitle() and len(w) > 3][:2]
        
        if len(entities) >= 2:
            return f"What is the relationship between {entities[0]} and {entities[1]}?"
        else:
            return f"More information about {' '.join(entities[:1]) if entities else 'the topic'}"


class AdvancedRAGFactory:
    """Factory for creating advanced RAG architectures"""
    
    @staticmethod
    def create_hierarchical(levels: int = 3) -> HierarchicalRAG:
        """Create a hierarchical RAG system"""
        return HierarchicalRAG(levels)
    
    @staticmethod
    def create_graph_based() -> GraphBasedRAG:
        """Create a graph-based RAG system"""
        return GraphBasedRAG()
    
    @staticmethod
    def create_recursive() -> RecursiveRAG:
        """Create a recursive RAG system"""
        return RecursiveRAG()
    
    @staticmethod
    def create_multimodal() -> MultiModalRAG:
        """Create a multi-modal RAG system"""
        return MultiModalRAG()
    
    @staticmethod
    def create_multihop() -> MultiHopRAG:
        """Create a multi-hop RAG system"""
        return MultiHopRAG()


def demonstrate_advanced_architectures():
    """Demonstrate advanced RAG architectures"""
    print("=== Advanced Architectures Demonstration ===\n")
    
    # Sample documents for demonstration
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
        "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        "Natural language processing allows computers to understand, interpret, and generate human language.",
        "Computer vision enables machines to identify and analyze visual content in images and videos.",
        "Reinforcement learning uses reward and punishment mechanisms to train agents to make decisions.",
        "Supervised learning requires labeled training data to learn input-output mappings.",
        "Unsupervised learning discovers hidden patterns in unlabeled data.",
        "Transfer learning adapts pre-trained models to new but related tasks."
    ]
    
    # 1. Hierarchical RAG
    print("1. Hierarchical RAG:")
    hierarchical_rag = AdvancedRAGFactory.create_hierarchical(levels=3)
    hierarchical_rag.build_hierarchical_structure(sample_docs)
    
    hierarchical_results = hierarchical_rag.retrieve_hierarchical(
        "machine learning techniques", 
        top_k_per_level=[2, 3, 4]
    )
    
    print(f"  Retrieved from {len(hierarchical_results)} levels")
    for level, results in hierarchical_results.items():
        print(f"    Level {level}: {len(results)} results")
    print()
    
    # 2. Graph-Based RAG
    print("2. Graph-Based RAG:")
    graph_rag = AdvancedRAGFactory.create_graph_based()
    graph_rag.build_knowledge_graph(sample_docs)
    
    graph_results = graph_rag.retrieve_via_graph("neural networks", top_k=3)
    print(f"  Retrieved {len(graph_results)} results with graph relationships")
    if graph_results:
        print(f"  Example - Related entities for first result: {graph_results[0].get('related_entities', [])[:2]}")
    print()
    
    # 3. Recursive RAG
    print("3. Recursive RAG:")
    recursive_rag = AdvancedRAGFactory.create_recursive()
    recursive_rag.add_documents(sample_docs)
    
    recursive_results = recursive_rag.retrieve_recursive("deep learning architectures")
    print(f"  Completed recursive retrieval at depth {recursive_results['depth']}")
    print(f"  Retrieved {len(recursive_results['results'])} final results")
    print()
    
    # 4. Multi-Modal RAG
    print("4. Multi-Modal RAG:")
    multimodal_rag = AdvancedRAGFactory.create_multimodal()
    multimodal_rag.add_text_content(sample_docs)
    
    multimodal_results = multimodal_rag.retrieve_multimodal("machine learning concepts")
    print(f"  Retrieved {len(multimodal_results['results'])} results using multi-modal approach")
    print(f"  Modalities used: {multimodal_results['modalities_used']}")
    print()
    
    # 5. Multi-Hop RAG
    print("5. Multi-Hop RAG:")
    multihop_rag = AdvancedRAGFactory.create_multihop()
    multihop_rag.add_documents(sample_docs)
    
    multihop_results = multihop_rag.retrieve_multihop("How does deep learning relate to neural networks?")
    print(f"  Completed {len(multihop_results['hops'])} reasoning hops")
    for hop in multihop_results['hops']:
        print(f"    Hop {hop['hop']}: '{hop['query']}' -> {len(hop['results'])} results")
    print()


def performance_comparison():
    """Compare performance of different architectures"""
    print("\n=== Performance Comparison ===\n")
    
    sample_docs = [
        "Artificial intelligence and machine learning are transforming various industries.",
        "Deep neural networks require significant computational resources for training.",
        "Natural language processing enables machines to understand human communication.",
        "Computer vision systems can identify objects and scenes in images.",
        "Reinforcement learning algorithms learn through trial and error.",
    ] * 10  # Multiply to have more documents
    
    architectures = {
        "Hierarchical": lambda: AdvancedRAGFactory.create_hierarchical().build_hierarchical_structure(sample_docs),
        "Graph-Based": lambda: AdvancedRAGFactory.create_graph_based().build_knowledge_graph(sample_docs),
        "Recursive": lambda: (lambda rag: [rag.add_documents(sample_docs), rag.retrieve_recursive("AI")][1])(AdvancedRAGFactory.create_recursive()),
        "Multi-Hop": lambda: (lambda rag: [rag.add_documents(sample_docs), rag.retrieve_multihop("AI")][1])(AdvancedRAGFactory.create_multihop())
    }
    
    import time
    
    results = {}
    for name, arch_func in architectures.items():
        start_time = time.time()
        result = arch_func()
        elapsed_time = time.time() - start_time
        results[name] = elapsed_time
        print(f"  {name}: {elapsed_time:.4f}s")
    
    print(f"\nFastest architecture: {min(results, key=results.get)} ({results[min(results, key=results.get)]:.4f}s)")


def main():
    """Main function to demonstrate advanced RAG architectures"""
    print("Module 14: Advanced Architectures")
    print("=" * 50)
    
    # Demonstrate advanced architectures
    demonstrate_advanced_architectures()
    
    # Show performance comparison
    performance_comparison()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of combining architectures
    print("Combining Hierarchical and Graph-Based Approaches:")
    
    # Create hierarchical structure
    hierarchical = AdvancedRAGFactory.create_hierarchical(levels=2)
    docs = ["Machine learning algorithms are powerful", "Neural networks are part of deep learning"]
    hierarchical.build_hierarchical_structure(docs)
    
    # Then apply graph-based analysis to the results
    graph_rag = AdvancedRAGFactory.create_graph_based()
    graph_rag.build_knowledge_graph(docs)
    
    # This demonstrates how architectures can be combined
    combined_results = graph_rag.retrieve_via_graph("machine learning algorithms")
    print(f"Combined approach retrieved {len(combined_results)} results")
    
    print(f"\nModule 14 completed - Advanced architectures implemented and demonstrated")


if __name__ == "__main__":
    main()