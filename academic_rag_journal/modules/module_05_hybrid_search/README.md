# Module 5: Hybrid Search

## Overview
This module explores hybrid search techniques that combine semantic search (vector similarity) with traditional keyword-based search methods. Hybrid search leverages the strengths of both approaches to improve retrieval accuracy and relevance in RAG systems.

## Learning Objectives
- Understand the principles of hybrid search in RAG systems
- Learn different methods for combining semantic and keyword search
- Implement various hybrid search algorithms
- Evaluate hybrid search effectiveness
- Optimize hybrid search for different use cases
- Deploy hybrid search in production environments

## Topics Covered
1. Fundamentals of Hybrid Search
2. Combining Semantic and Keyword Search
3. Scoring and Ranking Strategies
4. Implementation Techniques
5. Performance Optimization
6. Production Considerations

## Prerequisites
- Understanding of vector embeddings and similarity search
- Knowledge of keyword search and information retrieval
- Familiarity with Python and common ML libraries
- Understanding of Module 4 (Vector Databases)

## Table of Contents
1. [Introduction to Hybrid Search](#introduction)
2. [Hybrid Search Methods](#methods)
3. [Implementation Examples](#implementation)
4. [Scoring and Ranking](#scoring)
5. [Performance Considerations](#performance)
6. [Production Deployment](#production)
7. [Hands-on Exercises](#exercises)
8. [Evaluation](#evaluation)

## Introduction {#introduction}

Traditional RAG systems often rely primarily on semantic search using vector embeddings. However, semantic search can sometimes miss relevant documents that contain important keywords but don't match semantically, or it may return results that are semantically similar but not directly relevant to the query.

Hybrid search addresses these limitations by combining:
- **Keyword search**: Traditional search using inverted indices and term matching
- **Semantic search**: Vector-based search using embeddings and similarity metrics
- **Relevance scoring**: Combining scores from both approaches

### Benefits of Hybrid Search
- Improved recall: Catches documents that match keywords but not semantics
- Better precision: Filters out semantically similar but irrelevant results
- Robustness: Works well across diverse query types
- Flexibility: Allows tuning based on use case requirements

### Challenges
- Complexity: More complex than single-method approaches
- Parameter tuning: Requires balancing keyword and semantic components
- Performance: May be slower than single-method searches

## Hybrid Search Methods {#methods}

### 1. Reciprocal Rank Fusion (RRF)
Reciprocal Rank Fusion is a technique that combines results from different retrieval systems. For each document, RRF calculates a score based on the rank of the document in each result set.

Formula: `score(d) = Σ(1 / (k + rank_i(d)))` where k is a smoothing constant and rank_i(d) is the rank of document d in the i-th result set.

### 2. Linear Combination
A simple method where keyword scores and semantic scores are linearly combined with weights:

`final_score = α * keyword_score + β * semantic_score`

### 3. Late Interaction
Combines keyword and semantic features after initial retrieval from both systems, often using a learned ranking model.

### 4. Cross-Encoder Rescoring
Uses a cross-encoder model to re-rank results from both keyword and semantic search methods by considering query-document pairs.

## Implementation Examples {#implementation}

The `example.py` file contains implementations of:
- BM25 keyword search integration
- Vector similarity search
- Reciprocal Rank Fusion implementation
- Linear combination approaches
- Performance comparison of different methods

## Scoring and Ranking {#scoring}

### Normalization Techniques
Different search methods produce scores on different scales. Proper normalization is crucial:
- Min-max normalization
- Z-score normalization
- Probabilistic normalization

### Weight Selection
- Equal weights: Simple but often suboptimal
- Learned weights: Using validation data to optimize weights
- Dynamic weights: Adjusting weights based on query characteristics

## Performance Considerations {#performance}

### Indexing Strategies
- Separate keyword and vector indices
- Concurrent querying of both systems
- Caching frequently accessed results

### Latency Optimization
- Query parallelization
- Result set pruning
- Approximate methods for speed

## Production Deployment {#production}

### Architecture Patterns
- Separate search services
- Unified search API
- Caching layers
- Load balancing

### Monitoring and A/B Testing
- Performance metrics
- Relevance metrics
- A/B testing frameworks
- Continuous optimization

## Hands-on Exercises {#exercises}

1. **Implement RRF**: Build a Reciprocal Rank Fusion system
2. **Compare Methods**: Evaluate different hybrid search approaches
3. **Parameter Tuning**: Optimize weights for your specific dataset
4. **Performance Testing**: Benchmark your hybrid search system
5. **Production Setup**: Deploy a production-ready hybrid search system

## Evaluation {#evaluation}

### Metrics
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Recall@K
- Precision@K
- Hit rate

### Evaluation Datasets
- Custom annotated datasets
- Public benchmarks (MS MARCO, Natural Questions)
- Domain-specific test sets

## Next Steps
After completing this module, you'll be able to implement effective hybrid search systems that combine the best of both keyword and semantic search approaches. This leads to better retrieval quality in your RAG systems.

Proceed to [Module 6: Query Understanding](../module_06_query_understanding/README.md) to learn how to process and understand user queries more effectively.