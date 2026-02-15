# RAG Course: Deep Dive into Retrieval Augmented Generation



[![Website](https://img.shields.io/badge/Website-rag--course--two.vercel.app-blue)](https://rag-course-two.vercel.app/)

Welcome to the comprehensive RAG (Retrieval Augmented Generation) course! This deep-dive course covers 15 essential modules to master RAG systems, from foundational concepts to advanced production implementations.

**View the course online at [rag-course-two.vercel.app](https://rag-course-two.vercel.app/) for a better reading experience with syntax highlighting and diagrams.**

## Course Overview

This course provides an in-depth exploration of RAG systems with practical, testable code examples. Each module includes:

- **Theoretical Foundations**: Mathematical and conceptual understanding
- **Technical Implementation**: Production-ready code examples
- **Real-World Applications**: Industry-specific implementations
- **Performance Optimization**: Strategies for production environments
- **Evaluation Frameworks**: Methods to measure and improve system performance
- **Production Deployment**: Scalable architecture considerations

## Course Modules

1. **Module 1 - Reranking (Two-Step Retrieval with Cross-Encoders)**: Advanced cross-encoder techniques for improving retrieval precision
2. **Module 2 - Chunking Strategies and Document Segmentation**: Optimal document segmentation techniques
3. **Module 3 - Embedding Models and Selection**: Choosing and evaluating embedding models for specific domains
4. **Module 4 - Vector Databases and Storage**: Vector database options and optimization strategies
5. **Module 5 - Hybrid Search Techniques**: Combining semantic and keyword search methods effectively
6. **Module 6 - Query Understanding and Transformation**: Techniques for improving query quality and search performance
7. **Module 7 - Multi-Query and Query Expansion**: Methods for generating better search queries from user input
8. **Module 8 - Context Compression and Summarization**: Approaches to reduce context length while preserving relevance
9. **Module 9 - Evaluation Metrics for RAG**: Precision, recall, MRR, and other metrics for measuring RAG performance
10. **Module 10 - Metadata-Enhanced Retrieval**: Using metadata to improve search and generation quality
11. **Module 11 - Agentic RAG Workflows**: Advanced autonomous systems with memory and self-improvement
12. **Module 12 - RAG Optimization Techniques**: Latency reduction, cost optimization, and performance improvements
13. **Module 13 - Domain Adaptation in RAG**: Customizing RAG systems for specific industries and use cases
14. **Module 14 - Advanced RAG Architectures**: Self-RAG, CRAG, and other sophisticated approaches
15. **Module 15 - Production Deployment and Monitoring**: Scaling RAG systems and maintaining performance in production

## Prerequisites

- Intermediate Python programming skills
- Understanding of machine learning concepts
- Familiarity with Large Language Models
- Basic knowledge of vector databases and embeddings

## Installation

```bash
pip install torch transformers sentence-transformers langchain chromadb numpy pandas
```

## Navigation Guide

For a comprehensive overview of each module, its importance, and practical applications in the LLM world, please see the [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md) file.

## Running Examples

Each module contains testable code examples. To run examples from a specific module:

```bash
cd modules/module_01_reranking
python example.py
```

## Code Quality

All code examples in this course are:

- **Modular**: Clean, well-organized code structure
- **Testable**: Includes unit tests and validation examples
- **Production-Ready**: Optimized for real-world deployments
- **Well-Documented**: Comprehensive comments and documentation
- **Performance-Oriented**: Efficient algorithms and caching strategies

## Contributing

This course is designed to be a living document. Contributions and improvements are welcome through pull requests.

## License

This course is open-source and free to use for educational and commercial purposes.
