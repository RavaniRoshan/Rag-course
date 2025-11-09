"""
Module 15: Production Deployment
Implementation Examples

This module demonstrates production deployment patterns for RAG systems,
including monitoring, security, scalability, and operational best practices.
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
from datetime import datetime, timedelta
import threading
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import os
import hashlib
from functools import wraps
import psutil  # For system resource monitoring
import cpuinfo  # For CPU info
import secrets


class DeploymentStatus(Enum):
    """Enumeration for deployment status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    status: DeploymentStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ProductionLogger:
    """Production-grade logging system"""
    
    def __init__(self, service_name: str = "rag-service", log_level: str = "INFO"):
        self.service_name = service_name
        self.logger = logging.getLogger(f"{service_name}-logger")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter that includes service name and timestamp
        formatter = logging.Formatter(
            f'%(asctime)s - {service_name} - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (in a real system, you'd use a more robust solution)
        self.log_file = f"{service_name}-app.log"
    
    def log_request(self, query: str, response: str, processing_time: float, user_id: str = None):
        """Log a request with structured data"""
        log_data = {
            "event": "request_processed",
            "service": self.service_name,
            "query_length": len(query),
            "response_length": len(response),
            "processing_time": processing_time,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context"""
        log_data = {
            "event": "error_occurred",
            "service": self.service_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.error(json.dumps(log_data))
    
    def log_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Log a metric with tags"""
        log_data = {
            "event": "metric",
            "service": self.service_name,
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info(json.dumps(log_data))


class ResourceMonitor:
    """Monitor system resources and performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.max_history_size = 1000
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource usage metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_info.percent,
            "memory_available_gb": memory_info.available / (1024**3),
            "disk_percent": disk_usage.percent,
            "process_count": len(psutil.pids()),
            "cpu_info": cpuinfo.get_cpu_info()["brand_raw"] if cpuinfo.get_cpu_info() else "Unknown"
        }
        
        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        return metrics
    
    def get_average_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """Get average metrics over the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m["timestamp"][:-3] + m["timestamp"][-2:]) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        avg_metrics = {}
        for key in ["cpu_percent", "memory_percent", "disk_percent"]:
            values = [m[key] for m in recent_metrics]
            avg_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        return avg_metrics


class SecurityManager:
    """Security and access control manager"""
    
    def __init__(self):
        self.api_keys = {}  # In production, use a database
        self.rate_limits = {}  # Per API key rate limits
        self.encryption_keys = {}  # For data encryption
        self.access_logs = []
    
    def create_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Create a new API key for a user"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions or ["read", "query"],
            "created_at": datetime.now(),
            "last_used": None
        }
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key"""
        if api_key not in self.api_keys:
            return False
        
        # Update last used time
        self.api_keys[api_key]["last_used"] = datetime.now()
        return True
    
    def check_rate_limit(self, api_key: str, max_requests: int = 100, window_minutes: int = 1) -> bool:
        """Check if a user has exceeded rate limits"""
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = []
        
        # Clean old requests (older than window)
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        self.rate_limits[api_key] = [
            req_time for req_time in self.rate_limits[api_key]
            if req_time > cutoff_time
        ]
        
        # Check if under limit
        current_requests = len(self.rate_limits[api_key])
        if current_requests >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[api_key].append(datetime.now())
        return True
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data (simplified for example)"""
        # In production, use proper encryption like Fernet
        # For this example, we'll use a simple hash-based approach
        key = hashlib.sha256("production-deployment-key".encode()).digest()
        encrypted = hashlib.sha256((data + key.hex()).encode()).hexdigest()
        return encrypted
    
    def log_access(self, api_key: str, endpoint: str, ip_address: str = None):
        """Log access for audit trail"""
        log_entry = {
            "timestamp": datetime.now(),
            "api_key": api_key,
            "user_id": self.api_keys.get(api_key, {}).get("user_id", "unknown"),
            "endpoint": endpoint,
            "ip_address": ip_address,
            "permissions": self.api_keys.get(api_key, {}).get("permissions", [])
        }
        self.access_logs.append(log_entry)


class HealthChecker:
    """System health checker for production deployment"""
    
    def __init__(self, embedder, vector_store, logger):
        self.embedder = embedder
        self.vector_store = vector_store
        self.logger = logger
        self.last_health_check = None
        self.health_check_interval = 60  # seconds
    
    def perform_health_check(self) -> HealthCheckResult:
        """Perform a comprehensive health check"""
        try:
            # Check embedder is responsive
            test_embedding = self.embedder.encode(["health check"])
            embedder_healthy = test_embedding.shape[0] > 0
            
            # Check vector store is responsive (add and query a test item)
            test_id = f"health_check_{int(time.time())}"
            test_content = "This is a test document for health checking"
            
            self.vector_store.add(
                embeddings=[test_embedding[0].tolist()],
                documents=[test_content],
                metadatas=[{"type": "health_check"}],
                ids=[test_id]
            )
            
            # Query the test item back
            results = self.vector_store.query(
                query_embeddings=[test_embedding[0].tolist()],
                n_results=1
            )
            
            store_healthy = len(results['ids'][0]) > 0
            
            # Clean up test item
            # Note: In real chromadb, you'd need to delete this, but for this example we'll just log
            self.logger.logger.debug(f"Health check cleanup ID: {test_id}")
            
            # Overall assessment
            if embedder_healthy and store_healthy:
                status = DeploymentStatus.HEALTHY
                message = "All components healthy"
            elif embedder_healthy or store_healthy:
                status = DeploymentStatus.DEGRADED
                message = "Some components degraded"
            else:
                status = DeploymentStatus.UNHEALTHY
                message = "System unhealthy"
            
            details = {
                "embedder_healthy": embedder_healthy,
                "vector_store_healthy": store_healthy,
                "test_query_time": time.time() - time.time()  # Placeholder
            }
            
            self.last_health_check = datetime.now()
            
            return HealthCheckResult(status=status, message=message, details=details)
            
        except Exception as e:
            self.logger.log_error(e, "Health check failed")
            return HealthCheckResult(
                status=DeploymentStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                details={"error": str(e)}
            )
    
    def is_healthy(self) -> bool:
        """Check if the system is healthy"""
        result = self.perform_health_check()
        return result.status in [DeploymentStatus.HEALTHY, DeploymentStatus.DEGRADED]


class ProductionRAG:
    """Production-ready RAG system with all operational features"""
    
    def __init__(self, 
                 service_name: str = "production-rag",
                 log_level: str = "INFO"):
        # Initialize components
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="production_rag",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize operational components
        self.logger = ProductionLogger(service_name, log_level)
        self.resource_monitor = ResourceMonitor()
        self.security_manager = SecurityManager()
        self.health_checker = HealthChecker(self.embedder, self.collection, self.logger)
        
        # Performance metrics
        self.query_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        self.logger.logger.info(f"Initialized {service_name} production RAG system")
    
    def add_document(self, 
                    content: str, 
                    metadata: Dict[str, Any] = None,
                    api_key: str = None) -> str:
        """Add a document with security and logging"""
        start_time = time.time()
        
        try:
            # Validate API key if provided
            if api_key:
                if not self.security_manager.validate_api_key(api_key):
                    self.logger.log_error(
                        Exception("Invalid API key"), 
                        "Document add attempt with invalid API key"
                    )
                    raise Exception("Invalid API key")
                
                if not self.security_manager.check_rate_limit(api_key):
                    self.logger.log_error(
                        Exception("Rate limit exceeded"), 
                        "Document add attempt exceeded rate limit"
                    )
                    raise Exception("Rate limit exceeded")
            
            # Generate embedding and add to collection
            embedding = self.embedder.encode([content])
            doc_id = str(uuid.uuid4())
            metadata = metadata or {}
            metadata['id'] = doc_id
            metadata['created_at'] = datetime.now().isoformat()
            
            self.collection.add(
                embeddings=embedding.tolist(),
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            # Log the successful operation
            processing_time = time.time() - start_time
            self.logger.log_metric("document_add_duration", processing_time)
            
            return doc_id
            
        except Exception as e:
            self.error_count += 1
            self.logger.log_error(e, "Document add failed")
            raise e
    
    def query(self, 
              query: str, 
              top_k: int = 5,
              api_key: str = None,
              user_id: str = None) -> Dict[str, Any]:
        """Query with full operational support"""
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Validate API key if provided
            if api_key:
                if not self.security_manager.validate_api_key(api_key):
                    raise Exception("Invalid API key")
                
                if not self.security_manager.check_rate_limit(api_key):
                    raise Exception("Rate limit exceeded")
            
            # Perform query
            query_embedding = self.embedder.encode([query])
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1.0 - results['distances'][0][i]
                })
            
            # Calculate metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Log the query
            response_summary = f"Returned {len(formatted_results)} results"
            self.logger.log_request(query, response_summary, processing_time, user_id)
            
            # Log performance metrics
            self.logger.log_metric("query_duration", processing_time, 
                                 {"top_k": str(top_k), "result_count": str(len(formatted_results))})
            
            return {
                'query': query,
                'results': formatted_results,
                'processing_time': processing_time,
                'result_count': len(formatted_results),
                'api_key_validated': api_key is not None
            }
            
        except Exception as e:
            self.error_count += 1
            processing_time = time.time() - start_time
            self.logger.log_request(query, f"Error: {str(e)}", processing_time, user_id)
            self.logger.log_error(e, "Query failed")
            raise e
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance and health metrics"""
        avg_processing_time = (
            self.total_processing_time / self.query_count if self.query_count > 0 else 0
        )
        
        resource_metrics = self.resource_monitor.get_current_metrics()
        avg_resource_metrics = self.resource_monitor.get_average_metrics()
        
        return {
            'query_count': self.query_count,
            'error_count': self.error_count,
            'avg_processing_time': avg_processing_time,
            'error_rate': self.error_count / self.query_count if self.query_count > 0 else 0,
            'resource_metrics': resource_metrics,
            'avg_resource_metrics': avg_resource_metrics,
            'health_status': self.health_checker.perform_health_check().status.value
        }
    
    def create_user_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Create an API key for a user"""
        api_key = self.security_manager.create_api_key(user_id, permissions)
        self.logger.logger.info(f"Created API key for user {user_id}")
        return api_key
    
    def run_health_check(self) -> HealthCheckResult:
        """Run a health check on the system"""
        result = self.health_checker.perform_health_check()
        self.logger.logger.info(f"Health check result: {result.status.value} - {result.message}")
        return result
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information"""
        return self.resource_monitor.get_current_metrics()


class DeploymentOrchestrator:
    """Orchestrate production deployment operations"""
    
    def __init__(self):
        self.services = {}
        self.deployment_status = DeploymentStatus.STOPPING
        self.start_time = None
    
    def deploy_service(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Deploy a service with the given configuration"""
        try:
            # Initialize the RAG service
            service = ProductionRAG(
                service_name=service_name,
                log_level=config.get("log_level", "INFO")
            )
            
            # Store the service
            self.services[service_name] = service
            self.deployment_status = DeploymentStatus.STARTING
            self.start_time = datetime.now()
            
            # Perform initial health check
            health_result = service.run_health_check()
            if health_result.status == DeploymentStatus.HEALTHY:
                self.deployment_status = DeploymentStatus.HEALTHY
                return True
            else:
                self.deployment_status = DeploymentStatus.DEGRADED
                return False
                
        except Exception as e:
            self.deployment_status = DeploymentStatus.UNHEALTHY
            print(f"Deployment failed: {e}")
            return False
    
    def scale_service(self, service_name: str, replicas: int):
        """Scale a service to the specified number of replicas"""
        if service_name in self.services:
            print(f"Scaling {service_name} to {replicas} replicas")
            # In a real system, this would manage actual replicas
            # For this example, we'll just log the action
            self.services[service_name].logger.logger.info(
                f"Scaled service {service_name} to {replicas} replicas"
            )
    
    def update_configuration(self, service_name: str, new_config: Dict[str, Any]):
        """Update service configuration"""
        if service_name in self.services:
            print(f"Updating configuration for {service_name}")
            # In a real system, this would apply the new configuration
            # For this example, we'll just log the action
            self.services[service_name].logger.logger.info(
                f"Configuration updated for {service_name}"
            )
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get overall deployment status"""
        if not self.services:
            return {
                "status": self.deployment_status.value,
                "services_count": 0,
                "start_time": None
            }
        
        services_status = {}
        for name, service in self.services.items():
            health = service.run_health_check()
            services_status[name] = {
                "status": health.status.value,
                "message": health.message
            }
        
        return {
            "status": self.deployment_status.value,
            "services_count": len(self.services),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "services_status": services_status
        }


def demonstrate_production_deployment():
    """Demonstrate production deployment features"""
    print("=== Production Deployment Demonstration ===\n")
    
    # Create orchestrator
    orchestrator = DeploymentOrchestrator()
    
    # Deploy a production RAG service
    config = {
        "service_name": "production-rag-service",
        "log_level": "INFO",
        "replicas": 1
    }
    
    print("1. Deploying production RAG service...")
    success = orchestrator.deploy_service("production-rag", config)
    print(f"   Deployment successful: {success}")
    
    # Get the deployed service
    rag_service = orchestrator.services["production-rag"]
    
    # Create an API key for demonstration
    print("\n2. Creating user and API key...")
    api_key = rag_service.create_user_api_key("demo_user", ["read", "query", "write"])
    print(f"   API key created: {api_key[:10]}...")
    
    # Add some documents
    print("\n3. Adding documents with security...")
    doc1 = rag_service.add_document(
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data",
        metadata={"category": "AI", "source": "tutorial"},
        api_key=api_key
    )
    doc2 = rag_service.add_document(
        "Deep learning uses neural networks with multiple layers to process complex patterns",
        metadata={"category": "ML", "source": "research"},
        api_key=api_key
    )
    print(f"   Added documents with IDs: {doc1}, {doc2}")
    
    # Perform queries with monitoring
    print("\n4. Performing queries with full monitoring...")
    result1 = rag_service.query(
        "machine learning concepts",
        top_k=2,
        api_key=api_key,
        user_id="demo_user"
    )
    print(f"   Query 1: {len(result1['results'])} results in {result1['processing_time']:.4f}s")
    
    result2 = rag_service.query(
        "neural networks",
        top_k=1,
        api_key=api_key,
        user_id="demo_user"
    )
    print(f"   Query 2: {len(result2['results'])} results in {result2['processing_time']:.4f}s")
    
    # Check health
    print("\n5. Checking system health...")
    health = rag_service.run_health_check()
    print(f"   Health status: {health.status.value}")
    print(f"   Message: {health.message}")
    
    # Get performance metrics
    print("\n6. Getting performance metrics...")
    metrics = rag_service.get_performance_metrics()
    print(f"   Total queries: {metrics['query_count']}")
    print(f"   Error count: {metrics['error_count']}")
    print(f"   Average processing time: {metrics['avg_processing_time']:.4f}s")
    print(f"   Error rate: {metrics['error_rate']:.2%}")
    
    # Get system resources
    print("\n7. Checking system resources...")
    resources = rag_service.get_system_resources()
    print(f"   CPU usage: {resources['cpu_percent']:.1f}%")
    print(f"   Memory usage: {resources['memory_percent']:.1f}%")
    
    print(f"\nProduction deployment demonstration completed!")


def performance_and_scalability_demo():
    """Demonstrate performance and scalability features"""
    print("\n=== Performance and Scalability Demo ===\n")
    
    # Create production service
    rag = ProductionRAG("scalability-demo")
    
    # Add more documents for stress testing
    print("Adding multiple documents for performance testing...")
    for i in range(50):
        content = f"Document {i} containing information about topic number {i} for performance testing."
        rag.add_document(content, metadata={"type": f"doc_{i}"})
    
    # Perform multiple queries to test performance
    print("\nPerforming multiple queries to test performance...")
    start_time = time.time()
    
    query_results = []
    for i in range(20):  # 20 queries
        result = rag.query(f"topic {i % 10}", top_k=3)  # Query different topics
        query_results.append(result)
    
    total_time = time.time() - start_time
    
    print(f"Completed 20 queries in {total_time:.4f} seconds")
    print(f"Average query time: {total_time/20:.4f} seconds")
    print(f"Total processing time tracked: {rag.total_processing_time:.4f} seconds")
    
    # Check metrics
    metrics = rag.get_performance_metrics()
    print(f"Query rate: {metrics['query_count']/total_time:.2f} queries/second")
    print(f"Current error rate: {metrics['error_rate']:.2%}")


def main():
    """Main function to demonstrate production deployment"""
    print("Module 15: Production Deployment")
    print("=" * 50)
    
    # Demonstrate production deployment features
    demonstrate_production_deployment()
    
    # Show performance and scalability
    performance_and_scalability_demo()
    
    # Show deployment orchestration
    print("\n=== Deployment Orchestration Demo ===")
    
    orchestrator = DeploymentOrchestrator()
    
    config = {"service_name": "demo-service", "log_level": "INFO"}
    success = orchestrator.deploy_service("demo-service", config)
    
    if success:
        print("✓ Service deployed successfully")
        
        # Scale the service
        orchestrator.scale_service("demo-service", 3)
        print("✓ Service scaled to 3 replicas")
        
        # Update configuration
        new_config = {"log_level": "DEBUG", "max_connections": 100}
        orchestrator.update_configuration("demo-service", new_config)
        print("✓ Configuration updated")
        
        # Check deployment status
        status = orchestrator.get_deployment_status()
        print(f"✓ Deployment status: {status['status']}")
        print(f"✓ Number of services: {status['services_count']}")
    
    print(f"\nModule 15 completed - Production deployment patterns implemented and demonstrated")
    
    print(f"\nCourse Conclusion:")
    print(f"This comprehensive RAG course has covered all essential aspects from fundamental concepts to production deployment.")
    print(f"You now have a deep understanding of RAG systems and are well-equipped to deploy them in production environments.")
    print(f"Remember to continuously monitor and improve your systems based on usage patterns and performance metrics.")


if __name__ == "__main__":
    main()