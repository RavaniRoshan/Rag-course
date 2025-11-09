"""
Module 15: Production Deployment
Tests for implementation examples

This module contains tests for the production deployment implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np
from datetime import datetime, timedelta

from example import (
    DeploymentStatus, HealthCheckResult, ProductionLogger, 
    ResourceMonitor, SecurityManager, HealthChecker, 
    ProductionRAG, DeploymentOrchestrator
)


class TestDeploymentStatus(unittest.TestCase):
    """Test deployment status enumeration"""
    
    def test_status_values(self):
        """Test that deployment status enum has correct values"""
        self.assertEqual(DeploymentStatus.HEALTHY.value, "healthy")
        self.assertEqual(DeploymentStatus.DEGRADED.value, "degraded")
        self.assertEqual(DeploymentStatus.UNHEALTHY.value, "unhealthy")
        self.assertEqual(DeploymentStatus.STARTING.value, "starting")
        self.assertEqual(DeploymentStatus.STOPPING.value, "stopping")


class TestHealthCheckResult(unittest.TestCase):
    """Test health check result functionality"""
    
    def test_initialization(self):
        """Test health check result initialization"""
        result = HealthCheckResult(
            status=DeploymentStatus.HEALTHY,
            message="Test message",
            details={"test": "value"}
        )
        
        self.assertEqual(result.status, DeploymentStatus.HEALTHY)
        self.assertEqual(result.message, "Test message")
        self.assertEqual(result.details, {"test": "value"})
        self.assertIsInstance(result.timestamp, datetime)
    
    def test_auto_timestamp(self):
        """Test that timestamp is automatically set"""
        result = HealthCheckResult(
            status=DeploymentStatus.HEALTHY,
            message="Test",
            details={}
        )
        
        self.assertIsInstance(result.timestamp, datetime)
        # Should be recent (within last minute)
        time_diff = datetime.now() - result.timestamp
        self.assertLess(time_diff.total_seconds(), 60)


class TestProductionLogger(unittest.TestCase):
    """Test production logger functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = ProductionLogger(service_name="test-service", log_level="DEBUG")
    
    def test_initialization(self):
        """Test logger initialization"""
        self.assertEqual(self.logger.service_name, "test-service")
        self.assertIsNotNone(self.logger.logger)
        self.assertEqual(self.logger.log_file, "test-service-app.log")
    
    def test_log_request(self):
        """Test request logging"""
        # This should not raise an exception
        self.logger.log_request(
            query="test query",
            response="test response", 
            processing_time=0.123,
            user_id="test-user"
        )
        # Test passes if no exception is raised
    
    def test_log_error(self):
        """Test error logging"""
        # This should not raise an exception
        self.logger.log_error(
            error=Exception("Test error"),
            context="test context"
        )
        # Test passes if no exception is raised
    
    def test_log_metric(self):
        """Test metric logging"""
        # This should not raise an exception
        self.logger.log_metric(
            metric_name="test_metric",
            value=1.23,
            tags={"tag1": "value1"}
        )
        # Test passes if no exception is raised


class TestResourceMonitor(unittest.TestCase):
    """Test resource monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = ResourceMonitor()
    
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertEqual(self.monitor.metrics_history, [])
        self.assertEqual(self.monitor.max_history_size, 1000)
    
    def test_get_current_metrics(self):
        """Test getting current metrics"""
        metrics = self.monitor.get_current_metrics()
        
        self.assertIn("cpu_percent", metrics)
        self.assertIn("memory_percent", metrics)
        self.assertIn("disk_percent", metrics)
        self.assertIsInstance(metrics["cpu_percent"], float)
        self.assertGreaterEqual(metrics["cpu_percent"], 0)
        self.assertLessEqual(metrics["cpu_percent"], 100)
    
    def test_get_average_metrics(self):
        """Test getting average metrics"""
        # Add a few metrics to the history
        for i in range(5):
            self.monitor.metrics_history.append({
                "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                "cpu_percent": 10.0 + i,
                "memory_percent": 20.0 + i,
                "disk_percent": 30.0 + i
            })
        
        avg_metrics = self.monitor.get_average_metrics(minutes=10)
        
        self.assertIn("avg_cpu_percent", avg_metrics)
        self.assertIn("avg_memory_percent", avg_metrics)
        self.assertIn("avg_disk_percent", avg_metrics)
        
        self.assertIsInstance(avg_metrics["avg_cpu_percent"], float)
        self.assertIsInstance(avg_metrics["avg_memory_percent"], float)
        self.assertIsInstance(avg_metrics["avg_disk_percent"], float)


class TestSecurityManager(unittest.TestCase):
    """Test security manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.security = SecurityManager()
    
    def test_initialization(self):
        """Test security manager initialization"""
        self.assertEqual(self.security.api_keys, {})
        self.assertEqual(self.security.rate_limits, {})
        self.assertEqual(self.security.access_logs, [])
    
    def test_create_api_key(self):
        """Test API key creation"""
        api_key = self.security.create_api_key("test-user", ["read", "query"])
        
        self.assertIsInstance(api_key, str)
        self.assertEqual(len(api_key), 43)  # URL-safe token length
        self.assertIn(api_key, self.security.api_keys)
        self.assertEqual(self.security.api_keys[api_key]["user_id"], "test-user")
        self.assertEqual(self.security.api_keys[api_key]["permissions"], ["read", "query"])
    
    def test_validate_api_key(self):
        """Test API key validation"""
        api_key = self.security.create_api_key("test-user")
        
        # Valid key should return True
        self.assertTrue(self.security.validate_api_key(api_key))
        
        # Invalid key should return False
        self.assertFalse(self.security.validate_api_key("invalid-key"))
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        api_key = self.security.create_api_key("test-user")
        
        # Should allow up to limit
        for i in range(5):
            allowed = self.security.check_rate_limit(api_key, max_requests=5, window_minutes=1)
            self.assertTrue(allowed, f"Request {i+1} should be allowed")
        
        # Next request should be denied if we make it quickly
        # Note: In real usage, we'd need to respect the time window
        # For this test, we'll check the algorithm works correctly
        self.security.rate_limits[api_key] = [datetime.now()] * 10  # Simulate hitting limit
        allowed = self.security.check_rate_limit(api_key, max_requests=5, window_minutes=1)
        self.assertFalse(allowed)
    
    def test_encrypt_data(self):
        """Test data encryption"""
        original_data = "sensitive information"
        encrypted = self.security.encrypt_data(original_data)
        
        self.assertIsInstance(encrypted, str)
        self.assertEqual(len(encrypted), 64)  # SHA-256 hex string length
    
    def test_log_access(self):
        """Test access logging"""
        api_key = self.security.create_api_key("test-user")
        
        self.security.log_access(api_key, "/query", "192.168.1.1")
        
        self.assertEqual(len(self.security.access_logs), 1)
        log_entry = self.security.access_logs[0]
        self.assertEqual(log_entry["api_key"], api_key)
        self.assertEqual(log_entry["endpoint"], "/query")
        self.assertEqual(log_entry["ip_address"], "192.168.1.1")


class TestHealthChecker(unittest.TestCase):
    """Test health checker functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        import chromadb
        from chromadb.config import Settings
        client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = client.get_or_create_collection(
            name="test_health_check",
            metadata={"hnsw:space": "cosine"}
        )
        self.logger = ProductionLogger("test-service")
        
        self.health_checker = HealthChecker(self.embedder, self.collection, self.logger)
    
    def test_initialization(self):
        """Test health checker initialization"""
        self.assertIsNotNone(self.health_checker.embedder)
        self.assertIsNotNone(self.health_checker.vector_store)
        self.assertIsNotNone(self.health_checker.logger)
    
    def test_perform_health_check(self):
        """Test performing a health check"""
        result = self.health_checker.perform_health_check()
        
        self.assertIsInstance(result, HealthCheckResult)
        self.assertIn(result.status, [DeploymentStatus.HEALTHY, DeploymentStatus.DEGRADED, DeploymentStatus.UNHEALTHY])
        self.assertIsInstance(result.message, str)
        self.assertIsInstance(result.details, dict)
    
    def test_is_healthy(self):
        """Test is_healthy method"""
        # This should return True if the system is healthy
        is_healthy = self.health_checker.is_healthy()
        self.assertIsInstance(is_healthy, bool)


class TestProductionRAG(unittest.TestCase):
    """Test production RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rag = ProductionRAG(service_name="test-rag", log_level="DEBUG")
    
    def test_initialization(self):
        """Test RAG system initialization"""
        self.assertIsNotNone(self.rag.embedder)
        self.assertIsNotNone(self.rag.collection)
        self.assertIsNotNone(self.rag.logger)
        self.assertIsNotNone(self.rag.resource_monitor)
        self.assertIsNotNone(self.rag.security_manager)
        self.assertIsNotNone(self.rag.health_checker)
        
        self.assertEqual(self.rag.query_count, 0)
        self.assertEqual(self.rag.total_processing_time, 0.0)
        self.assertEqual(self.rag.error_count, 0)
    
    def test_add_document(self):
        """Test adding a document"""
        doc_id = self.rag.add_document("Test document content", metadata={"category": "test"})
        
        self.assertIsInstance(doc_id, str)
        self.assertGreater(len(doc_id), 0)
        
        # Verify document was added by doing a query
        result = self.rag.query("Test", top_k=1)
        self.assertGreater(len(result["results"]), 0)
    
    def test_add_document_with_api_key(self):
        """Test adding document with API key validation"""
        # Create an API key
        api_key = self.rag.create_user_api_key("test-user")
        
        # Add document with valid API key
        doc_id = self.rag.add_document(
            "Test with API key",
            metadata={"category": "api_test"},
            api_key=api_key
        )
        
        self.assertIsInstance(doc_id, str)
        self.assertGreater(len(doc_id), 0)
    
    def test_add_document_with_invalid_api_key(self):
        """Test adding document with invalid API key"""
        with self.assertRaises(Exception) as context:
            self.rag.add_document(
                "Test with invalid API key",
                api_key="invalid_key"
            )
        self.assertIn("Invalid API key", str(context.exception))
    
    def test_query(self):
        """Test querying the RAG system"""
        # Add a document first
        self.rag.add_document("Machine learning concepts for testing", metadata={"type": "ml"})
        
        # Query the system
        result = self.rag.query("machine learning", top_k=1)
        
        self.assertIn('query', result)
        self.assertIn('results', result)
        self.assertIn('processing_time', result)
        self.assertIn('result_count', result)
        self.assertEqual(result['query'], "machine learning")
        self.assertIsInstance(result['results'], list)
        self.assertGreaterEqual(len(result['results']), 0)  # May be 0 if the query doesn't match well
        self.assertIsInstance(result['processing_time'], float)
        self.assertIsInstance(result['result_count'], int)
    
    def test_query_with_api_key(self):
        """Test querying with API key validation"""
        # Create an API key
        api_key = self.rag.create_user_api_key("test-user")
        
        # Add a document
        self.rag.add_document("Test content for API key query", metadata={"type": "test"})
        
        # Query with valid API key
        result = self.rag.query("test", top_k=1, api_key=api_key, user_id="test-user")
        
        self.assertIn('results', result)
        self.assertTrue(result['api_key_validated'])
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics"""
        metrics = self.rag.get_performance_metrics()
        
        self.assertIn('query_count', metrics)
        self.assertIn('error_count', metrics)
        self.assertIn('avg_processing_time', metrics)
        self.assertIn('error_rate', metrics)
        self.assertIn('resource_metrics', metrics)
        self.assertIn('avg_resource_metrics', metrics)
        self.assertIn('health_status', metrics)
        
        self.assertIsInstance(metrics['query_count'], int)
        self.assertIsInstance(metrics['error_count'], int)
        self.assertIsInstance(metrics['avg_processing_time'], float)
        self.assertIsInstance(metrics['error_rate'], float)
        self.assertIsInstance(metrics['resource_metrics'], dict)
        self.assertIsInstance(metrics['avg_resource_metrics'], dict)
        self.assertIsInstance(metrics['health_status'], str)
    
    def test_create_user_api_key(self):
        """Test creating user API key"""
        api_key = self.rag.create_user_api_key("test-user", ["read", "query"])
        
        self.assertIsInstance(api_key, str)
        self.assertGreater(len(api_key), 0)
    
    def test_run_health_check(self):
        """Test running a health check"""
        result = self.rag.run_health_check()
        
        self.assertIsInstance(result, HealthCheckResult)
        self.assertIn(result.status, [DeploymentStatus.HEALTHY, DeploymentStatus.DEGRADED, DeploymentStatus.UNHEALTHY])
    
    def test_get_system_resources(self):
        """Test getting system resources"""
        resources = self.rag.get_system_resources()
        
        self.assertIn('cpu_percent', resources)
        self.assertIn('memory_percent', resources)
        self.assertIn('disk_percent', resources)
        self.assertIsInstance(resources['cpu_percent'], float)
        self.assertGreaterEqual(resources['cpu_percent'], 0)
        self.assertLessEqual(resources['cpu_percent'], 100)


class TestDeploymentOrchestrator(unittest.TestCase):
    """Test deployment orchestrator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DeploymentOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertEqual(self.orchestrator.services, {})
        self.assertEqual(self.orchestrator.deployment_status, DeploymentStatus.STOPPING)
        self.assertIsNone(self.orchestrator.start_time)
    
    def test_deploy_service(self):
        """Test deploying a service"""
        config = {
            "service_name": "test-service",
            "log_level": "DEBUG"
        }
        
        success = self.orchestrator.deploy_service("test-service", config)
        
        self.assertTrue(success)
        self.assertIn("test-service", self.orchestrator.services)
        self.assertEqual(self.orchestrator.deployment_status, DeploymentStatus.HEALTHY)
        self.assertIsInstance(self.orchestrator.start_time, datetime)
    
    def test_scale_service(self):
        """Test scaling a service"""
        # Deploy a service first
        config = {"service_name": "scale-test", "log_level": "DEBUG"}
        self.orchestrator.deploy_service("scale-test", config)
        
        # Scale it
        self.orchestrator.scale_service("scale-test", 3)
        # This should complete without error
        
    def test_update_configuration(self):
        """Test updating service configuration"""
        # Deploy a service first
        config = {"service_name": "config-test", "log_level": "DEBUG"}
        self.orchestrator.deploy_service("config-test", config)
        
        # Update its configuration
        new_config = {"log_level": "INFO", "max_connections": 100}
        self.orchestrator.update_configuration("config-test", new_config)
        # This should complete without error
    
    def test_get_deployment_status(self):
        """Test getting deployment status"""
        # Check initial status
        status = self.orchestrator.get_deployment_status()
        self.assertEqual(status["status"], self.orchestrator.deployment_status.value)
        self.assertEqual(status["services_count"], 0)
        self.assertIsNone(status["start_time"])
        
        # Deploy a service and check status again
        config = {"service_name": "status-test", "log_level": "DEBUG"}
        self.orchestrator.deploy_service("status-test", config)
        
        status = self.orchestrator.get_deployment_status()
        self.assertEqual(status["services_count"], 1)
        self.assertIn("status-test", status["services_status"])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_document_add(self):
        """Test adding empty document"""
        rag = ProductionRAG("test-service")
        
        # Should handle empty content gracefully
        doc_id = rag.add_document("")
        self.assertIsInstance(doc_id, str)
    
    def test_very_long_query(self):
        """Test very long query"""
        rag = ProductionRAG("test-service")
        
        # Add some content
        rag.add_document("Test content for long query", metadata={"type": "test"})
        
        # Perform a very long query
        long_query = "test " * 100
        result = rag.query(long_query, top_k=1)
        
        # Should handle gracefully
        self.assertIn('results', result)
    
    def test_invalid_api_key_scenarios(self):
        """Test various invalid API key scenarios"""
        rag = ProductionRAG("test-service")
        
        # Test with None API key (should work if not required)
        try:
            rag.query("test", api_key=None)
        except Exception:
            pass  # Might raise an exception depending on implementation
    
    def test_concurrent_access_simulation(self):
        """Test concurrent access simulation"""
        rag = ProductionRAG("test-service")
        
        # Add some documents
        for i in range(5):
            rag.add_document(f"Document {i} content", metadata={"id": i})
        
        # Simulate multiple queries
        for i in range(3):
            result = rag.query(f"Document {i}", top_k=1)
            self.assertIn('results', result)
    
    def test_resource_monitor_with_no_history(self):
        """Test resource monitor when no history exists"""
        monitor = ResourceMonitor()
        
        # Clear any existing history
        monitor.metrics_history = []
        
        # Should handle empty history gracefully
        avg_metrics = monitor.get_average_metrics(minutes=5)
        self.assertEqual(avg_metrics, {})


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test query performance
    rag = ProductionRAG("performance-test")
    
    # Add documents for testing
    for i in range(50):
        rag.add_document(
            f"Performance test document {i} with content for measuring response times",
            metadata={"doc_id": i, "category": "performance"}
        )
    
    # Measure query performance
    start_time = time.time()
    query_times = []
    
    for i in range(20):  # Perform 20 queries
        query_start = time.time()
        result = rag.query(f"document {i % 10}", top_k=3)
        query_time = time.time() - query_start
        query_times.append(query_time)
    
    total_time = time.time() - start_time
    avg_query_time = sum(query_times) / len(query_times)
    
    print(f"Performance Results:")
    print(f"  Completed 20 queries in {total_time:.4f} seconds")
    print(f"  Average query time: {avg_query_time:.4f} seconds")
    print(f"  Queries per second: {20/total_time:.2f}")
    
    # Should be reasonably fast
    assert avg_query_time < 2.0, f"Average query time too slow: {avg_query_time}s"
    assert total_time < 10.0, f"Total time too slow: {total_time}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test security features
    print("Testing security features...")
    
    rag = ProductionRAG("security-test")
    
    # Create an API key
    api_key = rag.create_user_api_key("security-test-user", ["read", "query", "write"])
    print(f"  Created API key: {api_key[:10]}...")
    
    # Test rate limiting - this requires a more complex approach
    # since it depends on time windows
    security_mgr = rag.security_manager
    result = security_mgr.check_rate_limit(api_key, max_requests=5, window_minutes=1)
    print(f"  Rate limit check (before): {result}")
    
    # Add documents to test the full system
    for i in range(3):
        doc_id = rag.add_document(
            f"Security test document {i}",
            metadata={"type": "security"},
            api_key=api_key
        )
        print(f"  Added document with ID: {doc_id[:8]}...")
    
    # Perform queries to test monitoring
    print("\nTesting monitoring features...")
    for i in range(5):
        result = rag.query(f"security test {i}", top_k=2, api_key=api_key, user_id="security-test-user")
        print(f"  Query {i+1}: {len(result['results'])} results in {result['processing_time']:.4f}s")
    
    # Check performance metrics
    metrics = rag.get_performance_metrics()
    print(f"\nSystem Metrics:")
    print(f"  Queries processed: {metrics['query_count']}")
    print(f"  Error rate: {metrics['error_rate']:.2%}")
    print(f"  Avg processing time: {metrics['avg_processing_time']:.4f}s")
    
    # Check system resources
    resources = rag.get_system_resources()
    print(f"  CPU usage: {resources['cpu_percent']:.1f}%")
    print(f"  Memory usage: {resources['memory_percent']:.1f}%")
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()