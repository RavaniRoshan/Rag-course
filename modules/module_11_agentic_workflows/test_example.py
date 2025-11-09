"""
Module 11: Agentic Workflows
Tests for implementation examples

This module contains tests for the agentic workflow implementations
to ensure they work correctly and meet performance requirements.
"""

import unittest
import tempfile
import shutil
from unittest.mock import patch
import numpy as np
from datetime import datetime

from example import (
    ToolType, ToolCall, AgentStep, SimpleMemory, Tool, RetrievalTool, 
    CalculatorTool, WebSearchTool, Agent, ReActAgent, ToolRegistry, 
    AgenticRAG, AgenticWorkflowEvaluator
)


class TestSimpleMemory(unittest.TestCase):
    """Test simple memory system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.memory = SimpleMemory(max_context_length=1000)
    
    def test_initialization(self):
        """Test memory initialization"""
        self.assertEqual(self.memory.max_context_length, 1000)
        self.assertEqual(self.memory.short_term, [])
        self.assertIsInstance(self.memory.long_term, dict)
    
    def test_add_to_short_term(self):
        """Test adding content to short-term memory"""
        content = "Test content"
        self.memory.add_to_short_term(content)
        
        self.assertEqual(len(self.memory.short_term), 1)
        self.assertEqual(self.memory.short_term[0]['content'], content)
        self.assertIsInstance(self.memory.short_term[0]['timestamp'], datetime)
    
    def test_short_term_length(self):
        """Test getting short-term memory length"""
        content1 = "Content 1"
        content2 = "Content 2"
        
        self.memory.add_to_short_term(content1)
        self.memory.add_to_short_term(content2)
        
        expected_length = len(content1) + len(content2)
        self.assertEqual(self.memory.get_short_term_length(), expected_length)
    
    def test_add_to_long_term(self):
        """Test adding to long-term memory"""
        self.memory.add_to_long_term("key1", "value1")
        self.assertEqual(self.memory.get_from_long_term("key1"), "value1")
        
        # Test overriding
        self.memory.add_to_long_term("key1", "value2")
        self.assertEqual(self.memory.get_from_long_term("key1"), "value2")
    
    def test_get_from_long_term_missing(self):
        """Test getting from long-term memory when key doesn't exist"""
        result = self.memory.get_from_long_term("nonexistent")
        self.assertIsNone(result)
    
    def test_get_recent_context(self):
        """Test getting recent context from memory"""
        # Add several items
        for i in range(5):
            self.memory.add_to_short_term(f"Item {i}")
        
        # Get context with different limits
        context = self.memory.get_recent_context(max_length=20)
        self.assertIsInstance(context, str)
        
        # Should contain the most recent items that fit
        self.assertLessEqual(len(context), 20)


class TestCalculatorTool(unittest.TestCase):
    """Test calculator tool"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = CalculatorTool()
    
    def test_initialization(self):
        """Test calculator tool initialization"""
        self.assertEqual(self.calculator.name, "calculator")
        self.assertEqual(self.calculator.tool_type, ToolType.CALCULATION)
        self.assertIn("mathematical", self.calculator.description)
    
    def test_execute_valid_expression(self):
        """Test executing a valid mathematical expression"""
        result = self.calculator.execute("2 + 3 * 4")
        self.assertEqual(result, 14.0)
    
    def test_execute_complex_expression(self):
        """Test executing a complex expression"""
        result = self.calculator.execute("10.5 + 5.5")
        self.assertEqual(result, 16.0)
    
    def test_execute_invalid_expression(self):
        """Test executing an invalid expression"""
        result = self.calculator.execute("2 + 3 * invalid_variable")
        self.assertEqual(result, "Error: Invalid expression")
    
    def test_execute_expression_with_variables(self):
        """Test executing expression with dangerous variables"""
        result = self.calculator.execute("__import__('os').listdir()")
        self.assertEqual(result, "Error: Invalid expression")


class TestWebSearchTool(unittest.TestCase):
    """Test web search tool"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.web_search = WebSearchTool()
    
    def test_initialization(self):
        """Test web search tool initialization"""
        self.assertEqual(self.web_search.name, "web_search")
        self.assertEqual(self.web_search.tool_type, ToolType.WEB_SEARCH)
    
    def test_execute_search(self):
        """Test executing a web search"""
        query = "test query"
        result = self.web_search.execute(query)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check structure of results
        first_result = result[0]
        self.assertIn('title', first_result)
        self.assertIn('url', first_result)
        self.assertIn('snippet', first_result)
        
        # Check that query is reflected in results
        self.assertIn(query, first_result['title'])


class TestToolCall(unittest.TestCase):
    """Test tool call functionality"""
    
    def test_initialization(self):
        """Test tool call initialization"""
        tool_call = ToolCall("test_tool", {"param": "value"})
        
        self.assertEqual(tool_call.tool_name, "test_tool")
        self.assertEqual(tool_call.arguments, {"param": "value"})
        self.assertIsNone(tool_call.result)
        self.assertIsInstance(tool_call.timestamp, datetime)
    
    def test_initialization_with_result(self):
        """Test tool call initialization with result"""
        tool_call = ToolCall("test_tool", {"param": "value"}, result="success")
        
        self.assertEqual(tool_call.result, "success")


class TestAgentStep(unittest.TestCase):
    """Test agent step functionality"""
    
    def test_initialization(self):
        """Test agent step initialization"""
        step = AgentStep("think", "This is a thought")
        
        self.assertEqual(step.step_type, "think")
        self.assertEqual(step.content, "This is a thought")
        self.assertIsInstance(step.timestamp, datetime)
        self.assertIsNone(step.tool_call)
    
    def test_initialization_with_tool_call(self):
        """Test agent step initialization with tool call"""
        tool_call = ToolCall("test", {})
        step = AgentStep("act", "Action taken", tool_call=tool_call)
        
        self.assertEqual(step.step_type, "act")
        self.assertEqual(step.content, "Action taken")
        self.assertEqual(step.tool_call, tool_call)


class TestAgent(unittest.TestCase):
    """Test base agent functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock tools for testing
        self.calculator = CalculatorTool()
        self.mock_tools = [self.calculator]
        self.agent = Agent(tools=self.mock_tools)
    
    def test_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent.tools)
        self.assertEqual(len(self.agent.tools), 1)
        self.assertIsNotNone(self.agent.memory)
        self.assertIsNotNone(self.agent.embedder)
    
    def test_think_method(self):
        """Test the think method"""
        query = "What is 2 + 2?"
        thought = self.agent.think(query)
        
        self.assertIsInstance(thought, str)
        self.assertIn("2 + 2", thought)
    
    def test_plan_method(self):
        """Test the plan method"""
        query = "Calculate 5 * 5"
        plan = self.agent.plan(query)
        
        self.assertIsInstance(plan, list)
        
        # Check if it plans to use calculator for calculation query
        calculator_plans = [p for p in plan if p['tool_name'] == 'calculator']
        self.assertGreater(len(calculator_plans), 0)
    
    def test_execute_tool(self):
        """Test tool execution"""
        result = self.agent.execute_tool("calculator", {"expression": "5 + 3"})
        self.assertEqual(result, 8.0)
    
    def test_execute_nonexistent_tool(self):
        """Test execution of non-existent tool"""
        result = self.agent.execute_tool("nonexistent_tool", {})
        self.assertIn("Error", result)
    
    def test_run_method(self):
        """Test the run method"""
        result = self.agent.run("Calculate 10 + 5", max_steps=2)
        
        self.assertIn('query', result)
        self.assertIn('steps', result)
        self.assertIn('final_response', result)
        self.assertIn('tool_calls', result)
        self.assertIn('step_count', result)
        
        self.assertEqual(result['query'], "Calculate 10 + 5")
        self.assertIsInstance(result['steps'], list)
        self.assertIsInstance(result['final_response'], str)
        self.assertIsInstance(result['tool_calls'], list)


class TestReActAgent(unittest.TestCase):
    """Test ReAct-style agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = CalculatorTool()
        self.web_search = WebSearchTool()
        
        class MockRetrievalTool(Tool):
            def __init__(self):
                super().__init__("retrieval", "Mock retrieval tool", ToolType.RETRIEVAL)
            
            def execute(self, query: str, top_k: int = 3) -> list:
                return [{"content": f"Mock result for {query}" for _ in range(top_k)}]
        
        self.retrieval = MockRetrievalTool()
        self.tools = [self.calculator, self.web_search, self.retrieval]
        self.agent = ReActAgent(tools=self.tools)
    
    def test_initialization(self):
        """Test ReAct agent initialization"""
        self.assertIsNotNone(self.agent.tools)
        self.assertEqual(len(self.agent.tools), 3)
        self.assertIsNotNone(self.agent.memory)
    
    def test_run_calculation_query(self):
        """Test running a calculation query"""
        result = self.agent.run("Calculate 20 / 4", max_steps=3)
        
        self.assertIn('query', result)
        self.assertIn('final_response', result)
        self.assertIn('step_count', result)
        
        # Should have used calculator tool
        tool_calls = [tc for tc in result['tool_calls'] if tc]
        calculator_calls = [tc for tc in tool_calls if tc.tool_name == 'calculator']
        self.assertGreater(len(calculator_calls), 0)


class TestToolRegistry(unittest.TestCase):
    """Test tool registry functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = ToolRegistry()
        self.calculator = CalculatorTool()
    
    def test_initialization(self):
        """Test registry initialization"""
        self.assertEqual(self.registry.tools, {})
    
    def test_register_tool(self):
        """Test registering a tool"""
        self.registry.register_tool(self.calculator)
        
        self.assertIn("calculator", self.registry.tools)
        self.assertEqual(self.registry.tools["calculator"], self.calculator)
    
    def test_get_tool(self):
        """Test getting a registered tool"""
        self.registry.register_tool(self.calculator)
        
        retrieved = self.registry.get_tool("calculator")
        self.assertEqual(retrieved, self.calculator)
    
    def test_get_nonexistent_tool(self):
        """Test getting a non-existent tool"""
        result = self.registry.get_tool("nonexistent")
        self.assertIsNone(result)
    
    def test_list_tools(self):
        """Test listing tools"""
        self.registry.register_tool(self.calculator)
        
        tools_list = self.registry.list_tools()
        self.assertIn("calculator", tools_list)
        self.assertEqual(len(tools_list), 1)


class TestAgenticRAG(unittest.TestCase):
    """Test agentic RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = CalculatorTool()
        self.web_search = WebSearchTool()
        
        class MockRetrievalTool(Tool):
            def __init__(self):
                super().__init__("retrieval", "Mock retrieval tool", ToolType.RETRIEVAL)
            
            def execute(self, query: str, top_k: int = 3) -> list:
                return [{"content": f"Mock result for {query}"} for _ in range(top_k)]
        
        self.retrieval = MockRetrievalTool()
        self.tools = [self.calculator, self.web_search, self.retrieval]
        self.agentic_rag = AgenticRAG(tools=self.tools)
    
    def test_initialization(self):
        """Test AgenticRAG initialization"""
        self.assertIsNotNone(self.agentic_rag.tool_registry)
        self.assertIsNotNone(self.agentic_rag.agent)
        self.assertEqual(len(self.agentic_rag.processing_history), 0)
    
    def test_query_method(self):
        """Test the query method"""
        result = self.agentic_rag.query("Calculate 15 * 3")
        
        self.assertIn('original_query', result)
        self.assertIn('response', result)
        self.assertIn('steps', result)
        self.assertIn('tool_calls', result)
        self.assertIn('processing_time', result)
        self.assertIn('step_count', result)
        
        self.assertEqual(result['original_query'], "Calculate 15 * 3")
        self.assertIsInstance(result['response'], str)
        self.assertIsInstance(result['steps'], list)
        self.assertIsInstance(result['processing_time'], float)
    
    def test_processing_stats_empty(self):
        """Test processing stats when no queries have been processed"""
        stats = self.agentic_rag.get_processing_stats()
        self.assertEqual(stats, {})
    
    def test_processing_stats_with_queries(self):
        """Test processing stats after processing queries"""
        # Process a few queries
        results = []
        for i in range(3):
            results.append(self.agentic_rag.query(f"Calculate {i+1} * 2"))
        
        stats = self.agentic_rag.get_processing_stats()
        
        self.assertIn('total_queries', stats)
        self.assertIn('avg_processing_time', stats)
        self.assertIn('total_processing_time', stats)
        self.assertIn('avg_steps_per_query', stats)
        self.assertIn('last_query_time', stats)
        
        self.assertEqual(stats['total_queries'], 3)
        self.assertGreaterEqual(stats['total_processing_time'], 0)
        self.assertGreaterEqual(stats['avg_processing_time'], 0)
        self.assertGreaterEqual(stats['avg_steps_per_query'], 0)


class TestAgenticWorkflowEvaluator(unittest.TestCase):
    """Test agentic workflow evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = AgenticWorkflowEvaluator()
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator.calculator)
        self.assertIsNotNone(self.evaluator.web_search)
        self.assertIsNotNone(self.evaluator.retrieval)
        self.assertEqual(len(self.evaluator.tools), 3)
    
    def test_evaluate_agent_types(self):
        """Test agent type evaluation"""
        results = self.evaluator.evaluate_agent_types()
        
        self.assertIn('react', results)
        self.assertEqual(len(results['react']), 3)  # 3 test queries
        
        # Check structure of results
        for result in results['react']:
            self.assertIn('query', result)
            self.assertIn('response_length', result)
            self.assertIn('step_count', result)
            self.assertIn('has_tool_calls', result)
    
    def test_evaluate_tool_usage(self):
        """Test tool usage evaluation"""
        results = self.evaluator.evaluate_tool_usage()
        
        expected_tools = {'calculator', 'retrieval', 'web_search'}
        self.assertEqual(set(results.keys()), expected_tools)
        
        # Check structure of each result
        for tool_name, result in results.items():
            self.assertIn('query', result)
            self.assertIn('correct_tool_used', result)
            self.assertIn('tools_used', result)
            self.assertIn('response_length', result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_tool_list(self):
        """Test agent with empty tool list"""
        agent = Agent(tools=[])
        result = agent.run("Simple query", max_steps=2)
        
        # Should still work, just without tools
        self.assertIn('final_response', result)
        self.assertIsInstance(result['final_response'], str)
    
    def test_memory_with_very_long_content(self):
        """Test memory with very long content"""
        memory = SimpleMemory(max_context_length=50)  # Very small limit
        
        long_content = "This is a very long content that exceeds the memory limit. " * 10
        memory.add_to_short_term(long_content)
        
        # Should have trimmed the content
        context = memory.get_recent_context(max_length=100)
        self.assertLessEqual(len(context), 100)
    
    def test_agent_with_no_memory(self):
        """Test agent with custom memory"""
        memory = SimpleMemory(max_context_length=100)
        agent = Agent(memory=memory)
        
        result = agent.run("Test query", max_steps=1)
        self.assertIn('final_response', result)
    
    def test_tool_execution_error_handling(self):
        """Test tool execution error handling"""
        calculator = CalculatorTool()
        
        # This should cause an error in the calculator
        result = calculator.execute("invalid syntax +")
        self.assertEqual(result, "Error: Invalid expression")
    
    def test_agent_max_steps_limit(self):
        """Test that agent respects max steps limit"""
        calculator = CalculatorTool()
        agent = ReActAgent(tools=[calculator])
        
        result = agent.run("Calculate 5 + 5", max_steps=1)
        
        # Should not exceed max steps
        self.assertLessEqual(result['step_count'], 1)


def performance_tests():
    """Run performance-related tests"""
    print("Running performance tests...")
    
    import time
    
    # Test with multiple queries
    calculator = CalculatorTool()
    web_search = WebSearchTool()
    
    class MockRetrievalTool(Tool):
        def __init__(self):
            super().__init__("retrieval", "Mock retrieval tool", ToolType.RETRIEVAL)
        
        def execute(self, query: str, top_k: int = 3) -> list:
            return [{"content": f"Mock result for {query}"} for _ in range(top_k)]
    
    retrieval = MockRetrievalTool()
    tools = [calculator, web_search, retrieval]
    agentic_rag = AgenticRAG(tools=tools)
    
    queries = [
        "Calculate 12 * 12",
        "Search for Python programming",
        "Calculate 2 ^ 10"
    ] * 5  # Repeat 5 times
    
    start_time = time.time()
    for query in queries:
        result = agentic_rag.query(query, max_steps=2)
    total_time = time.time() - start_time
    
    avg_time = total_time / len(queries)
    
    print(f"Processed {len(queries)} queries in {total_time:.3f} seconds")
    print(f"Average time per query: {avg_time:.3f} seconds")
    
    # Should be reasonably fast
    assert avg_time < 5.0, f"Average query processing time too slow: {avg_time}s"


def run_additional_tests():
    """Run additional tests that are not typical unit tests"""
    print("\nRunning additional tests...")
    
    # Test complex agent interaction
    print("Testing complex agent interaction...")
    
    calculator = CalculatorTool()
    web_search = WebSearchTool()
    
    class MockRetrievalTool(Tool):
        def __init__(self):
            super().__init__("retrieval", "Mock retrieval tool", ToolType.RETRIEVAL)
        
        def execute(self, query: str, top_k: int = 3) -> list:
            return [{"content": f"Retrieved: {query}"} for _ in range(top_k)]
    
    retrieval = MockRetrievalTool()
    tools = [calculator, web_search, retrieval]
    agentic_rag = AgenticRAG(tools=tools)
    
    # Test a complex multi-step query
    complex_query = "Find information about RAG systems, calculate 15% of 200, and tell me the current year"
    result = agentic_rag.query(complex_query, max_steps=5)
    
    print(f"Complex query result has {result['step_count']} steps")
    print(f"Used tools: {[tc.tool_name for tc in result['tool_calls'] if tc]}")
    
    # Verify that multiple tools were used
    tool_names = {tc.tool_name for tc in result['tool_calls'] if tc}
    expected_tools = {'retrieval', 'calculator', 'web_search'}
    print(f"Expected tools {expected_tools}, got {tool_names}")
    
    print("Additional tests completed successfully")


if __name__ == "__main__":
    # Run standard unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    performance_tests()
    run_additional_tests()