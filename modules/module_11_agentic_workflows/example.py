"""
Module 11: Agentic Workflows
Implementation Examples

This module demonstrates agentic workflows in RAG systems,
including planning, reasoning, tool usage, and memory management.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import time
import uuid
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class ToolType(Enum):
    RETRIEVAL = "retrieval"
    CALCULATION = "calculation"
    WEB_SEARCH = "web_search"
    OTHER = "other"


@dataclass
class ToolCall:
    """Represents a tool call in the agent workflow"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentStep:
    """Represents a step in the agent's reasoning process"""
    step_type: str  # "think", "act", "observe", "response"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_call: Optional[ToolCall] = None


class SimpleMemory:
    """Simple memory system for the agent"""
    
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
        self.short_term = []  # Conversation history
        self.long_term = {}   # Persistent knowledge
    
    def add_to_short_term(self, content: str):
        """Add content to short-term memory"""
        self.short_term.append({
            'content': content,
            'timestamp': datetime.now()
        })
        
        # Trim if context gets too long
        while self.get_short_term_length() > self.max_context_length and len(self.short_term) > 1:
            self.short_term.pop(0)
    
    def get_short_term_length(self) -> int:
        """Get the length of short-term memory content"""
        return sum(len(item['content']) for item in self.short_term)
    
    def get_recent_context(self, max_length: int = 2000) -> str:
        """Get recent context from memory"""
        context_parts = []
        current_length = 0
        
        # Add items in reverse order (most recent first)
        for item in reversed(self.short_term):
            item_length = len(item['content'])
            if current_length + item_length > max_length:
                break
            context_parts.insert(0, item['content'])
            current_length += item_length
        
        return "\n".join(context_parts)
    
    def add_to_long_term(self, key: str, value: Any):
        """Add knowledge to long-term memory"""
        self.long_term[key] = value
    
    def get_from_long_term(self, key: str) -> Any:
        """Retrieve from long-term memory"""
        return self.long_term.get(key)


class Tool:
    """Base class for tools that agents can use"""
    
    def __init__(self, name: str, description: str, tool_type: ToolType):
        self.name = name
        self.description = description
        self.tool_type = tool_type
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments"""
        raise NotImplementedError


class RetrievalTool(Tool):
    """Tool for retrieving documents from a vector database"""
    
    def __init__(self, retriever):
        super().__init__("retrieval", "Search a document collection for relevant information", ToolType.RETRIEVAL)
        self.retriever = retriever
    
    def execute(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Execute retrieval from the database"""
        results = self.retriever.retrieve(query, top_k=top_k)
        return results


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations"""
    
    def __init__(self):
        super().__init__("calculator", "Perform mathematical calculations", ToolType.CALCULATION)
    
    def execute(self, expression: str) -> float:
        """Execute a mathematical expression"""
        # This is a simple calculator - in production, use a safe evaluation method
        try:
            # Only allow safe mathematical expressions
            allowed_chars = set('0123456789+-*/().e ')
            if not all(c in allowed_chars for c in expression.replace(' ', '')):
                raise ValueError("Invalid characters in expression")
            
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except:
            return "Error: Invalid expression"


class WebSearchTool(Tool):
    """Tool for simulating web search"""
    
    def __init__(self):
        super().__init__("web_search", "Simulate web search for current information", ToolType.WEB_SEARCH)
    
    def execute(self, query: str) -> List[Dict[str, str]]:
        """Simulate web search results"""
        # In a real implementation, this would call an actual search API
        # For this example, we'll return simulated results
        return [
            {"title": f"Result 1 for: {query}", "url": "https://example.com/1", "snippet": "This is a relevant snippet about your query"},
            {"title": f"Result 2 for: {query}", "url": "https://example.com/2", "snippet": "Another relevant snippet about your query"}
        ]


class Agent:
    """Base agent class with planning, reasoning, and tool usage capabilities"""
    
    def __init__(self, tools: List[Tool] = None, memory: SimpleMemory = None):
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.memory = memory or SimpleMemory()
        self.conversation_history = []
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def think(self, query: str, context: str = "") -> str:
        """Perform reasoning/thinking step"""
        thought = f"Let me think about this query: '{query}'. I should consider: {context[:100] if context else 'the context and relevant information'}"
        return thought
    
    def plan(self, query: str) -> List[Dict[str, Any]]:
        """Create a plan to answer the query"""
        # Simple planning logic - in practice, this would be more sophisticated
        tools_to_use = []
        
        # Determine which tools might be needed based on query
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['calculate', 'compute', 'math', 'sum', 'total']):
            tools_to_use.append({
                'tool_name': 'calculator',
                'arguments': {'expression': '1+1'}  # Placeholder - would be extracted from query
            })
        
        if any(word in query_lower for word in ['find', 'search', 'information', 'document', 'paper']):
            tools_to_use.append({
                'tool_name': 'retrieval',
                'arguments': {'query': query, 'top_k': 3}
            })
            
        if any(word in query_lower for word in ['current', 'latest', 'recent', 'today']):
            tools_to_use.append({
                'tool_name': 'web_search',
                'arguments': {'query': query}
            })
        
        return tools_to_use or [{'tool_name': 'retrieval', 'arguments': {'query': query, 'top_k': 3}}]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with given arguments"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not available"
        
        tool = self.tools[tool_name]
        try:
            result = tool.execute(**arguments)
            return result
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def react_step(self, query: str, thought: str, tool_call: Dict[str, Any]) -> Tuple[str, Any]:
        """Execute a single ReAct step (Reason + Act)"""
        # Add thought to memory
        self.memory.add_to_short_term(f"Thought: {thought}")
        
        # Execute tool
        tool_result = self.execute_tool(tool_call['tool_name'], tool_call['arguments'])
        
        # Add tool call and result to memory
        tool_call_str = f"Action: {tool_call['tool_name']} with args {tool_call['arguments']}"
        self.memory.add_to_short_term(tool_call_str)
        self.memory.add_to_short_term(f"Observation: {tool_result}")
        
        return tool_call_str, tool_result
    
    def run(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        """Run the agent on a query using ReAct approach"""
        all_steps = []
        
        # Initial planning
        plan = self.plan(query)
        all_steps.append(AgentStep("plan", f"Plan: {plan}"))
        
        # Execute up to max_steps
        step_count = 0
        current_query = query
        
        while step_count < max_steps and plan:
            # Take first action from plan
            next_action = plan.pop(0)
            
            # Think about the action
            context = self.memory.get_recent_context(max_length=1000)
            thought = self.think(current_query, context)
            
            # Execute ReAct step
            action_str, result = self.react_step(current_query, thought, next_action)
            
            # Add step to history
            step = AgentStep(
                "act", 
                f"{thought}\n{action_str}\nObservation: {result}",
                tool_call=ToolCall(next_action['tool_name'], next_action['arguments'], result)
            )
            all_steps.append(step)
            
            # Update query if needed based on result
            # In a more complex system, this would involve reasoning about the result
            if "Error" not in str(result) and result:
                # If we got useful results, we might want to refine the query
                current_query = query  # In this simple example, we don't refine the query
            
            step_count += 1
        
        # Generate final response based on all steps
        final_response = self.generate_response(query, all_steps)
        
        return {
            'query': query,
            'steps': [step.content for step in all_steps],
            'final_response': final_response,
            'tool_calls': [step.tool_call for step in all_steps if step.tool_call],
            'step_count': step_count
        }
    
    def generate_response(self, query: str, steps: List[AgentStep]) -> str:
        """Generate the final response based on all steps"""
        # In a real implementation, this would synthesize the results from all steps
        # For this example, we'll just summarize the steps
        response_parts = [f"I addressed your query: '{query}'"]
        
        for i, step in enumerate(steps):
            if step.step_type != "plan":
                response_parts.append(f"Step {i+1}: {step.content[:200]}...")
        
        # Add a conclusion based on the tools used
        tool_calls = [step.tool_call for step in steps if step.tool_call]
        if tool_calls:
            tool_names = list(set(tc.tool_name for tc in tool_calls if tc))
            response_parts.append(f"I used the following tools: {', '.join(tool_names)}")
        
        response_parts.append("Is there anything else you'd like to know?")
        
        return "\n\n".join(response_parts)


class ReActAgent(Agent):
    """ReAct-style agent that interleaves reasoning and actions"""
    
    def __init__(self, tools: List[Tool] = None, memory: SimpleMemory = None):
        super().__init__(tools, memory)
    
    def run(self, query: str, max_steps: int = 5) -> Dict[str, Any]:
        """Run the ReAct agent on a query"""
        all_steps = []
        
        current_query = query
        step_count = 0
        
        while step_count < max_steps:
            # Think
            context = self.memory.get_recent_context(max_length=1000)
            thought = self.think(current_query, context)
            all_steps.append(AgentStep("think", thought))
            
            # Plan next action based on thought
            plan = self.plan(current_query)
            if not plan:
                break
            
            # Execute first action in plan
            action = plan[0]
            action_str, result = self.react_step(current_query, thought, action)
            
            # Add action and observation
            all_steps.append(AgentStep("act", action_str, tool_call=ToolCall(
                action['tool_name'], action['arguments'], result
            )))
            all_steps.append(AgentStep("observe", f"Observation: {result}"))
            
            step_count += 1
            
            # In a more sophisticated agent, we would update our query or plan based on the observation
            # For now, we'll continue with the original query
        
        final_response = self.generate_response(query, all_steps)
        
        return {
            'query': query,
            'steps': [step.content for step in all_steps],
            'final_response': final_response,
            'tool_calls': [step.tool_call for step in all_steps if step.tool_call],
            'step_count': step_count
        }


class ToolRegistry:
    """Registry for managing tools in the agentic system"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a registered tool"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())


class AgenticRAG:
    """Main class for agentic RAG system"""
    
    def __init__(self, tools: List[Tool] = None, agent: Agent = None):
        self.tool_registry = ToolRegistry()
        for tool in tools or []:
            self.tool_registry.register_tool(tool)
        
        self.agent = agent or ReActAgent(list(self.tool_registry.tools.values()))
        self.processing_history = []
    
    def query(self, user_query: str, max_steps: int = 5) -> Dict[str, Any]:
        """Process a user query using the agentic system"""
        start_time = time.time()
        
        # Run the agent
        result = self.agent.run(user_query, max_steps)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        # Add to processing history
        self.processing_history.append({
            'query': user_query,
            'result': result,
            'processing_time': processing_time,
            'timestamp': datetime.now()
        })
        
        return {
            'original_query': user_query,
            'response': result['final_response'],
            'steps': result['steps'],
            'tool_calls': result['tool_calls'],
            'processing_time': processing_time,
            'step_count': result['step_count']
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processing performance"""
        if not self.processing_history:
            return {}
        
        processing_times = [item['processing_time'] for item in self.processing_history]
        step_counts = [item['result']['step_count'] for item in self.processing_history]
        
        return {
            'total_queries': len(self.processing_history),
            'avg_processing_time': sum(processing_times) / len(processing_times),
            'total_processing_time': sum(processing_times),
            'avg_steps_per_query': sum(step_counts) / len(step_counts),
            'last_query_time': self.processing_history[-1]['processing_time']
        }


def demonstrate_agentic_workflows():
    """Demonstrate agentic workflow capabilities"""
    print("=== Agentic Workflows Demonstration ===\n")
    
    # Create tools
    calculator = CalculatorTool()
    web_search = WebSearchTool()
    
    # Note: We won't create a real retrieval tool for this example without a populated database
    # Instead, we'll create a mock one
    class MockRetrievalTool(Tool):
        def __init__(self):
            super().__init__("retrieval", "Mock retrieval tool", ToolType.RETRIEVAL)
        
        def execute(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
            return [
                {"id": 1, "content": f"Mock result for query: {query}", "metadata": {"source": "mock_db"}}
            ][:top_k]
    
    retrieval = MockRetrievalTool()
    tools = [calculator, web_search, retrieval]
    
    # Create agent with tools
    agent = ReActAgent(tools=tools)
    
    # Create agentic RAG system
    agentic_rag = AgenticRAG(tools=tools, agent=agent)
    
    print("Testing different types of queries:\n")
    
    # Test 1: Calculation query
    print("1. Calculation query:")
    calc_result = agentic_rag.query("What is 25 * 4 + 10?")
    print(f"Query: {calc_result['original_query']}")
    print(f"Response: {calc_result['response']}")
    print(f"Processing time: {calc_result['processing_time']:.3f}s")
    print(f"Steps taken: {calc_result['step_count']}")
    print()
    
    # Test 2: Informational query
    print("2. Informational query:")
    info_result = agentic_rag.query("What can you tell me about agentic workflows?")
    print(f"Query: {info_result['original_query']}")
    print(f"Response: {info_result['response'][:200]}...")
    print(f"Processing time: {info_result['processing_time']:.3f}s")
    print()
    
    # Test 3: Multi-step query
    print("3. Multi-step query:")
    multi_result = agentic_rag.query("Find information about RAG systems, then calculate 15% of 200")
    print(f"Query: {multi_result['original_query']}")
    print(f"Response: {multi_result['response'][:200]}...")
    print(f"Processing time: {multi_result['processing_time']:.3f}s")
    print(f"Steps taken: {multi_result['step_count']}")
    print()
    
    # Show processing stats
    stats = agentic_rag.get_processing_stats()
    print("Processing Statistics:")
    print(f"  Total queries processed: {stats['total_queries']}")
    print(f"  Average processing time: {stats['avg_processing_time']:.3f}s")
    print(f"  Average steps per query: {stats['avg_steps_per_query']:.1f}")


class AgenticWorkflowEvaluator:
    """Evaluator for agentic workflows"""
    
    def __init__(self):
        # Create mock tools for testing
        self.calculator = CalculatorTool()
        self.web_search = WebSearchTool()
        
        class MockRetrievalTool(Tool):
            def __init__(self):
                super().__init__("retrieval", "Mock retrieval tool", ToolType.RETRIEVAL)
            
            def execute(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
                return [{"id": 1, "content": f"Mock result for: {query}", "metadata": {"source": "mock"}}][:top_k]
        
        self.retrieval = MockRetrievalTool()
        self.tools = [self.calculator, self.web_search, self.retrieval]
    
    def evaluate_agent_types(self) -> Dict[str, Any]:
        """Evaluate different agent types"""
        queries = [
            "Calculate 12 * 15",
            "What is the capital of France?",
            "Find information about machine learning, then compute 2^8"
        ]
        
        results = {
            'react': []
        }
        
        # Test ReAct agent
        agent = ReActAgent(tools=self.tools)
        for query in queries:
            result = agent.run(query, max_steps=3)
            results['react'].append({
                'query': query,
                'response_length': len(result['final_response']),
                'step_count': result['step_count'],
                'has_tool_calls': any(result['tool_calls'])
            })
        
        return results
    
    def evaluate_tool_usage(self) -> Dict[str, Any]:
        """Evaluate tool usage effectiveness"""
        agent = ReActAgent(tools=self.tools)
        
        # Test each tool
        test_queries = {
            'calculator': "What is 50 + 25?",
            'retrieval': "Tell me about agentic systems",
            'web_search': "What is the current date?"
        }
        
        tool_usage_results = {}
        
        for tool_name, query in test_queries.items():
            result = agent.run(query, max_steps=2)
            
            # Check if the correct tool was used
            tool_calls = [tc.tool_name for tc in result['tool_calls'] if tc]
            correct_tool_used = tool_name in tool_calls or any(tn.startswith(tool_name) for tn in tool_calls)
            
            tool_usage_results[tool_name] = {
                'query': query,
                'correct_tool_used': correct_tool_used,
                'tools_used': tool_calls,
                'response_length': len(result['final_response'])
            }
        
        return tool_usage_results


def performance_comparison():
    """Compare performance of different approaches"""
    print("\n=== Performance Comparison ===\n")
    
    evaluator = AgenticWorkflowEvaluator()
    
    # Time the evaluation
    start_time = time.time()
    results = evaluator.evaluate_agent_types()
    elapsed_time = time.time() - start_time
    
    print(f"Evaluated agent types in {elapsed_time:.3f} seconds")
    print(f"Tested queries: {len(results['react'])}")
    
    # Analyze results
    avg_steps = np.mean([r['step_count'] for r in results['react']])
    print(f"Average steps per query: {avg_steps:.1f}")
    
    tool_usage = evaluator.evaluate_tool_usage()
    print(f"\nTool usage evaluation:")
    for tool_name, result in tool_usage.items():
        print(f"  {tool_name}: Correctly used = {result['correct_tool_used']}, Tools used = {result['tools_used']}")


def main():
    """Main function to demonstrate agentic workflow implementations"""
    print("Module 11: Agentic Workflows")
    print("=" * 50)
    
    # Demonstrate agentic workflows
    demonstrate_agentic_workflows()
    
    # Show performance comparison
    performance_comparison()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example of tool registration and usage
    print("Tool Registration and Usage Example:")
    
    # Create and register tools
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(WebSearchTool())
    
    print(f"Registered tools: {registry.list_tools()}")
    
    # Create an agent with these tools
    agent_with_registry = ReActAgent(tools=[registry.get_tool(name) for name in registry.list_tools()])
    
    # Process a query that uses tools
    result = agent_with_registry.run("Calculate 100 / 4", max_steps=2)
    print(f"Result for calculation query: {result['final_response'][:100]}...")
    
    print(f"\nModule 11 completed - Agentic workflows implemented and demonstrated")


if __name__ == "__main__":
    main()