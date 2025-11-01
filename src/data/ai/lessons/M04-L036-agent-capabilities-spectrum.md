# Agent Capabilities Spectrum

## Core Concepts

### Technical Definition

An agent is a system that uses an LLM to autonomously make decisions about which actions to take, execute those actions, and iterate based on results until reaching a goal. The agent capabilities spectrum describes the range from simple single-turn completions to complex multi-step autonomous systems.

Unlike traditional software that follows predetermined execution paths, agents combine:
- **Dynamic decision-making**: The LLM determines next steps based on current state
- **Tool integration**: Ability to interact with external systems and APIs
- **Iterative refinement**: Learning from action outcomes to adjust approach
- **Goal-oriented behavior**: Working toward objectives rather than just responding

### Engineering Analogy

Consider the difference between traditional scripting and agent-based approaches:

**Traditional Approach (Fixed Logic):**
```python
def process_customer_request(request: str) -> str:
    """Fixed decision tree - every path predetermined"""
    if "refund" in request.lower():
        return check_refund_eligibility()
    elif "shipping" in request.lower():
        return get_shipping_status()
    elif "product" in request.lower():
        return search_product_catalog()
    else:
        return "I don't understand. Please contact support."
```

**Agent Approach (Dynamic Decision-Making):**
```python
from typing import List, Dict, Callable
import json

class CustomerServiceAgent:
    """Agent that decides which tools to use based on context"""
    
    def __init__(self, llm_client, tools: Dict[str, Callable]):
        self.llm = llm_client
        self.tools = tools
        self.max_iterations = 5
    
    def process_request(self, request: str) -> str:
        """Agent decides what actions to take"""
        conversation_history = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": request}
        ]
        
        for iteration in range(self.max_iterations):
            # Agent decides next action
            response = self.llm.complete(
                messages=conversation_history,
                tools=self._format_tools()
            )
            
            # Check if agent wants to use a tool
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = self._execute_tool(
                        tool_call.name, 
                        tool_call.arguments
                    )
                    # Feed result back to agent
                    conversation_history.append({
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": tool_call.id
                    })
            else:
                # Agent has final answer
                return response.content
                
        return "Could not complete request in allowed steps"
    
    def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute tool and return structured result"""
        if tool_name in self.tools:
            try:
                return {"success": True, "data": self.tools[tool_name](**args)}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "Tool not found"}
    
    def _format_tools(self) -> List[dict]:
        """Convert tools to OpenAI function format"""
        return [
            {
                "name": "check_order_status",
                "description": "Get current status of customer order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"}
                    }
                }
            },
            {
                "name": "process_refund",
                "description": "Initiate refund if eligible",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "reason": {"type": "string"}
                    }
                }
            },
            {
                "name": "search_knowledge_base",
                "description": "Search for product information and policies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        ]
    
    def _build_system_prompt(self) -> str:
        return """You are a customer service agent. Use available tools to:
1. Gather information about the customer's situation
2. Take appropriate actions
3. Provide a helpful response

Think step-by-step and use tools as needed."""
```

The traditional approach breaks with unexpected inputs. The agent approach handles novel situations by reasoning about which tools apply, potentially chaining multiple actions, and adapting based on intermediate results.

### Key Insights

**1. Agents Trade Determinism for Flexibility**

Traditional software is predictable—same input always produces same output. Agents introduce variability because the LLM makes decisions. This means:
- You can't unit test exact responses
- You must test behavior patterns and boundaries
- Debugging requires examining decision chains, not just code paths

**2. The Spectrum is About Control vs. Autonomy**

You don't need full autonomy for most problems. The spectrum lets you dial in the right level:
- **Zero autonomy**: LLM generates text, you control everything else
- **Tool selection**: LLM picks which tools to use, but you approve actions
- **Supervised chains**: LLM executes sequences, but you review critical steps
- **Full autonomy**: LLM plans and executes without intervention (highest risk)

**3. More Autonomy ≠ Better Solution**

Each level up the spectrum adds:
- Latency (more LLM calls = more time)
- Cost (every decision point costs tokens)
- Unpredictability (more places for unexpected behavior)
- Failure modes (longer chains have more break points)

Start with minimal autonomy and add only what you need.

### Why This Matters Now

The gap between "LLM generates text" and "agent autonomously solves problems" is where most production implementations fail. Understanding this spectrum helps you:

- **Right-size your architecture**: Don't build a full agent when a single LLM call with structured output suffices
- **Manage risk**: Higher autonomy requires more guardrails, monitoring, and fallback logic
- **Optimize costs**: Each additional iteration in an agent loop multiplies your token spend
- **Set realistic expectations**: Agents excel at some tasks but struggle with others—knowing the boundaries prevents costly failures

## Technical Components

### 1. Tool Integration Layer

**Technical Explanation:**

Tools (also called functions or plugins) are the agent's interface to the outside world. Modern LLMs support "function calling" where the model outputs structured JSON specifying which function to call and with what arguments, rather than executing the function itself.

**Practical Implementation:**

```python
from typing import Any, Callable, Dict, List
from dataclasses import dataclass
import inspect
import json

@dataclass
class Tool:
    """Wrapper for agent-executable functions"""
    name: str
    description: str
    function: Callable
    parameters_schema: dict
    
    @classmethod
    def from_function(cls, func: Callable) -> 'Tool':
        """Auto-generate tool definition from Python function"""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or "No description"
        
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_type = "string"  # Simplified - real version would inspect types
            parameters["properties"][param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return cls(
            name=func.__name__,
            description=doc,
            function=func,
            parameters_schema=parameters
        )
    
    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }
        }
    
    def execute(self, arguments: dict) -> Any:
        """Execute tool with validation"""
        try:
            return self.function(**arguments)
        except TypeError as e:
            raise ValueError(f"Invalid arguments for {self.name}: {e}")

# Example tools
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get current weather for a location"""
    # Simulated API call
    return {
        "location": location,
        "temperature": 22,
        "units": units,
        "condition": "sunny"
    }

def calculate(expression: str) -> float:
    """Safely evaluate mathematical expression"""
    # In production, use a safe expression evaluator
    allowed_chars = set("0123456789+-*/().")
    if not set(expression).issubset(allowed_chars):
        raise ValueError("Invalid characters in expression")
    return eval(expression)

# Register tools
tools = [
    Tool.from_function(get_weather),
    Tool.from_function(calculate)
]
```

**Real Constraints:**

- **Tool descriptions are critical**: The LLM decides which tool to use based solely on descriptions. Vague descriptions = wrong tool selections.
- **Parameter validation happens twice**: LLM generates arguments, but you must validate before execution (LLMs make mistakes).
- **Tool execution can fail**: Network timeouts, API errors, invalid data. Agent must handle gracefully.
- **Cost scales with tool count**: More tools = larger context = higher token costs per decision.

**Trade-offs:**

Give agents fewer, well-designed tools rather than many narrow ones. A `search_database(query, filters)` tool is better than separate `search_by_name()`, `search_by_date()`, etc.

### 2. Decision Loop Architecture

**Technical Explanation:**

The agent decision loop is a ReAct (Reasoning + Acting) pattern where the agent alternates between thinking about what to do and taking actions. Each iteration consumes tokens and adds latency.

**Practical Implementation:**

```python
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass
import time

class StepType(Enum):
    REASONING = "reasoning"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"

@dataclass
class AgentStep:
    """Single step in agent execution"""
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class AgentExecutor:
    """Manages agent decision loop with proper limits and monitoring"""
    
    def __init__(
        self,
        llm_client,
        tools: List[Tool],
        max_iterations: int = 10,
        max_execution_time: float = 30.0
    ):
        self.llm = llm_client
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
    
    def execute(self, task: str) -> dict:
        """Execute agent loop with monitoring"""
        start_time = time.time()
        steps: List[AgentStep] = []
        
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task}
        ]
        
        for iteration in range(self.max_iterations):
            # Check timeout
            if time.time() - start_time > self.max_execution_time:
                return self._build_result(
                    steps, 
                    success=False,
                    error="Execution timeout"
                )
            
            # Get LLM decision
            try:
                response = self.llm.complete(
                    messages=messages,
                    tools=[t.to_openai_format() for t in self.tools.values()],
                    temperature=0.1  # Lower temp for more consistent decisions
                )
            except Exception as e:
                return self._build_result(
                    steps,
                    success=False,
                    error=f"LLM error: {str(e)}"
                )
            
            # Process response
            if response.tool_calls:
                # Agent wants to use tool(s)
                for tool_call in response.tool_calls:
                    action_step = AgentStep(
                        step_type=StepType.ACTION,
                        content=f"Using {tool_call.name}",
                        tool_name=tool_call.name,
                        tool_args=tool_call.arguments
                    )
                    steps.append(action_step)
                    
                    # Execute tool
                    observation = self._execute_tool_safely(
                        tool_call.name,
                        tool_call.arguments
                    )
                    
                    obs_step = AgentStep(
                        step_type=StepType.OBSERVATION,
                        content=json.dumps(observation)
                    )
                    steps.append(obs_step)
                    
                    # Add to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(observation),
                        "tool_call_id": tool_call.id
                    })
            else:
                # Agent has final answer
                answer_step = AgentStep(
                    step_type=StepType.ANSWER,
                    content=response.content
                )
                steps.append(answer_step)
                
                return self._build_result(
                    steps,
                    success=True,
                    answer=response.content
                )
        
        return self._build_result(
            steps,
            success=False,
            error=f"Max iterations ({self.max_iterations}) reached"
        )
    
    def _execute_tool_safely(self, tool_name: str, arguments: dict) -> dict:
        """Execute tool with error handling"""
        if tool_name not in self.tools:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            result = self.tools[tool_name].execute(arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"error": str(e)}
    
    def _build_result(
        self,
        steps: List[AgentStep],
        success: bool,
        answer: Optional[str] = None,
        error: Optional[str] = None
    ) -> dict:
        """Build structured execution result"""
        return {
            "success": success,
            "answer": answer,
            "error": error,
            "steps": steps,
            "iterations": len([s for s in steps if s.step_type == StepType.ACTION]),
            "execution_time": steps[-1].timestamp - steps[0].timestamp if steps else 0
        }
    
    def _system_prompt(self) -> str:
        return """You are an agent that solves tasks using available tools.

For each task:
1. Think about what information you need
2. Use tools to gather that information
3. Reason about the results
4. Either use more tools or provide final answer

Be concise and efficient."""
```

**Real Constraints:**

- **Every iteration adds 2-5 seconds**: LLM inference isn't instant. A 5-iteration agent takes 10-25 seconds.
- **Token costs multiply**: Each iteration includes full conversation history. A 5-iteration loop might use 10x tokens of a single call.
- **Error propagation**: If iteration 3 fails, iterations 4-5 work with bad data. Early validation is critical.

**Trade-offs:**

Lower `max_iterations` prevents runaway costs but may cause premature termination. Monitor actual iteration counts in production to tune this value.

### 3. State Management & Memory

**Technical Explanation:**

Agents need to track conversation history, tool results, and intermediate state. Unlike stateless API calls, agent state grows with each iteration and must be managed carefully to avoid context overflow.

**Practical Implementation:**

```python
from typing import List, Dict, Optional