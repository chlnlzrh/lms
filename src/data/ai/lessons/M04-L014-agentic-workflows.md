# Agentic Workflows: Building Systems That Think and Act

## Core Concepts

### What Are Agentic Workflows?

An agentic workflow is a system where an LLM dynamically determines its own execution path—deciding what actions to take, in what order, and when to stop—rather than following a predetermined sequence of operations. Unlike traditional workflows where you hardcode decision trees, agentic systems leverage the LLM's reasoning capabilities to navigate complex tasks autonomously.

### The Fundamental Shift

**Traditional Workflow:**
```python
def traditional_research_workflow(topic: str) -> str:
    """Hardcoded sequence with fixed branching logic."""
    # Step 1: Always search
    search_results = search_api(topic)
    
    # Step 2: Always extract from first 3 results
    extracts = [extract_text(url) for url in search_results[:3]]
    
    # Step 3: Always summarize
    summary = summarize(extracts)
    
    # Step 4: Fixed quality check
    if len(summary) < 100:
        summary = "Need more information"
    
    return summary
```

**Agentic Workflow:**
```python
from typing import List, Dict, Callable
import json

def agentic_research_workflow(topic: str, llm_client) -> str:
    """LLM decides what to do at each step based on progress."""
    
    tools = {
        "search": search_api,
        "extract": extract_text,
        "summarize": summarize,
        "fact_check": verify_claims
    }
    
    context = {"topic": topic, "findings": []}
    max_iterations = 10
    
    for iteration in range(max_iterations):
        # LLM examines current state and decides next action
        prompt = f"""
        Task: Research {topic}
        Current findings: {json.dumps(context['findings'])}
        
        Available tools: {list(tools.keys())}
        
        What should you do next? Respond in JSON:
        {{"action": "tool_name", "params": {{}}, "reasoning": "why"}}
        
        Or if complete: {{"action": "finish", "final_answer": "..."}}
        """
        
        decision = llm_client.generate(prompt)
        action_data = json.loads(decision)
        
        if action_data["action"] == "finish":
            return action_data["final_answer"]
        
        # Execute chosen tool
        tool_fn = tools[action_data["action"]]
        result = tool_fn(**action_data["params"])
        context["findings"].append({
            "action": action_data["action"],
            "result": result,
            "reasoning": action_data["reasoning"]
        })
    
    return "Max iterations reached"
```

The traditional approach is a rigid pipeline. The agentic approach lets the LLM assess the situation at each step: "Do I have enough information? Should I verify this claim? Should I search for more specific details?"

### Why This Matters Now

Three technical developments make agentic workflows practical:

1. **Function calling APIs**: LLMs can now reliably output structured JSON that maps to tool invocations
2. **Extended context windows**: 128K+ token contexts let agents maintain rich state across many iterations
3. **Improved reasoning**: Models like GPT-4 and Claude can perform multi-step reasoning without immediately degrading into nonsense

The trade-off: you exchange predictability for capability. Agentic workflows are non-deterministic, harder to debug, and more expensive (more LLM calls), but they handle complexity that would require thousands of lines of hardcoded logic.

### Key Mental Model Shift

Stop thinking: "How do I break this into steps?"  
Start thinking: "What tools does an intelligent agent need to solve this?"

Traditional programming: You are the intelligence, code is the tool.  
Agentic systems: LLM is the intelligence, your code provides the tools.

## Technical Components

### 1. The Agent Loop

The core of any agentic system is the perception-decision-action loop:

```python
from typing import Protocol, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ToolResult:
    """Structured tool execution result."""
    success: bool
    data: Any
    error: Optional[str] = None
    cost_estimate: float = 0.0  # For tracking

class Tool(Protocol):
    """Interface every tool must implement."""
    name: str
    description: str
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute tool with given parameters."""
        ...

@dataclass
class AgentState:
    """Maintains agent's working memory."""
    goal: str
    observations: List[Dict[str, Any]]
    iteration: int
    max_iterations: int = 15
    total_cost: float = 0.0

class AgentLoop:
    def __init__(self, llm_client, tools: List[Tool]):
        self.llm = llm_client
        self.tools = {t.name: t for t in tools}
    
    def run(self, goal: str) -> str:
        """Execute the agent loop until completion or max iterations."""
        state = AgentState(goal=goal, observations=[], iteration=0)
        
        while state.iteration < state.max_iterations:
            # 1. PERCEIVE: Format current state for LLM
            context = self._format_context(state)
            
            # 2. DECIDE: LLM chooses next action
            decision = self._get_decision(context)
            
            # 3. ACT: Execute chosen action
            if decision["action"] == "finish":
                return self._format_final_answer(decision, state)
            
            result = self._execute_tool(decision, state)
            
            # 4. OBSERVE: Record result for next iteration
            state.observations.append({
                "action": decision["action"],
                "input": decision.get("params", {}),
                "result": result.data,
                "success": result.success
            })
            state.total_cost += result.cost_estimate
            state.iteration += 1
        
        return f"Failed to complete in {state.max_iterations} iterations"
    
    def _format_context(self, state: AgentState) -> str:
        """Convert state to prompt for LLM."""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        history = "\n".join([
            f"Step {i+1}: {obs['action']} -> {obs['result']}"
            for i, obs in enumerate(state.observations[-5:])  # Last 5 only
        ])
        
        return f"""Goal: {state.goal}

Available tools:
{tools_desc}

Previous actions:
{history}

Decide next action. Respond in JSON:
{{"action": "tool_name", "params": {{"key": "value"}}, "reasoning": "why"}}

Or finish: {{"action": "finish", "answer": "final result"}}"""
    
    def _get_decision(self, context: str) -> Dict[str, Any]:
        """Call LLM to get next action decision."""
        response = self.llm.generate(
            context,
            temperature=0.2,  # Lower temp for more consistent decisions
            max_tokens=500
        )
        return json.loads(response)
    
    def _execute_tool(self, decision: Dict, state: AgentState) -> ToolResult:
        """Execute the tool specified in decision."""
        tool_name = decision["action"]
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}"
            )
        
        tool = self.tools[tool_name]
        try:
            return tool.execute(**decision.get("params", {}))
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
    
    def _format_final_answer(self, decision: Dict, state: AgentState) -> str:
        """Format final answer with metadata."""
        return f"""{decision['answer']}

[Completed in {state.iteration} steps, estimated cost: ${state.total_cost:.4f}]"""
```

**Practical Implications:**

- **State management**: Keep `observations` bounded (last 5-10) or you'll exceed context limits
- **Error handling**: Always wrap tool execution; failed tools shouldn't crash the agent
- **Cost tracking**: Each iteration costs money; enforce `max_iterations` strictly
- **Temperature tuning**: Lower (0.1-0.3) for consistent tool selection, higher (0.7+) for creative problem-solving

**Real Constraints:**

The agent loop is where costs spiral. A 10-iteration loop with 2K token prompts = 20K tokens input. At $10/1M tokens, that's $0.20 per query. Multiply by 1000 users/day = $200/day. Monitor this religiously.

### 2. Tool Interfaces

Tools are how agents interact with the world. A well-designed tool interface is critical:

```python
from typing import Literal, get_args
import requests

class WebSearchTool:
    """Search the web for information."""
    
    name = "web_search"
    description = """Search the web for current information.
    Parameters:
        query (str): Search query
        num_results (int): Number of results (1-10)
    Returns: List of {title, snippet, url}"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """Execute web search."""
        try:
            # Using a search API (pseudocode)
            response = requests.get(
                "https://api.search.example/search",
                params={"q": query, "count": num_results},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            response.raise_for_status()
            
            results = [
                {
                    "title": r["title"],
                    "snippet": r["snippet"][:200],  # Limit length
                    "url": r["url"]
                }
                for r in response.json()["results"]
            ]
            
            return ToolResult(
                success=True,
                data=results,
                cost_estimate=0.002  # $0.002 per search
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

class CalculatorTool:
    """Perform mathematical calculations safely."""
    
    name = "calculator"
    description = """Evaluate mathematical expressions.
    Parameters:
        expression (str): Math expression (e.g., "2 + 2", "sqrt(16)")
    Returns: Numeric result"""
    
    def execute(self, expression: str) -> ToolResult:
        """Safely evaluate math expression."""
        import ast
        import operator
        import math
        
        # Safe evaluation: only allow specific operations
        allowed_operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        allowed_functions = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "log": math.log,
        }
        
        try:
            # Parse and validate AST
            tree = ast.parse(expression, mode='eval')
            
            # Validate only safe operations used
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if node.func.id not in allowed_functions:
                        raise ValueError(f"Function {node.func.id} not allowed")
            
            # Evaluate
            result = self._eval_node(tree.body, allowed_operations, allowed_functions)
            
            return ToolResult(
                success=True,
                data=float(result),
                cost_estimate=0.0  # Free
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Calculation error: {e}")
    
    def _eval_node(self, node, ops, funcs):
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](
                self._eval_node(node.left, ops, funcs),
                self._eval_node(node.right, ops, funcs)
            )
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](self._eval_node(node.operand, ops, funcs))
        elif isinstance(node, ast.Call):
            func = funcs[node.func.id]
            args = [self._eval_node(arg, ops, funcs) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")

class MemoryTool:
    """Store and retrieve information across agent iterations."""
    
    name = "memory"
    description = """Store or retrieve information from persistent memory.
    Parameters:
        operation (str): "store" or "retrieve"
        key (str): Memory key
        value (str): Value to store (only for "store" operation)
    Returns: Success confirmation or retrieved value"""
    
    def __init__(self):
        self.storage: Dict[str, str] = {}
    
    def execute(
        self,
        operation: Literal["store", "retrieve"],
        key: str,
        value: Optional[str] = None
    ) -> ToolResult:
        """Execute memory operation."""
        try:
            if operation == "store":
                if value is None:
                    raise ValueError("value required for store operation")
                self.storage[key] = value
                return ToolResult(
                    success=True,
                    data=f"Stored '{key}'",
                    cost_estimate=0.0
                )
            
            elif operation == "retrieve":
                if key in self.storage:
                    return ToolResult(
                        success=True,
                        data=self.storage[key],
                        cost_estimate=0.0
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Key '{key}' not found"
                    )
            
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
```

**Design Principles:**

1. **Clear descriptions**: The LLM only knows tools through their descriptions. Be explicit about parameters, return types, and use cases.
2. **Bounded execution**: Set timeouts, rate limits, and result size limits. Never trust the LLM to be reasonable.
3. **Consistent return format**: Always return `ToolResult` with success/failure and structured data.
4. **Cost transparency**: Track and report costs so you can optimize later.

**Trade-offs:**

- **Too many tools**: LLM gets confused choosing, performance degrades beyond ~20 tools
- **Too few tools**: Agent can't solve complex tasks, makes inappropriate tool choices
- **Tool granularity**: Prefer atomic operations (search, calculate) over composite ones (search-and-summarize)

### 3. Reasoning Strategies

How you prompt the agent dramatically affects performance. Three critical patterns:

**ReAct (Reasoning + Acting):**

```python
def react_prompt_template(state: AgentState) -> str:
    """Chain-of-thought reasoning before each action."""
    return f"""Goal: {state.goal}

Observations so far:
{format_observations(state.observations)}

Think step-by-step:
Thought: What do I need to do next?
Action: {{"action": "tool_name", "params": {{}}}}

Or if done:
Thought: I have everything needed to answer.
Action: {{"action": "finish", "answer": "..."}}

Your turn:
Thought:"""

# The LLM naturally continues with reasoning, then specifies action
```