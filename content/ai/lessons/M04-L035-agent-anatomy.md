# Agent Anatomy: Understanding the Core Components of LLM Agents

## Core Concepts

### What Is an LLM Agent?

An LLM agent is a software system that uses a large language model as its reasoning engine to autonomously work toward goals by breaking down tasks, making decisions, taking actions, and adjusting based on feedback. Unlike a simple API call to an LLM that returns a single response, an agent operates in a loop: observe → think → act → repeat.

Think of the difference this way:

**Traditional LLM Integration (Single Call):**
```python
from anthropic import Anthropic

client = Anthropic(api_key="your-key")

# One shot: user asks, model responds, done
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
)

print(response.content[0].text)
# Output: "I don't have access to real-time weather data..."
```

**Agent-Based Approach (Agentic Loop):**
```python
from anthropic import Anthropic
import json
import requests
from typing import List, Dict, Any

client = Anthropic(api_key="your-key")

# Define tools the agent can use
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
]

def get_weather(city: str) -> Dict[str, Any]:
    """Simulated weather API call"""
    # In production, this would call a real weather API
    return {
        "city": city,
        "temperature": 18,
        "condition": "partly cloudy",
        "humidity": 65
    }

def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    # Agent loop: model can make multiple tool calls until task is complete
    while True:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Check if model wants to use a tool
        if response.stop_reason == "tool_use":
            # Extract tool use from response
            tool_use = next(block for block in response.content if block.type == "tool_use")
            tool_name = tool_use.name
            tool_input = tool_use.input
            
            # Execute the tool
            if tool_name == "get_weather":
                result = get_weather(tool_input["city"])
            
            # Add assistant response and tool result to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result)
                    }
                ]
            })
            # Loop continues - model can now use this information
        else:
            # Model has finished, extract final text response
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                ""
            )
            return final_text

# Now the agent can actually answer the question
result = run_agent("What's the weather in Paris?")
print(result)
# Output: "The current weather in Paris is partly cloudy with a temperature 
# of 18°C and 65% humidity."
```

### Key Engineering Insights

**1. Agents are control flow systems, not models:** The LLM is the decision-making component, but the agent is the orchestration layer. You're building a system that happens to use an LLM for reasoning, not just wrapping an API.

**2. Non-deterministic execution paths:** Unlike traditional software where you control the flow, agents choose their own path through available actions. This requires different debugging and testing strategies.

**3. The agent loop is expensive:** Every iteration costs tokens, time, and money. A simple task might take 5-10 LLM calls. Production agents need careful cost management.

### Why This Matters Now

**Agents solve the "capability gap":** LLMs are brilliant at language but can't access databases, APIs, or real-time information. Agents bridge this gap by giving models the ability to take actions in the world.

**The shift from scripting to delegation:** Instead of writing explicit code for every scenario, you define capabilities (tools) and let the agent figure out how to combine them. This is powerful for variable workflows where users might ask anything.

**Real cost implications:** A single agent interaction that takes 8 tool calls with 2K tokens each = 16K tokens. At scale, understanding agent anatomy directly impacts your infrastructure budget.

## Technical Components

### 1. The Reasoning Engine (LLM Core)

**Technical Explanation:**

The LLM serves as the agent's "brain" - it interprets user requests, decides which tools to use, reasons about results, and determines when the task is complete. Modern LLMs with tool/function calling capabilities can output structured data requesting specific actions.

**Practical Implications:**

- **Model selection matters:** Not all LLMs are equally good at agentic reasoning. Models trained on tool use (like Claude 3.5, GPT-4, Gemini) significantly outperform base models.
- **Context window is your working memory:** Every previous message, tool call, and result consumes context. Long agent sessions hit limits.
- **Reasoning quality varies:** Sometimes agents make poor tool choices or miss obvious solutions.

**Real Constraints:**

```python
from anthropic import Anthropic
from typing import Optional
import time

class AgentMetrics:
    def __init__(self):
        self.total_tokens = 0
        self.tool_calls = 0
        self.iterations = 0
        self.start_time = time.time()
    
    def log_iteration(self, input_tokens: int, output_tokens: int, tool_called: bool):
        self.total_tokens += input_tokens + output_tokens
        self.iterations += 1
        if tool_called:
            self.tool_calls += 1
    
    def report(self) -> dict:
        duration = time.time() - self.start_time
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "duration_seconds": round(duration, 2),
            "estimated_cost_usd": (self.total_tokens / 1_000_000) * 3.0  # Rough estimate
        }

def run_agent_with_metrics(user_message: str, tools: list, max_iterations: int = 10) -> tuple[str, dict]:
    """Agent with metrics and safety limits"""
    client = Anthropic(api_key="your-key")
    metrics = AgentMetrics()
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        metrics.log_iteration(
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.stop_reason == "tool_use"
        )
        
        if response.stop_reason == "tool_use":
            # Tool execution logic here (simplified)
            tool_use = next(block for block in response.content if block.type == "tool_use")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "{}"}]
            })
        else:
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                ""
            )
            return final_text, metrics.report()
    
    return "Max iterations reached", metrics.report()

# Measuring real agent cost
result, metrics = run_agent_with_metrics("What's the weather in Paris?", tools)
print(f"Result: {result}")
print(f"Cost: {metrics}")
# Example output:
# Cost: {'iterations': 3, 'tool_calls': 1, 'total_tokens': 4200, 
#        'duration_seconds': 2.3, 'estimated_cost_usd': 0.0126}
```

### 2. Tools/Functions (Action Interface)

**Technical Explanation:**

Tools are functions the agent can invoke. Each tool needs a schema (name, description, parameters) that the LLM uses to decide when and how to call it. When an LLM decides to use a tool, it returns structured output matching the schema. Your code executes the actual function and returns results to the LLM.

**Practical Implications:**

- **Tool descriptions are prompts:** The LLM only knows about tools through their descriptions. Poorly described tools lead to misuse or being ignored.
- **Type safety matters:** Tool schemas enforce structure, but you still need validation. LLMs can hallucinate invalid parameter values.
- **Tool granularity is a design choice:** One `database_query` tool with SQL parameters vs. separate `get_user`, `update_order` tools. Granular tools are safer but require more reasoning.

**Concrete Example:**

```python
from typing import List, Dict, Any
import re

def create_search_tool(description_quality: str = "good") -> Dict[str, Any]:
    """Demonstrating how tool descriptions affect agent behavior"""
    
    if description_quality == "poor":
        return {
            "name": "search",
            "description": "Search for things",  # Vague!
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    
    elif description_quality == "good":
        return {
            "name": "search_knowledge_base",
            "description": "Search technical documentation and knowledge base. Use for finding information about APIs, architecture, or technical procedures. Returns relevant document excerpts with metadata.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - be specific (e.g., 'rate limiting configuration' not just 'limits')"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["api", "architecture", "procedures", "troubleshooting"],
                        "description": "Document category to narrow search"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (1-10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    
    return {}

def validate_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> tuple[bool, str]:
    """Validate tool inputs beyond schema (LLMs can be creative)"""
    
    if tool_name == "search_knowledge_base":
        query = tool_input.get("query", "")
        
        # Check for SQL injection patterns (if tool queries database)
        sql_patterns = [r";\s*DROP", r"--", r"UNION\s+SELECT"]
        for pattern in sql_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains potentially unsafe patterns"
        
        # Check max_results bounds
        max_results = tool_input.get("max_results", 5)
        if not 1 <= max_results <= 10:
            return False, f"max_results must be 1-10, got {max_results}"
        
        return True, ""
    
    return True, ""

# Example usage
tool_input_from_llm = {
    "query": "rate limiting; DROP TABLE users--",
    "category": "api",
    "max_results": 100
}

valid, error = validate_tool_input("search_knowledge_base", tool_input_from_llm)
if not valid:
    print(f"Tool input validation failed: {error}")
    # Return error to LLM, don't execute tool
```

### 3. Memory/State Management

**Technical Explanation:**

Agents need to track conversation history (what's been said), tool execution history (what actions were taken and their results), and sometimes long-term memory (facts to remember across sessions). This state grows with every iteration and must be managed within context window limits.

**Practical Implications:**

- **Context window management is critical:** A conversation with 10 tool calls can easily exceed 20K tokens. You need strategies to compress or truncate history.
- **Tool results can be huge:** An API might return 50KB of JSON, but the agent only needs a summary. Preprocessing results saves tokens.
- **State affects reproducibility:** Same user query with different conversation history produces different results.

**Real Constraints:**

```python
from typing import List, Dict, Any
from collections import deque
import json

class AgentMemory:
    def __init__(self, max_messages: int = 20, max_tokens_per_result: int = 500):
        self.messages: deque = deque(maxlen=max_messages)
        self.max_tokens_per_result = max_tokens_per_result
        self.system_message = {
            "role": "system",
            "content": "You are a helpful assistant with access to tools."
        }
    
    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: List[Dict[str, Any]]):
        self.messages.append({"role": "assistant", "content": content})
    
    def add_tool_result(self, tool_use_id: str, result: Any):
        """Add tool result, compressing if too large"""
        result_str = json.dumps(result) if not isinstance(result, str) else result
        
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(result_str) / 4
        
        if estimated_tokens > self.max_tokens_per_result:
            # Compress large results
            compressed = self._compress_result(result, self.max_tokens_per_result)
            result_str = f"[Result compressed due to size]\n{compressed}"
        
        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_str
                }
            ]
        })
    
    def _compress_result(self, result: Any, max_tokens: int) -> str:
        """Compress large results while preserving key information"""
        result_str = json.dumps(result, indent=2) if not isinstance(result, str) else result
        
        # Simple truncation with metadata
        max_chars = max_tokens * 4
        if len(result_str) > max_chars:
            truncated = result_str[:max_chars]
            return f"{truncated}\n... [truncated {len(result_str) - max_chars} characters]"
        
        return result_str
    
    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """Get messages formatted for API call"""
        return list(self.messages)
    
    def estimate_token_usage(self) -> int:
        """Rough token estimation for current context"""
        total_chars = sum(
            len(json.dumps(msg)) for msg in self.messages
        )
        return total_chars // 4

# Usage example
memory = AgentMemory(max_messages=20, max_tokens_per_result=500)

memory.add_user_message("Search for Python documentation on asyncio")

# Simulate assistant requesting tool use
memory.add_assistant_message([
    {
        "type": "tool