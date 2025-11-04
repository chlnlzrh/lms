# Agent Frameworks & Orchestration

## Core Concepts

Agent frameworks transform LLMs from single-shot text generators into autonomous systems that can plan, execute tools, reason about results, and maintain state across multi-step workflows. While a basic LLM call is stateless and synchronous, agent frameworks add control flow, memory, tool execution, and decision-making layers.

### Traditional vs. Agent-Based Approach

**Traditional API Integration:**
```python
import anthropic

def get_weather_report(city: str) -> str:
    """Traditional approach: sequential, hardcoded logic"""
    client = anthropic.Anthropic()
    
    # Step 1: Get coordinates (hardcoded call)
    coord_prompt = f"What are the coordinates of {city}? Return as 'lat,lon'"
    coord_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": coord_prompt}]
    )
    coords = coord_response.content[0].text.strip()
    
    # Step 2: Mock weather API call (would be real in production)
    weather_data = f"Temperature: 72°F, Conditions: Sunny"
    
    # Step 3: Format response (another hardcoded call)
    format_prompt = f"Summarize this weather: {weather_data}"
    format_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": format_prompt}]
    )
    
    return format_response.content[0].text
```

**Agent Framework Approach:**
```python
from typing import Any, Callable
import anthropic
import json

class Agent:
    """Agent that autonomously decides which tools to use"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.tools: list[dict] = []
        self.tool_functions: dict[str, Callable] = {}
        
    def register_tool(self, name: str, description: str, 
                     parameters: dict, function: Callable):
        """Register a tool the agent can use"""
        self.tools.append({
            "name": name,
            "description": description,
            "input_schema": parameters
        })
        self.tool_functions[name] = function
    
    def run(self, user_message: str, max_iterations: int = 5) -> str:
        """Run agent loop: think -> act -> observe -> repeat"""
        messages = [{"role": "user", "content": user_message}]
        
        for iteration in range(max_iterations):
            # Agent decides what to do
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=self.tools,
                messages=messages
            )
            
            # If agent is done thinking, return final answer
            if response.stop_reason == "end_turn":
                final_text = next(
                    (block.text for block in response.content 
                     if hasattr(block, "text")), 
                    "No response"
                )
                return final_text
            
            # Agent wants to use a tool
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                
                # Execute all requested tools
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        
                        # Execute the tool
                        result = self.tool_functions[tool_name](**tool_input)
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        })
                
                # Send tool results back to agent
                messages.append({"role": "user", "content": tool_results})
            else:
                break
        
        return "Max iterations reached without completion"

# Define tools
def get_coordinates(city: str) -> dict:
    """Mock geocoding - in production, call real API"""
    coords_db = {
        "Seattle": {"lat": 47.6062, "lon": -122.3321},
        "Portland": {"lat": 45.5152, "lon": -122.6784},
        "San Francisco": {"lat": 37.7749, "lon": -122.4194}
    }
    return coords_db.get(city, {"lat": 0, "lon": 0})

def get_weather(lat: float, lon: float) -> dict:
    """Mock weather API - in production, call real service"""
    return {
        "temperature": 72,
        "conditions": "Sunny",
        "humidity": 65,
        "wind_speed": 8
    }

# Create and configure agent
agent = Agent()

agent.register_tool(
    name="get_coordinates",
    description="Get latitude and longitude for a city name",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    },
    function=get_coordinates
)

agent.register_tool(
    name="get_weather",
    description="Get current weather for coordinates",
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude"},
            "lon": {"type": "number", "description": "Longitude"}
        },
        "required": ["lat", "lon"]
    },
    function=get_weather
)

# Agent autonomously plans and executes
result = agent.run("What's the weather like in Seattle?")
print(result)
```

### Key Differences That Matter

1. **Autonomous Planning**: The agent decides which tools to use and in what order, rather than following hardcoded logic
2. **State Management**: The agent maintains conversation context across multiple tool calls
3. **Error Recovery**: Agents can retry with different approaches if initial attempts fail
4. **Composability**: Tools are modular and reusable across different queries

### Why This Matters Now

Agent frameworks enable **compound AI systems** that exceed the capabilities of any single model call. As models become more reliable at tool use and reasoning, the bottleneck shifts from model capability to orchestration architecture. Understanding agent patterns is critical because:

- **Cost efficiency**: Agents can route simple queries to small models, complex ones to large models
- **Reliability**: Multi-step verification and self-correction patterns reduce hallucinations
- **Integration**: Agents bridge LLMs with existing systems (databases, APIs, internal tools)
- **Scalability**: Well-designed agent loops handle variable complexity without code changes

## Technical Components

### 1. Tool Definition & Registration

Tools are the interface between LLMs and external functionality. Proper tool design directly impacts agent reliability and cost.

**Technical Implementation:**
```python
from typing import Literal, Optional
from pydantic import BaseModel, Field

class ToolParameter(BaseModel):
    """Type-safe tool parameter definition"""
    name: str
    type: Literal["string", "number", "boolean", "array", "object"]
    description: str
    required: bool = True
    enum: Optional[list[str]] = None

class Tool:
    """Structured tool definition with validation"""
    
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: list[ToolParameter]
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters
        
    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def execute(self, **kwargs) -> Any:
        """Execute tool with error handling"""
        try:
            return self.function(**kwargs)
        except Exception as e:
            return {"error": str(e), "type": type(e).__name__}

# Example: Database query tool
def query_database(
    table: str, 
    filters: dict, 
    limit: int = 10
) -> list[dict]:
    """Mock database query - replace with real DB"""
    if table == "users":
        return [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ][:limit]
    return []

db_tool = Tool(
    name="query_database",
    description="Query the database with filters. Returns matching records.",
    function=query_database,
    parameters=[
        ToolParameter(
            name="table",
            type="string",
            description="Table name to query",
            enum=["users", "orders", "products"]
        ),
        ToolParameter(
            name="filters",
            type="object",
            description="Key-value pairs to filter results"
        ),
        ToolParameter(
            name="limit",
            type="number",
            description="Maximum results to return",
            required=False
        )
    ]
)
```

**Practical Implications:**
- **Specific descriptions**: Vague tool descriptions cause agents to misuse tools. "Query database" vs "Query the users table to find account information by email or user ID"
- **Type constraints**: Enums and schemas prevent invalid tool calls, reducing wasted tokens
- **Error handling**: Tools must return structured errors so agents can retry intelligently

**Trade-offs:**
- More specific tools = more reliable but less flexible
- Type-strict schemas = fewer errors but more engineering overhead
- Synchronous execution = simpler but blocks on slow operations

### 2. Control Flow Patterns

Agent loops require careful control flow to balance autonomy with safety and cost.

**Technical Implementation:**
```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class StopReason(Enum):
    MAX_ITERATIONS = "max_iterations"
    EXPLICIT_STOP = "explicit_stop"
    ERROR = "error"
    SUCCESS = "success"

@dataclass
class AgentResult:
    output: str
    stop_reason: StopReason
    iterations: int
    total_tokens: int
    tool_calls: list[dict]

class OrchestrationLoop:
    """Advanced control flow with safety mechanisms"""
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        tools: list[Tool],
        max_iterations: int = 10,
        max_tokens_per_call: int = 4096,
        max_total_tokens: int = 100000
    ):
        self.client = client
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.tool_schemas = [tool.to_anthropic_schema() for tool in tools]
        self.max_iterations = max_iterations
        self.max_tokens_per_call = max_tokens_per_call
        self.max_total_tokens = max_total_tokens
    
    def execute(self, user_message: str) -> AgentResult:
        """Execute agent loop with comprehensive tracking"""
        messages = [{"role": "user", "content": user_message}]
        total_tokens = 0
        tool_calls = []
        
        for iteration in range(self.max_iterations):
            # Check token budget
            if total_tokens >= self.max_total_tokens:
                return AgentResult(
                    output="Token budget exceeded",
                    stop_reason=StopReason.ERROR,
                    iterations=iteration,
                    total_tokens=total_tokens,
                    tool_calls=tool_calls
                )
            
            # Get agent response
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens_per_call,
                    tools=self.tool_schemas,
                    messages=messages
                )
            except Exception as e:
                return AgentResult(
                    output=f"API error: {str(e)}",
                    stop_reason=StopReason.ERROR,
                    iterations=iteration,
                    total_tokens=total_tokens,
                    tool_calls=tool_calls
                )
            
            # Track token usage
            total_tokens += response.usage.input_tokens + response.usage.output_tokens
            
            # Agent completed task
            if response.stop_reason == "end_turn":
                final_text = next(
                    (block.text for block in response.content 
                     if hasattr(block, "text")),
                    ""
                )
                return AgentResult(
                    output=final_text,
                    stop_reason=StopReason.SUCCESS,
                    iterations=iteration + 1,
                    total_tokens=total_tokens,
                    tool_calls=tool_calls
                )
            
            # Execute tools
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_call = {
                            "name": block.name,
                            "input": block.input,
                            "iteration": iteration
                        }
                        tool_calls.append(tool_call)
                        
                        # Execute with timeout protection
                        tool = self.tools.get(block.name)
                        if tool:
                            result = tool.execute(**block.input)
                        else:
                            result = {"error": f"Unknown tool: {block.name}"}
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })
                
                messages.append({"role": "user", "content": tool_results})
        
        # Max iterations reached
        return AgentResult(
            output="Maximum iterations reached without completion",
            stop_reason=StopReason.MAX_ITERATIONS,
            iterations=self.max_iterations,
            total_tokens=total_tokens,
            tool_calls=tool_calls
        )
```

**Critical Patterns:**

1. **Token budgets**: Prevent runaway costs from infinite loops
2. **Iteration limits**: Stop agents that can't converge to a solution
3. **Result tracking**: Log every tool call for debugging and optimization
4. **Graceful degradation**: Return partial results rather than failing completely

**Common Failure Modes:**
- **Infinite loops**: Agent repeatedly calls same tool with same inputs (add deduplication)
- **Token exhaustion**: Complex queries exceed budget before completion (implement streaming or chunking)
- **Tool errors**: One failing tool blocks entire workflow (implement retry logic with backoff)

### 3. Memory & State Management

Agents need context beyond the immediate conversation to maintain coherence and efficiency.

**Technical Implementation:**
```python
from collections import deque
from datetime import datetime
from typing import Deque

class ConversationMemory:
    """Manage conversation history with summarization"""
    
    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 8000,
        summarization_client: Optional[anthropic.Anthropic] = None
    ):
        self.messages: Deque[dict] = deque(maxlen=max_messages)
        self.max_tokens = max_tokens
        self.summarization_client = summarization_client
        self.summary: Optional[str] = None
        self.metadata: dict = {"created_at": datetime.utcnow().isoformat()}
    
    def add_message