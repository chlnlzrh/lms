# Declarative AI Agents: Configuration Over Code

## Core Concepts

### Technical Definition

Declarative AI agents specify *what* the agent should accomplish through configuration rather than *how* it should accomplish it through procedural code. Instead of writing explicit control flow for LLM interactions, tool selection, and response handling, you define the agent's capabilities, goals, and constraints in structured formats (JSON, YAML, or domain-specific languages). The agent framework interprets this configuration to orchestrate LLM calls, tool usage, and decision-making at runtime.

This architectural pattern separates agent behavior specification from execution logic—similar to how SQL declares desired results rather than database traversal algorithms, or how Kubernetes manifests describe desired cluster state rather than imperative deployment steps.

### Engineering Analogy: Imperative vs. Declarative

**Imperative Approach (Traditional):**

```python
from typing import List, Dict, Any
import openai
import json

class ImperativeAgent:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []
    
    def run(self, user_input: str) -> str:
        # Step 1: Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Step 2: Determine if we need tools
        classification_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Classify if this requires: weather, calculation, or general"
            }, {
                "role": "user",
                "content": user_input
            }]
        )
        intent = classification_response.choices[0].message.content
        
        # Step 3: Execute appropriate tool
        if "weather" in intent.lower():
            tool_result = self._get_weather(user_input)
            context = f"Weather data: {tool_result}"
        elif "calculation" in intent.lower():
            tool_result = self._calculate(user_input)
            context = f"Calculation result: {tool_result}"
        else:
            context = ""
        
        # Step 4: Generate final response
        if context:
            self.conversation_history.append({
                "role": "system",
                "content": context
            })
        
        final_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.conversation_history
        )
        
        response_text = final_response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        return response_text
    
    def _get_weather(self, query: str) -> str:
        # Extract location, call API, etc.
        return "Sunny, 72°F"
    
    def _calculate(self, query: str) -> str:
        # Parse expression, compute, etc.
        return "42"
```

**Declarative Approach:**

```python
from typing import List, Dict, Any, Callable
import openai
import json
from dataclasses import dataclass, field

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]

@dataclass
class AgentConfig:
    name: str
    model: str
    system_prompt: str
    tools: List[Tool] = field(default_factory=list)
    temperature: float = 0.7
    max_iterations: int = 5

class DeclarativeAgent:
    def __init__(self, config: AgentConfig, api_key: str):
        self.config = config
        self.client = openai.OpenAI(api_key=api_key)
        self.conversation_history: List[Dict[str, Any]] = [{
            "role": "system",
            "content": config.system_prompt
        }]
    
    def run(self, user_input: str) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        for iteration in range(self.config.max_iterations):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=self.conversation_history,
                tools=self._format_tools(),
                temperature=self.config.temperature
            )
            
            message = response.choices[0].message
            
            # No tool call - we're done
            if not message.tool_calls:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content
                })
                return message.content
            
            # Execute tool calls
            self.conversation_history.append(message.model_dump())
            
            for tool_call in message.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        
        return "Max iterations reached"
    
    def _format_tools(self) -> List[Dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        } for tool in self.config.tools]
    
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        tool = next((t for t in self.config.tools if t.name == name), None)
        if tool:
            return tool.function(**args)
        raise ValueError(f"Tool {name} not found")

# Agent definition as configuration
def get_weather(location: str) -> Dict[str, Any]:
    # Actual implementation
    return {"location": location, "temperature": 72, "condition": "Sunny"}

def calculate(expression: str) -> Dict[str, Any]:
    # Actual implementation
    return {"result": eval(expression)}

assistant_config = AgentConfig(
    name="general_assistant",
    model="gpt-4",
    system_prompt="You are a helpful assistant with access to weather and calculation tools.",
    tools=[
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            function=get_weather,
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        ),
        Tool(
            name="calculate",
            description="Evaluate mathematical expressions",
            function=calculate,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        )
    ],
    temperature=0.7,
    max_iterations=5
)

# Usage
agent = DeclarativeAgent(assistant_config, api_key="your-key")
result = agent.run("What's the weather in Seattle and what's 123 * 456?")
```

### Key Insights

**1. Behavior becomes data:** Agent capabilities can be version-controlled, A/B tested, and modified without code deployment. You can diff agent configurations like you diff code, enabling rapid iteration.

**2. The LLM becomes the orchestrator:** Rather than writing explicit branching logic, the LLM itself decides when and how to use tools based on the declared capabilities. This adapts to varied inputs without explicit handling.

**3. Complexity moves from control flow to configuration design:** The hard problem shifts from "how do I chain these API calls?" to "how do I describe this tool so the LLM uses it correctly?" This is a different skill set—prompt engineering over procedural logic.

**4. Testing changes fundamentally:** Instead of unit testing individual code paths, you test whether configurations produce desired behaviors across example inputs. This resembles integration testing more than unit testing.

### Why This Matters Now

The proliferation of LLM features (function calling, structured outputs, vision, etc.) makes imperative orchestration unwieldy. A single agent might need to coordinate 10+ tools, maintain conversation context, handle errors, implement retry logic, and more. Declarative patterns emerged because:

- **Function calling APIs standardized** (OpenAI, Anthropic, others): LLMs can now reliably select and invoke tools, eliminating need for custom routing logic
- **Token costs decreased 10-100x**: Complex multi-turn conversations became economically viable for production use
- **Reasoning capabilities improved**: Modern LLMs can handle sophisticated decision-making previously requiring explicit code
- **Agent frameworks matured**: Libraries like LangChain, LlamaIndex, and others provided declarative abstractions that proved production-ready

## Technical Components

### 1. Configuration Schema Design

**Technical Explanation:**

Configuration schemas define the structure and constraints of agent behavior specifications. Well-designed schemas balance expressiveness (what behaviors can be specified) with simplicity (how easily humans understand them). The schema must capture:

- Agent identity and behavior guidelines (system prompts)
- Available tools with semantic descriptions
- Execution constraints (timeouts, retry policies, cost limits)
- State management rules (what to remember, when to forget)

**Practical Implications:**

Your schema IS your API contract with future maintainers. Over-specified schemas (too many options) create cognitive overhead; under-specified schemas (too rigid) require code changes for simple tweaks. The sweet spot: parameterize what changes frequently, hardcode what's stable.

**Real Constraints:**

- **JSON Schema limitations**: Cannot express runtime dependencies ("if tool A is used, parameter B becomes required")
- **Type safety**: Python dictionaries lose type checking—consider using Pydantic for validation
- **Version migration**: Changing schemas breaks existing configurations; plan for backward compatibility
- **Secret management**: API keys and credentials must stay out of version control but remain accessible to agents

**Example - Production-Grade Schema:**

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

class ToolParameter(BaseModel):
    name: str
    type: Literal["string", "number", "boolean", "array", "object"]
    description: str
    required: bool = False
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None

class ToolDefinition(BaseModel):
    name: str = Field(..., pattern=r'^[a-z_][a-z0-9_]*$')
    description: str = Field(..., min_length=10, max_length=500)
    parameters: List[ToolParameter]
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    retry_attempts: int = Field(default=3, ge=0, le=5)
    
    @validator('description')
    def description_must_be_clear(cls, v):
        if any(word in v.lower() for word in ['maybe', 'might', 'possibly']):
            raise ValueError('Tool descriptions must be definitive, not uncertain')
        return v

class ExecutionPolicy(BaseModel):
    max_iterations: int = Field(default=5, ge=1, le=20)
    max_tool_calls_per_iteration: int = Field(default=5, ge=1, le=10)
    total_timeout_seconds: int = Field(default=300, ge=10, le=3600)
    max_tokens: int = Field(default=4000, ge=100, le=128000)

class MemoryConfig(BaseModel):
    type: Literal["none", "conversation", "sliding_window", "summary"]
    window_size: Optional[int] = Field(default=None, ge=1, le=100)
    summary_trigger_tokens: Optional[int] = None

class AgentManifest(BaseModel):
    """Complete declarative agent specification"""
    version: str = Field(default="1.0.0", pattern=r'^\d+\.\d+\.\d+$')
    name: str = Field(..., min_length=1, max_length=100)
    model: str
    system_prompt: str = Field(..., min_length=20)
    tools: List[ToolDefinition] = Field(default_factory=list)
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    memory: MemoryConfig = Field(default_factory=lambda: MemoryConfig(type="conversation"))
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('tools')
    def no_duplicate_tool_names(cls, v):
        names = [tool.name for tool in v]
        if len(names) != len(set(names)):
            raise ValueError('Tool names must be unique')
        return v

    def to_json_file(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.json(indent=2))
    
    @classmethod
    def from_json_file(cls, path: str) -> 'AgentManifest':
        with open(path, 'r') as f:
            return cls.parse_raw(f.read())

# Example usage
manifest = AgentManifest(
    name="customer_support_agent",
    model="gpt-4",
    system_prompt="You are a customer support agent. Be empathetic, clear, and solution-focused.",
    tools=[
        ToolDefinition(
            name="lookup_order",
            description="Retrieve order details by order ID from the database",
            parameters=[
                ToolParameter(name="order_id", type="string", description="Unique order identifier", required=True)
            ],
            timeout_seconds=10
        ),
        ToolDefinition(
            name="issue_refund",
            description="Process a refund for an order",
            parameters=[
                ToolParameter(name="order_id", type="string", description="Order to refund", required=True),
                ToolParameter(name="amount", type="number", description="Refund amount in USD", required=True),
                ToolParameter(name="reason", type="string", description="Reason for refund", required=True)
            ],
            timeout_seconds=30,
            retry_attempts=1  # Financial operations should not auto-retry
        )
    ],
    execution_policy=ExecutionPolicy(max_iterations=8, total_timeout_seconds=120),
    memory=MemoryConfig(type="sliding_window", window_size=10)
)

manifest.to_json_file("customer_support_agent.json")
```

### 2. Tool Registration and Discovery

**Technical Explanation:**

Tools are functions the agent can invoke. The declarative pattern requires tools to be:
1. **Self-describing**: Include metadata about parameters, return types, and semantics
2. **Isolated**: Execute independently without shared state
3. **Idempotent or clearly marked**: Agent may retry, so side effects must be managed

Tool registration maps human-readable descriptions to executable code. The LLM uses descriptions for selection; the framework uses mappings for execution.

**Practical Implications:**

Tool descriptions are prompts. Vague descriptions cause incorrect tool selection; overly specific descriptions reduce flexibility. The LLM doesn't see your code—only the description and parameter schema.

**Real Constraints:**

- **Description quality is critical**: "Get weather" performs worse than "Retrieve current weather conditions including temperature, humidity, and forecast for a specific city"
- **Parameter validation happens after LLM selection**: The LLM might call tools with invalid parameters; validate and provide clear error messages
- **Async vs sync**: Most tool frameworks assume synchronous execution; async tools require special handling

**Example - Tool Registry with Type Safety:**

```python
from typing import Callable, Any, Dict, get_type_hints
import inspect
import asyncio
from functools import wraps

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, description: str, timeout: int = 30):
        """Decorator to register functions as tools"""
        def decorator(func: Callable) -> Callable:
            # Extract type hints for automatic schema generation
            sig = inspect.signature(func)
            type