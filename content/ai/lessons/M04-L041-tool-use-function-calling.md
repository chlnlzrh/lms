# Tool Use & Function Calling: Engineering Deterministic Bridges in Stochastic Systems

## Core Concepts

Tool use—also called function calling—is a mechanism that allows language models to request execution of deterministic code during inference. Instead of the model attempting to perform calculations, query databases, or interact with external systems through text generation, it outputs structured requests that your application intercepts, executes, and feeds back as context.

### Traditional vs. Modern Approaches

**Traditional approach: Parsing model outputs**

```python
import re
from typing import Optional

def get_weather_traditional(user_query: str, llm_client) -> str:
    """Fragile pattern matching approach"""
    prompt = f"""
    User query: {user_query}
    
    If the user asks about weather, respond with:
    WEATHER_REQUEST: city_name
    
    Otherwise respond normally.
    """
    
    response = llm_client.generate(prompt)
    
    # Brittle parsing logic
    match = re.search(r'WEATHER_REQUEST:\s*(\w+)', response)
    if match:
        city = match.group(1)
        weather_data = call_weather_api(city)  # Your actual API call
        
        # Second LLM call to format response
        final_prompt = f"Weather data for {city}: {weather_data}. Format this nicely."
        return llm_client.generate(final_prompt)
    
    return response

# Problems:
# - Model may format the marker inconsistently
# - Extra tokens wasted on explaining the pattern
# - Two separate LLM calls required
# - No parameter validation
# - Hard to extend to multiple tools
```

**Modern approach: Structured function calling**

```python
from typing import Literal
from pydantic import BaseModel, Field

class WeatherRequest(BaseModel):
    city: str = Field(description="City name for weather lookup")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": WeatherRequest.model_json_schema()
    }
}]

def get_weather_modern(user_query: str, llm_client) -> str:
    """Robust structured approach"""
    messages = [{"role": "user", "content": user_query}]
    
    response = llm_client.chat(messages=messages, tools=tools)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        # Parameters are already validated JSON
        params = WeatherRequest(**tool_call.function.arguments)
        weather_data = call_weather_api(params.city, params.units)
        
        # Append tool result to conversation
        messages.append(response.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": weather_data
        })
        
        # Single final call with full context
        final_response = llm_client.chat(messages=messages, tools=tools)
        return final_response.content
    
    return response.content

# Benefits:
# - Guaranteed valid JSON structure
# - Type validation via Pydantic
# - Single conversation thread
# - Extensible to dozens of tools
# - Model trained specifically for this format
```

### Key Engineering Insights

**1. Function calling is prompt engineering made declarative**  
You're not teaching the model a protocol through examples—you're providing a schema that gets compiled into the model's constrained output space. The model has been specifically fine-tuned to recognize tool schemas and produce valid JSON matching those schemas.

**2. The model doesn't execute functions; it generates execution requests**  
This is critical for security and architecture. The model outputs a structured request; your code decides whether to execute it, how to sandbox it, and what results to return. You maintain complete control over side effects.

**3. Tool calls create conversation branches that must be merged**  
Each tool invocation creates a fork: the model's request, your execution, and the continuation. Managing this conversation state correctly is essential for multi-turn interactions and parallel tool use.

### Why This Matters Now

Modern LLM applications are moving beyond text generation to become orchestration layers. The pattern of "LLM as reasoning engine + deterministic tools for execution" has become the standard architecture because:

- **Reliability**: Deterministic operations (math, API calls, database queries) happen in deterministic code
- **Auditability**: Every tool call is logged and can be traced
- **Cost efficiency**: Offloading computation to specialized systems is orders of magnitude cheaper than token generation
- **Latency optimization**: Parallel tool execution beats sequential generation
- **Security**: Controlled execution environment rather than attempting to sandbox generated code

## Technical Components

### 1. Function Schema Definition

The function schema is a JSON Schema object that describes your function's signature to the model. The model's fine-tuning has taught it to parse these schemas and generate matching JSON.

**Technical explanation**: The schema becomes part of the system prompt (or equivalent internal representation) during inference. The model's attention mechanism learns to reference schema constraints when generating the tool call JSON, similar to how it learns to follow formatting instructions.

**Practical implementation**:

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class SearchType(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class DatabaseSearchParams(BaseModel):
    """Schema automatically converts to JSON Schema"""
    query: str = Field(
        description="Search query - be specific about what information is needed"
    )
    search_type: SearchType = Field(
        default=SearchType.SEMANTIC,
        description="Type of search to perform"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results"
    )
    filters: Optional[dict[str, str]] = Field(
        default=None,
        description="Key-value filters to apply, e.g. {'category': 'documentation'}"
    )

def create_tool_schema(name: str, description: str, params_model: type[BaseModel]) -> dict:
    """Convert Pydantic model to LLM tool schema"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params_model.model_json_schema()
        }
    }

# Usage
search_tool = create_tool_schema(
    name="search_database",
    description="Search the technical documentation database. Use this when users ask about system architecture, APIs, or implementation details.",
    params_model=DatabaseSearchParams
)
```

**Real constraints**:

- **Schema complexity limits**: Most models handle schemas up to ~2000 tokens well. Beyond that, tool selection accuracy degrades.
- **Description quality matters significantly**: Models rely heavily on descriptions for tool selection. Vague descriptions cause incorrect tool choices.
- **Nested objects are supported but slow**: Deep nesting (>3 levels) increases latency by 20-40% as the model generates more complex JSON.

**Trade-offs**:

```python
# OPTION A: Single complex tool with many parameters
def analyze_data(
    data_source: str,
    operation: Literal["aggregate", "filter", "transform"],
    aggregation_type: Optional[str] = None,
    filter_conditions: Optional[dict] = None,
    transform_function: Optional[str] = None
) -> dict:
    """One tool, many optional parameters"""
    pass

# OPTION B: Multiple focused tools
def aggregate_data(data_source: str, aggregation_type: str) -> dict:
    """Simpler schema, clearer intent"""
    pass

def filter_data(data_source: str, conditions: dict) -> dict:
    pass

def transform_data(data_source: str, function: str) -> dict:
    pass

# OPTION A: Fewer tool calls, but harder for model to use correctly
# OPTION B: Higher accuracy, but may require sequential calls
# Decision: Use B for user-facing features (reliability), A for internal agents (efficiency)
```

### 2. Tool Selection & Invocation Logic

The model performs two distinct operations: selecting which tool(s) to use, then generating valid parameters for those tools.

**Technical explanation**: During inference, the model's logits are constrained to ensure valid JSON output. This is similar to grammar-based sampling but specific to the provided schemas. The model effectively compiles multiple possible output formats (one per tool, plus "no tool") into its generation strategy.

**Practical implementation**:

```python
from typing import Callable, Any
import json

class ToolRegistry:
    """Type-safe tool registration and execution"""
    
    def __init__(self):
        self.tools: dict[str, Callable] = {}
        self.schemas: list[dict] = []
    
    def register(
        self,
        func: Callable,
        description: str,
        params_model: type[BaseModel]
    ) -> None:
        """Register a function as an available tool"""
        tool_name = func.__name__
        self.tools[tool_name] = func
        self.schemas.append(
            create_tool_schema(tool_name, description, params_model)
        )
    
    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> tuple[bool, Any]:
        """Execute a tool call with error handling"""
        if tool_name not in self.tools:
            return False, f"Unknown tool: {tool_name}"
        
        try:
            func = self.tools[tool_name]
            # Get the params model from the schema
            params_model = self._get_params_model(tool_name)
            # Validate and parse arguments
            validated_params = params_model(**arguments)
            # Execute with validated parameters
            result = func(**validated_params.model_dump())
            return True, result
        except Exception as e:
            return False, f"Tool execution error: {str(e)}"
    
    def _get_params_model(self, tool_name: str) -> type[BaseModel]:
        """Extract params model from schema (implementation detail)"""
        # In production, store this mapping during registration
        pass

# Example usage
registry = ToolRegistry()

class CalculateParams(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

def calculate(expression: str) -> float:
    """Safe expression evaluation"""
    import ast
    import operator
    
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    
    def eval_node(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_node(node.left), eval_node(node.right))
        raise ValueError(f"Unsupported operation: {type(node)}")
    
    tree = ast.parse(expression, mode='eval')
    return eval_node(tree.body)

registry.register(
    func=calculate,
    description="Evaluate mathematical expressions. Supports +, -, *, /, **",
    params_model=CalculateParams
)
```

**Real constraints**:

- **Tool selection accuracy**: With 5 tools, expect ~95% accuracy. With 20+ tools, accuracy drops to ~80-85% without hierarchical organization.
- **Parallel tool calls**: Some models support generating multiple tool calls in one response, but execution order isn't guaranteed. You must handle dependencies.
- **Token overhead**: Each tool schema adds to the prompt. 10 tools with moderate descriptions ≈ 500-1000 tokens of context.

### 3. Conversation State Management

Tool calls create a specific message structure that must be maintained for the model to understand context.

**Technical explanation**: The conversation becomes a graph, not a list. Each tool call creates a branch: assistant message with tool call → tool message with result → next assistant message. Breaking this structure causes the model to lose context about what was executed and why.

**Practical implementation**:

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class Message:
    role: Literal["user", "assistant", "tool", "system"]
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool messages
    name: Optional[str] = None  # Tool name for tool messages

class ConversationManager:
    """Manages multi-turn conversations with tool calls"""
    
    def __init__(self, system_prompt: str = ""):
        self.messages: list[Message] = []
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))
    
    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
    
    def add_assistant_response(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None
    ) -> None:
        """Add assistant message (may include tool calls)"""
        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )
    
    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> None:
        """Add tool execution result"""
        self.messages.append(
            Message(
                role="tool",
                content=result,
                tool_call_id=tool_call_id,
                name=tool_name
            )
        )
    
    def get_messages_for_api(self) -> list[dict]:
        """Convert to API format"""
        return [self._message_to_dict(m) for m in self.messages]
    
    def _message_to_dict(self, msg: Message) -> dict:
        """Convert message to API dict format"""
        result = {"role": msg.role}
        if msg.content:
            result["content"] = msg.content
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if msg.name:
            result["name"] = msg.name
        return result
    
    def get_context_size(self) -> int:
        """Estimate token count (rough approximation)"""
        return sum(
            len(json.dumps(self._message_to_dict(m))) // 4
            for m in self.messages
        )
```

**Real constraints**:

- **Context window consumption**: Each tool call adds 3+ messages. A conversation with 5 tool calls can consume 1000+ tokens just in structure.
- **Message ordering is strict**: Tool messages must immediately follow the assistant message containing the corresponding tool call. Out-of-order messages cause API errors or hallucinations.
- **Tool results format matters**: Returning complex objects as JSON strings works better than natural language descriptions for subsequent tool calls.

### 4. Parallel Tool Execution

Some models can request multiple tools in a single response. This requires careful orchestration.

**Technical explanation**: The model generates an array of tool call objects in one output. Your application must determine execution order based on dependencies, execute them (potentially in parallel), and return all results before the next model invocation.

**Practical implementation**:

```python
import asyncio
from typing import Coroutine
from dataclasses import dataclass

@dataclass
class ToolDependency:
    """Defines tool execution dependencies"""
    tool_name: str
    depends_on: set[str]  # Other tool names this depends on

class ParallelToolExecutor:
    """Executes tools with dependency management"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.dependencies: dict[str, set[str]] = {}
    
    def register_dependency(self, tool_name: str, depends_on: