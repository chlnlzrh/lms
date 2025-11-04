# Integration with Enterprise Systems

## Core Concepts

### Technical Definition

Enterprise system integration with LLMs involves creating bidirectional interfaces between language models and established business systems (ERP, CRM, data warehouses, legacy APIs) while maintaining security boundaries, data consistency, transaction integrity, and operational observability. Unlike traditional point-to-point integrations that follow deterministic workflows, LLM integrations require handling non-deterministic outputs, implementing semantic routing, managing context injection from multiple sources, and orchestrating multi-step operations where the LLM acts as an intelligent coordination layer.

### Engineering Analogy: Traditional vs. LLM-Mediated Integration

**Traditional Integration Pattern:**

```python
from typing import Dict, List
import requests

class TraditionalIntegration:
    """Direct API orchestration - fixed workflow"""
    
    def __init__(self, crm_url: str, erp_url: str):
        self.crm_url = crm_url
        self.erp_url = erp_url
    
    def process_customer_order(self, customer_id: str, product_id: str) -> Dict:
        # Fixed sequence: validate -> check inventory -> create order
        customer = requests.get(f"{self.crm_url}/customers/{customer_id}").json()
        
        if customer['credit_status'] != 'approved':
            raise ValueError("Customer credit not approved")
        
        inventory = requests.get(
            f"{self.erp_url}/inventory/{product_id}"
        ).json()
        
        if inventory['quantity'] < 1:
            raise ValueError("Out of stock")
        
        order = requests.post(
            f"{self.erp_url}/orders",
            json={'customer_id': customer_id, 'product_id': product_id}
        ).json()
        
        return order
```

**LLM-Mediated Integration:**

```python
from typing import Dict, List, Callable, Any
import json
from dataclasses import dataclass
import openai

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict
    function: Callable

class LLMIntegration:
    """LLM decides workflow based on context and intent"""
    
    def __init__(self, crm_url: str, erp_url: str, api_key: str):
        self.crm_url = crm_url
        self.erp_url = erp_url
        self.client = openai.OpenAI(api_key=api_key)
        self.tools = self._register_tools()
    
    def _register_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_customer_info",
                description="Retrieves customer credit status, history, and preferences",
                parameters={
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"}
                    },
                    "required": ["customer_id"]
                },
                function=self._get_customer_info
            ),
            Tool(
                name="check_inventory",
                description="Checks current inventory and estimates restock date",
                parameters={
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"}
                    },
                    "required": ["product_id"]
                },
                function=self._check_inventory
            ),
            Tool(
                name="create_order",
                description="Creates order in ERP system",
                parameters={
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "product_id": {"type": "string"},
                        "priority": {"type": "string", "enum": ["standard", "expedited"]}
                    },
                    "required": ["customer_id", "product_id"]
                },
                function=self._create_order
            ),
            Tool(
                name="create_backorder",
                description="Creates backorder when product out of stock",
                parameters={
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "product_id": {"type": "string"}
                    },
                    "required": ["customer_id", "product_id"]
                },
                function=self._create_backorder
            )
        ]
    
    def process_request(self, natural_language_request: str) -> Dict:
        """LLM analyzes intent and orchestrates appropriate workflow"""
        messages = [
            {
                "role": "system",
                "content": """You are an enterprise system orchestrator. 
                Analyze requests and execute appropriate system operations.
                Handle edge cases intelligently (credit issues, stock problems).
                Always verify customer status before creating orders."""
            },
            {"role": "user", "content": natural_language_request}
        ]
        
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in self.tools
        ]
        
        # LLM decides which tools to call and in what order
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        # Execute tool calls
        results = []
        while response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                tool = next(t for t in self.tools if t.name == tool_call.function.name)
                args = json.loads(tool_call.function.arguments)
                result = tool.function(**args)
                results.append({
                    "tool": tool_call.function.name,
                    "result": result
                })
                
                # Add tool response to conversation
                messages.append({
                    "role": "function",
                    "name": tool_call.function.name,
                    "content": json.dumps(result)
                })
            
            # Get next action from LLM
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=tools_schema,
                tool_choice="auto"
            )
        
        return {
            "final_response": response.choices[0].message.content,
            "operations_executed": results
        }
    
    def _get_customer_info(self, customer_id: str) -> Dict:
        # Actual API call
        return {"customer_id": customer_id, "credit_status": "approved", "vip": True}
    
    def _check_inventory(self, product_id: str) -> Dict:
        return {"product_id": product_id, "quantity": 0, "restock_date": "2024-02-15"}
    
    def _create_order(self, customer_id: str, product_id: str, priority: str = "standard") -> Dict:
        return {"order_id": "ORD-12345", "status": "created"}
    
    def _create_backorder(self, customer_id: str, product_id: str) -> Dict:
        return {"backorder_id": "BO-67890", "estimated_ship": "2024-02-15"}
```

**Key difference:** The traditional system follows a rigid workflow that fails on edge cases. The LLM system can reason about the situation—if inventory is out, it automatically creates a backorder for VIP customers instead of just failing. The workflow adapts to context.

### Why This Matters NOW

Three technical factors make this critical in 2024:

1. **Function calling maturity**: Modern LLMs (GPT-4, Claude 3+) reliably parse JSON schemas and generate valid function calls 95%+ of the time, crossing the threshold for production use.

2. **Context window expansion**: 128K+ token windows allow injecting complete database schemas, API documentation, and multi-step operation history, enabling the LLM to reason about complex system states.

3. **Cost/latency improvements**: Sub-second response times and $0.01/1K tokens make LLM-mediated integration economically viable for high-volume operations where previously only deterministic systems were feasible.

Engineers who master this pattern can replace thousands of lines of brittle integration code with flexible, context-aware orchestration that handles edge cases without explicit programming.

## Technical Components

### 1. Semantic Tool Registration & Discovery

**Technical Explanation:**

Unlike REST APIs with OpenAPI specs or GraphQL schemas that define exact inputs/outputs, LLM tool registration requires semantic descriptions that enable the model to infer when and how to use each tool. The quality of these descriptions directly impacts tool selection accuracy.

**Implementation Pattern:**

```python
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import inspect

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool
    enum: Optional[List[str]] = None
    default: Optional[Any] = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: str
    constraints: List[str]
    examples: List[Dict[str, Any]]

class EnterpriseToolRegistry:
    """Manages tool definitions with semantic richness for LLM discovery"""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
    
    def register(
        self,
        description: str,
        constraints: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None
    ):
        """Decorator to register functions as LLM-callable tools"""
        def decorator(func: Callable):
            # Extract type hints for automatic parameter schema
            sig = inspect.signature(func)
            params = []
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "string"
                
                # Extract enum from Literal type hint
                enum_values = None
                if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is Literal:
                    enum_values = list(param.annotation.__args__)
                
                params.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter {param_name}",  # Could parse from docstring
                    required=param.default == inspect.Parameter.empty,
                    enum=enum_values,
                    default=param.default if param.default != inspect.Parameter.empty else None
                ))
            
            tool_def = ToolDefinition(
                name=func.__name__,
                description=description,
                parameters=params,
                returns=sig.return_annotation.__name__ if sig.return_annotation != inspect.Signature.empty else "Dict",
                constraints=constraints or [],
                examples=examples or []
            )
            
            self.tools[func.__name__] = tool_def
            return func
        
        return decorator
    
    def to_openai_schema(self) -> List[Dict]:
        """Convert to OpenAI function calling format"""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{tool.description}\n\nConstraints:\n" + "\n".join(f"- {c}" for c in tool.constraints),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            p.name: {
                                "type": p.type,
                                "description": p.description,
                                **({"enum": p.enum} if p.enum else {})
                            }
                            for p in tool.parameters
                        },
                        "required": [p.name for p in tool.parameters if p.required]
                    }
                }
            })
        return schemas

# Usage example
registry = EnterpriseToolRegistry()

@registry.register(
    description="Retrieves customer account information including credit limit, payment history, and current outstanding balance. Use this before any financial transaction.",
    constraints=[
        "Only returns data for active customers",
        "Requires valid customer_id in UUID format",
        "Returns cached data (max 5 minutes old)"
    ],
    examples=[
        {"input": {"customer_id": "123e4567-e89b-12d3-a456-426614174000"}, "output": {"credit_limit": 50000, "outstanding": 12000}}
    ]
)
def get_customer_account(customer_id: str) -> Dict:
    # Implementation
    pass
```

**Trade-offs:**

- **Rich descriptions improve accuracy** but increase prompt token usage (typically 100-300 tokens per tool)
- **Too many tools** (>20) degrades selection accuracy; use namespacing or hierarchical registration
- **Semantic ambiguity** between similar tools requires explicit constraint specifications

### 2. Stateful Execution Context Management

**Technical Explanation:**

Enterprise operations often span multiple API calls where later calls depend on earlier results. LLMs are stateless, so maintaining execution context—including intermediate results, error states, and business rule violations—requires explicit state management patterns.

**Implementation Pattern:**

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

class ExecutionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    REQUIRES_APPROVAL = "requires_approval"

@dataclass
class ToolExecution:
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[int] = None

@dataclass
class ExecutionContext:
    """Maintains state across multi-step LLM orchestration"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    intent: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    executions: List[ToolExecution] = field(default_factory=list)
    accumulated_data: Dict[str, Any] = field(default_factory=dict)
    business_rules_checked: List[str] = field(default_factory=list)
    approval_required: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_execution(self, execution: ToolExecution):
        self.executions.append(execution)
        
        # Store results for cross-tool reference
        if execution.result:
            self.accumulated_data[execution.tool_name] = execution.result
    
    def get_context_summary(self) -> str:
        """Generate summary for LLM prompt injection"""
        summary = f"Session: {self.session_id}\n"
        summary += f"Intent: {self.intent}\n"
        summary += f"Executed operations ({len(self.executions)}):\n"
        
        for exec in self.executions:
            status = "✓" if exec.result else "✗"
            summary += f"  {status} {exec.tool_name}({exec.arguments})\n"
            if exec.error:
                summary += f"    Error: {exec.error}\n"
        
        summary += f"\nAccumulated data: {list(self.accumulated_data.keys())}\n"
        summary += f"Business rules checked: {', '.join(self.business_rules_checked)}\n"
        
        return summary
    
    def requires_rollback(self) -> bool:
        """Determine if previous operations need compensation"""
        return any(exec.error for exec in self.executions)

class StatefulOrchestrator:
    """Manages execution context across LLM calls"""
    
    def __init__(self):
        self.contexts: Dict[str, ExecutionContext] = {}
    
    def create_context(self, user_id: str, intent: str) -> ExecutionContext:
        ctx = ExecutionContext(user_id=user_id, intent