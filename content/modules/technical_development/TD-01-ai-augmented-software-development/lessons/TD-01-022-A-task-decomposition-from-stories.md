# Task Decomposition from Stories: Engineering LLM-Powered Planning Systems

## Core Concepts

Task decomposition is the process of converting high-level objectives into structured, executable subtasks. In traditional software, this happens through explicit programming—developers hard-code decision trees, state machines, or rule engines. With LLMs, task decomposition becomes a prompt-driven reasoning process where the model analyzes context and generates action sequences.

### Traditional vs. LLM-Driven Decomposition

**Traditional approach:**

```python
def process_customer_request(request_type: str, data: dict) -> list[str]:
    """Hard-coded decomposition logic"""
    tasks = []
    
    if request_type == "refund":
        tasks.append("verify_purchase_date")
        tasks.append("check_refund_policy")
        tasks.append("validate_return_condition")
        tasks.append("process_refund")
    elif request_type == "exchange":
        tasks.append("verify_purchase_date")
        tasks.append("check_inventory")
        tasks.append("initiate_exchange")
    # Every scenario requires explicit coding
    
    return tasks
```

**LLM-driven approach:**

```python
from typing import List
import json

def decompose_with_llm(user_story: str, context: dict) -> List[dict]:
    """Dynamic decomposition through reasoning"""
    prompt = f"""Given this user story: "{user_story}"
    
Context: {json.dumps(context, indent=2)}

Break this into concrete, executable subtasks. For each subtask:
1. What specific action needs to happen
2. What information/resources are required
3. What validation is needed
4. Expected outcome

Return as JSON array of tasks."""

    # LLM generates appropriate decomposition
    response = call_llm(prompt)
    return json.loads(response)

# Handles novel scenarios without code changes
story = "Customer wants refund for damaged item bought 40 days ago"
tasks = decompose_with_llm(story, {"refund_policy": "30 days", "damage_exceptions": True})
```

### Why This Matters Now

Three engineering realities make LLM-driven decomposition critical:

1. **Requirement volatility**: Business logic changes faster than development cycles. LLMs adapt through prompt updates rather than code rewrites.

2. **Long-tail complexity**: Traditional systems handle 80% of cases well, but the remaining 20% require exponential complexity. LLMs reason through edge cases without explicit programming.

3. **Natural language interfaces**: Users increasingly interact through conversational interfaces. Decomposing unstructured requests into structured workflows is now a core requirement, not an edge case.

The fundamental shift: decomposition moves from compile-time (developer writes rules) to runtime (model reasons about context). This trades deterministic predictability for flexible adaptability—a trade-off you must engineer around, not ignore.

## Technical Components

### 1. Structured Output Formatting

LLMs generate text, but task decomposition requires structured data for downstream systems. The challenge is constraining free-form generation into parseable formats.

**Technical implementation:**

```python
from typing import List, Literal
from pydantic import BaseModel, Field
import json

class Subtask(BaseModel):
    """Structured task definition"""
    id: str = Field(description="Unique task identifier")
    action: str = Field(description="Specific action to perform")
    dependencies: List[str] = Field(default_factory=list, 
                                   description="Task IDs that must complete first")
    validation_criteria: str = Field(description="How to verify completion")
    estimated_duration: int = Field(description="Expected minutes to complete")
    task_type: Literal["api_call", "human_review", "calculation", "data_fetch"]

class TaskPlan(BaseModel):
    """Complete decomposition"""
    goal: str
    tasks: List[Subtask]
    critical_path: List[str]  # Task IDs in execution order

def decompose_story(story: str, schema: type[BaseModel]) -> TaskPlan:
    """Force structured output using schema"""
    prompt = f"""Decompose this story into tasks: {story}

Return ONLY valid JSON matching this schema:
{schema.model_json_schema()}

Example:
{{
  "goal": "Process customer refund",
  "tasks": [
    {{
      "id": "task_1",
      "action": "Verify purchase in database",
      "dependencies": [],
      "validation_criteria": "Purchase record found with matching order ID",
      "estimated_duration": 2,
      "task_type": "data_fetch"
    }}
  ],
  "critical_path": ["task_1", "task_2"]
}}
"""
    
    response = call_llm(prompt, temperature=0.0)
    
    # Parse and validate against schema
    try:
        data = json.loads(response)
        return TaskPlan.model_validate(data)
    except Exception as e:
        # Retry with error feedback
        return decompose_story_with_validation_error(story, schema, str(e))

# Usage
story = "User reports subscription charge after cancellation, wants refund"
plan = decompose_story(story, TaskPlan)

print(f"Generated {len(plan.tasks)} tasks for: {plan.goal}")
for task in plan.tasks:
    print(f"  {task.id}: {task.action} (requires: {task.dependencies})")
```

**Practical implications:**

- **Schema evolution**: When business requirements change, update the Pydantic model—prompts automatically adapt.
- **Validation at the boundary**: Catch malformed outputs before they enter your workflow engine.
- **Type safety**: Downstream code works with strongly-typed objects, not raw strings.

**Constraints:**

- Schema complexity affects success rate. Keep models shallow (max 3 levels of nesting).
- LLMs struggle with exact counts or precise numerical constraints in schemas.
- Always implement retry logic with validation error feedback.

### 2. Context Window Management for Complex Stories

Complex user stories require rich context (business rules, user history, system state), but context windows are finite. Effective decomposition requires strategic information packing.

**Technical implementation:**

```python
from dataclasses import dataclass
from typing import Optional
import tiktoken

@dataclass
class DecompositionContext:
    """Structured context for task planning"""
    user_story: str
    business_rules: dict
    user_profile: dict
    system_state: dict
    constraints: list[str]
    
    def estimate_tokens(self, model: str = "gpt-4") -> int:
        """Calculate token usage"""
        encoding = tiktoken.encoding_for_model(model)
        full_text = json.dumps(self.__dict__)
        return len(encoding.encode(full_text))
    
    def compress_for_budget(self, max_tokens: int) -> 'DecompositionContext':
        """Intelligently trim context to fit budget"""
        current_tokens = self.estimate_tokens()
        
        if current_tokens <= max_tokens:
            return self
        
        # Priority-based compression
        compressed = DecompositionContext(
            user_story=self.user_story,  # Never compress
            business_rules=self._compress_rules(self.business_rules, 0.5),
            user_profile=self._keep_critical_fields(self.user_profile),
            system_state=self._summarize_state(self.system_state),
            constraints=self.constraints[:5]  # Keep top 5 constraints
        )
        
        return compressed
    
    def _compress_rules(self, rules: dict, target_ratio: float) -> dict:
        """Keep most relevant rules"""
        # Sort by relevance to user story
        scored_rules = [
            (key, value, self._relevance_score(key, value))
            for key, value in rules.items()
        ]
        scored_rules.sort(key=lambda x: x[2], reverse=True)
        
        keep_count = int(len(scored_rules) * target_ratio)
        return {key: value for key, value, _ in scored_rules[:keep_count]}
    
    def _relevance_score(self, key: str, value: str) -> float:
        """Simple keyword matching relevance"""
        story_words = set(self.user_story.lower().split())
        rule_words = set(f"{key} {value}".lower().split())
        return len(story_words & rule_words) / len(story_words)
    
    def _keep_critical_fields(self, profile: dict) -> dict:
        """Extract only essential profile data"""
        critical = ["account_status", "subscription_tier", "total_purchases"]
        return {k: v for k, v in profile.items() if k in critical}
    
    def _summarize_state(self, state: dict) -> dict:
        """Convert detailed state to summary"""
        return {
            "active_orders": len(state.get("orders", [])),
            "pending_tickets": len(state.get("tickets", [])),
            "last_interaction": state.get("last_interaction_date")
        }

def decompose_with_context_management(
    context: DecompositionContext,
    max_context_tokens: int = 6000
) -> TaskPlan:
    """Decompose while respecting token budgets"""
    
    # Compress context if needed
    compressed = context.compress_for_budget(max_context_tokens)
    
    prompt = f"""User Story: {compressed.user_story}

Business Rules:
{json.dumps(compressed.business_rules, indent=2)}

User Profile:
{json.dumps(compressed.user_profile, indent=2)}

System State:
{json.dumps(compressed.system_state, indent=2)}

Constraints:
{chr(10).join(f"- {c}" for c in compressed.constraints)}

Decompose into executable tasks as JSON TaskPlan."""
    
    response = call_llm(prompt, temperature=0.0)
    return TaskPlan.model_validate(json.loads(response))

# Usage
context = DecompositionContext(
    user_story="VIP customer wants emergency Sunday delivery for order #12345",
    business_rules={
        "weekend_delivery": "Available for Premium tier only",
        "emergency_surcharge": "Additional $50 for same-day",
        "vip_benefits": "Free expedited shipping, 24/7 support"
        # ... 50 more rules
    },
    user_profile={
        "subscription_tier": "Premium",
        "account_status": "vip",
        "total_purchases": 127,
        # ... 20 more fields
    },
    system_state={
        "orders": [...],  # Large nested structure
        "warehouse_capacity": {...}
    },
    constraints=["Must complete before 6 PM Sunday", "Customer allergic to peanuts", ...]
)

plan = decompose_with_context_management(context)
```

**Practical implications:**

- **Token budgeting**: Reserve 40% for output generation, 60% for input context.
- **Relevance scoring**: Simple keyword matching works surprisingly well; avoid over-engineering.
- **Lossy compression**: Accept that some context will be dropped. Design systems to handle incomplete information gracefully.

**Trade-offs:**

- Compression loses information, potentially missing edge cases.
- Relevance scoring adds computational overhead.
- Context summarization requires domain knowledge encoded in the compression logic.

### 3. Dependency Graph Generation

Task decomposition must capture execution order. Sequential lists work for simple cases, but complex stories require dependency graphs.

**Technical implementation:**

```python
from typing import Set, Dict
import networkx as nx
from enum import Enum

class DependencyType(Enum):
    HARD = "hard"  # Must complete before
    SOFT = "soft"  # Preferred order but not required
    DATA = "data"  # Output of one task feeds into another

class TaskGraph:
    """Graph-based task representation"""
    
    def __init__(self, plan: TaskPlan):
        self.graph = nx.DiGraph()
        self._build_graph(plan)
    
    def _build_graph(self, plan: TaskPlan):
        """Convert flat task list to dependency graph"""
        for task in plan.tasks:
            self.graph.add_node(
                task.id,
                action=task.action,
                task_type=task.task_type,
                duration=task.estimated_duration
            )
        
        for task in plan.tasks:
            for dep_id in task.dependencies:
                self.graph.add_edge(dep_id, task.id, type=DependencyType.HARD)
    
    def get_execution_order(self) -> List[List[str]]:
        """Generate parallel execution stages"""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Circular dependency detected!")
        
        # Topological generations = tasks that can run in parallel
        return list(nx.topological_generations(self.graph))
    
    def get_critical_path(self) -> List[str]:
        """Find longest path (bottleneck)"""
        # Weight by estimated duration
        for u, v in self.graph.edges():
            self.graph[u][v]['weight'] = self.graph.nodes[v]['duration']
        
        return nx.dag_longest_path(self.graph)
    
    def estimate_min_completion_time(self) -> int:
        """Minimum time with infinite parallelization"""
        critical_path = self.get_critical_path()
        return sum(
            self.graph.nodes[task_id]['duration'] 
            for task_id in critical_path
        )
    
    def validate_consistency(self) -> List[str]:
        """Check for logical errors in decomposition"""
        issues = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            issues.append(f"Circular dependencies: {cycles}")
        
        # Check for orphaned tasks
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0 and self.graph.out_degree(node) == 0:
                if len(self.graph.nodes()) > 1:
                    issues.append(f"Orphaned task: {node}")
        
        # Check for missing dependencies
        for node in self.graph.nodes():
            action = self.graph.nodes[node]['action'].lower()
            if "verify" in action or "check" in action:
                # Verification tasks should come after action tasks
                if self.graph.in_degree(node) == 0:
                    issues.append(f"Verification task {node} has no dependencies")
        
        return issues

def decompose_with_dependency_validation(story: str) -> TaskGraph:
    """Decompose and validate dependency logic"""
    plan = decompose_story(story, TaskPlan)
    graph = TaskGraph(plan)
    
    # Validate and fix if needed
    issues = graph.validate_consistency()
    if issues:
        # Retry with validation feedback
        repair_prompt = f"""Previous decomposition had issues:
{chr(10).join(f"- {issue}" for issue in issues)}

Regenerate task decomposition fixing these problems."""
        
        # ... retry logic
    
    return graph

# Usage
story = "Process insurance claim for vehicle damage"
task_graph = decompose_with_dependency_validation(story)

# Analyze execution plan
stages = task_graph.get_execution_order()
print(f"Can complete in {len(stages)} stages:")
for i, stage in enumerate(stages, 1):
    print(f"  Stage {i} (parallel): {stage}")

critical = task_graph.get_critical_path()
min_time = task_graph.estimate_min_completion_time()
print(f"\nCritical path: {' -> '.join(critical)}")
print(f"Minimum completion time: {min_time} minutes")
```

**Practical implications:**

- **Parallelization opportunities**: Dependency graphs reveal which tasks can run concurrently, directly impacting total execution time.
- **Bottleneck identification**: Critical path analysis shows where to focus optimization efforts.
- **Validation hooks**: Graph algorithms catch logical errors (cycles, orphans) that text-based validation would miss.

**Constraints:**

- LLMs sometimes generate implicit dependencies that aren't explicitly listed. You may need to infer missing edges.
- Graph complexity grows quadratically with task count. For stories with >50 subtasks, consider hierarchical decomposition.

### 4. Iterative Refinement Through Feedback