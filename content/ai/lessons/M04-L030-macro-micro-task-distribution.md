# Macro-Micro Task Distribution: Engineering Efficient AI Workflows

## Core Concepts

Macro-Micro Task Distribution is an architectural pattern for decomposing complex problems into a hierarchy where LLMs handle high-level orchestration (macro) while specialized tools or simpler models execute specific subtasks (micro). This inverts traditional software architecture: instead of hardcoding orchestration logic and using AI as a black-box component, you delegate strategic decision-making to the LLM while keeping deterministic operations deterministic.

### Traditional vs. Macro-Micro Architecture

**Traditional Approach:**
```python
def process_customer_feedback(feedback: str) -> dict:
    # Hardcoded orchestration logic
    sentiment = sentiment_analysis_api(feedback)
    
    if sentiment == "negative":
        category = categorize_issue(feedback)
        priority = calculate_priority(category, sentiment)
        response = generate_apology_template(category)
    else:
        category = "general"
        priority = "low"
        response = generate_thanks_template()
    
    return {
        "sentiment": sentiment,
        "category": category,
        "priority": priority,
        "response": response
    }
```

**Macro-Micro Approach:**
```python
from typing import Literal
import json

def process_customer_feedback(feedback: str, llm_client) -> dict:
    # LLM handles orchestration and decision-making (MACRO)
    macro_prompt = f"""
Analyze this customer feedback and determine the workflow:

Feedback: {feedback}

Provide a JSON response with:
- sentiment: positive/negative/neutral
- requires_categorization: boolean
- requires_priority: boolean
- needs_specialist_review: boolean
- next_steps: list of specific actions
"""
    
    workflow = llm_client.generate(macro_prompt)
    workflow_data = json.loads(workflow)
    
    result = {"sentiment": workflow_data["sentiment"]}
    
    # Execute micro-tasks with specialized tools
    if workflow_data["requires_categorization"]:
        result["category"] = categorize_with_keyword_matching(feedback)  # MICRO
    
    if workflow_data["requires_priority"]:
        result["priority"] = calculate_priority_score(
            sentiment=result["sentiment"],
            urgency_keywords=extract_urgency_indicators(feedback)  # MICRO
        )
    
    if workflow_data["needs_specialist_review"]:
        result["assigned_to"] = route_to_specialist(result.get("category"))  # MICRO
    
    # LLM generates context-aware response (MACRO)
    result["response"] = generate_contextual_response(
        feedback, 
        result, 
        llm_client
    )
    
    return result
```

### Key Engineering Insights

**1. Flexibility vs. Determinism Trade-off:** The LLM makes workflow decisions that would require extensive if-else logic, but deterministic operations (keyword matching, scoring algorithms) remain fast and predictable. You gain adaptability without sacrificing reliability where it matters.

**2. Token Economy:** By delegating computational subtasks to traditional code, you reduce token consumption by 60-80% compared to asking the LLM to perform every operation. An LLM analyzing sentiment costs ~500 tokens; extracting keywords via regex costs zero tokens.

**3. Debuggability Hierarchy:** When something fails, you can isolate whether the issue is in macro-level decision-making (prompt engineering problem) or micro-level execution (code bug). Traditional monolithic LLM calls create debugging black holes.

### Why This Matters Now

LLM costs have dropped 90% since 2022, but latency and reliability remain critical bottlenecks in production systems. Engineers treating LLMs as "magic black boxes" for entire workflows are building systems that are slow, expensive, and impossible to optimize. The macro-micro pattern emerged from production deployments where teams needed to ship reliable systems, not research prototypes.

The pattern maps cleanly to how humans delegate: a project manager (macro) doesn't write every line of code—they decide *what* needs building and delegate *how* to specialists (micro). Your LLM should orchestrate, not micromanage.

## Technical Components

### 1. Task Decomposition Strategy

**Technical Explanation:** Task decomposition identifies natural boundaries where decisions require semantic understanding (macro) versus where execution follows deterministic rules (micro). The decomposition isn't arbitrary—it maps to computational complexity and error propagation.

**Practical Implications:**

```python
from enum import Enum
from dataclasses import dataclass

class TaskType(Enum):
    MACRO = "semantic_decision"  # Requires LLM
    MICRO = "deterministic_execution"  # Traditional code

@dataclass
class Task:
    name: str
    type: TaskType
    input_dependencies: list[str]
    estimated_tokens: int  # For MACRO tasks
    estimated_ms: int      # For MICRO tasks

def analyze_email_workflow() -> list[Task]:
    """
    Example: Email processing workflow decomposition
    """
    return [
        Task("determine_intent", TaskType.MACRO, [], 300, 0),
        Task("extract_dates", TaskType.MICRO, ["determine_intent"], 0, 50),
        Task("validate_dates", TaskType.MICRO, ["extract_dates"], 0, 10),
        Task("check_calendar_conflicts", TaskType.MICRO, ["validate_dates"], 0, 200),
        Task("formulate_response", TaskType.MACRO, ["check_calendar_conflicts"], 500, 0),
    ]

def estimate_workflow_cost(tasks: list[Task], cost_per_1k_tokens: float = 0.002) -> dict:
    """Calculate total cost and latency for workflow."""
    total_tokens = sum(t.estimated_tokens for t in tasks)
    total_ms = sum(t.estimated_ms for t in tasks)
    macro_calls = sum(1 for t in tasks if t.type == TaskType.MACRO)
    
    # Account for LLM latency (average ~2s per call)
    total_latency_s = (total_ms / 1000) + (macro_calls * 2.0)
    
    return {
        "total_cost_usd": (total_tokens / 1000) * cost_per_1k_tokens,
        "total_latency_s": total_latency_s,
        "macro_tasks": macro_calls,
        "micro_tasks": len(tasks) - macro_calls
    }

# Example usage
workflow = analyze_email_workflow()
print(estimate_workflow_cost(workflow))
# Output: {'total_cost_usd': 0.0016, 'total_latency_s': 4.26, 'macro_tasks': 2, 'micro_tasks': 3}
```

**Real Constraints:** Over-decomposition creates overhead—each macro task adds ~2 seconds of latency. Under-decomposition wastes tokens on tasks that regex could handle in microseconds. The optimal split minimizes `(total_tokens * cost_per_token) + (total_latency * latency_penalty)`.

### 2. Context Passing Between Layers

**Technical Explanation:** Macro tasks produce semantic outputs (decisions, classifications, summaries) that micro tasks consume as structured inputs. The interface between layers must be explicit—no implicit state or hidden context.

**Practical Implications:**

```python
from typing import TypedDict, Optional
from datetime import datetime

class MacroOutput(TypedDict):
    """Structured output from LLM macro task."""
    intent: Literal["schedule_meeting", "request_info", "complaint", "other"]
    urgency: Literal["high", "medium", "low"]
    entities_mentioned: list[str]
    requires_calendar_check: bool
    confidence: float

class MicroInput(TypedDict):
    """Input to deterministic micro task."""
    raw_text: str
    macro_context: MacroOutput
    user_id: str
    timestamp: datetime

def macro_analyze_email(email_body: str, llm_client) -> MacroOutput:
    """LLM extracts semantic meaning."""
    prompt = f"""
Analyze this email and return JSON:
{{
    "intent": "schedule_meeting|request_info|complaint|other",
    "urgency": "high|medium|low",
    "entities_mentioned": ["entity1", "entity2"],
    "requires_calendar_check": true/false,
    "confidence": 0.0-1.0
}}

Email: {email_body}
"""
    response = llm_client.generate(prompt, temperature=0.1)
    return json.loads(response)

def micro_extract_datetime(micro_input: MicroInput) -> Optional[list[datetime]]:
    """Deterministic datetime extraction."""
    import re
    from dateutil import parser
    
    # Only run if macro layer determined calendar check is needed
    if not micro_input["macro_context"]["requires_calendar_check"]:
        return None
    
    # Use regex patterns for common date formats
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{4}-\d{2}-\d{2}',
        r'(next|this) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, micro_input["raw_text"], re.IGNORECASE)
        for match in matches:
            try:
                dates.append(parser.parse(match))
            except:
                continue
    
    return dates if dates else None

# Usage demonstrating context flow
email = "Can we meet next Tuesday at 2pm to discuss the Q4 budget?"
macro_result = macro_analyze_email(email, llm_client)

micro_context: MicroInput = {
    "raw_text": email,
    "macro_context": macro_result,
    "user_id": "user_123",
    "timestamp": datetime.now()
}

extracted_dates = micro_extract_datetime(micro_context)
```

**Real Constraints:** Context serialization adds 10-50ms per layer transition. For low-latency systems (<500ms), limit to 2-3 layers. Deep hierarchies (>5 layers) are only viable for batch processing.

### 3. Error Handling and Fallback Chains

**Technical Explanation:** Macro and micro tasks fail differently. Macro tasks produce invalid JSON or hallucinations; micro tasks throw exceptions or return None. A robust system needs layer-specific error handling with graceful degradation.

**Practical Implications:**

```python
from typing import Union, Callable
import logging

logger = logging.getLogger(__name__)

class WorkflowError(Exception):
    """Base exception for workflow failures."""
    pass

class MacroError(WorkflowError):
    """LLM produced invalid or low-confidence output."""
    pass

class MicroError(WorkflowError):
    """Deterministic task failed."""
    pass

def safe_macro_call(
    llm_func: Callable,
    fallback_value: dict,
    min_confidence: float = 0.7
) -> dict:
    """Execute macro task with validation and fallback."""
    try:
        result = llm_func()
        
        # Validate structure
        if not isinstance(result, dict):
            raise MacroError("LLM returned non-dict")
        
        # Check confidence threshold
        if result.get("confidence", 0) < min_confidence:
            logger.warning(f"Low confidence ({result['confidence']}), using fallback")
            return fallback_value
        
        return result
    
    except (json.JSONDecodeError, KeyError, MacroError) as e:
        logger.error(f"Macro task failed: {e}")
        return fallback_value

def safe_micro_call(
    micro_func: Callable,
    fallback_value: any = None,
    required: bool = False
) -> any:
    """Execute micro task with error handling."""
    try:
        result = micro_func()
        
        if required and result is None:
            raise MicroError("Required micro task returned None")
        
        return result
    
    except Exception as e:
        if required:
            raise MicroError(f"Critical micro task failed: {e}")
        
        logger.warning(f"Optional micro task failed: {e}")
        return fallback_value

# Example: Resilient workflow
def process_with_fallbacks(email: str, llm_client) -> dict:
    # Macro with fallback to conservative defaults
    macro_result = safe_macro_call(
        lambda: macro_analyze_email(email, llm_client),
        fallback_value={
            "intent": "other",
            "urgency": "medium",
            "entities_mentioned": [],
            "requires_calendar_check": False,
            "confidence": 0.0
        },
        min_confidence=0.6
    )
    
    # Optional micro task
    dates = safe_micro_call(
        lambda: micro_extract_datetime({
            "raw_text": email,
            "macro_context": macro_result,
            "user_id": "user_123",
            "timestamp": datetime.now()
        }),
        fallback_value=[],
        required=False
    )
    
    return {
        "macro": macro_result,
        "dates": dates,
        "status": "success"
    }
```

**Real Constraints:** Fallback chains increase reliability but mask underlying issues. Log every fallback activation—if >5% of macro calls use fallbacks, your prompts need refinement.

### 4. Parallel Micro-Task Execution

**Technical Explanation:** Once macro decisions are made, independent micro tasks can run concurrently. This parallelization often reduces latency by 40-60% compared to sequential execution.

**Practical Implications:**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine

async def async_micro_task(task_name: str, duration_ms: int) -> dict:
    """Simulate micro task with I/O."""
    await asyncio.sleep(duration_ms / 1000)
    return {"task": task_name, "result": f"completed in {duration_ms}ms"}

async def parallel_micro_execution(macro_output: MacroOutput) -> dict:
    """Execute independent micro tasks in parallel."""
    
    tasks: list[Coroutine] = []
    
    # These micro tasks don't depend on each other
    if macro_output["requires_calendar_check"]:
        tasks.append(async_micro_task("extract_dates", 50))
        tasks.append(async_micro_task("check_availability", 200))
    
    tasks.append(async_micro_task("sentiment_analysis", 100))
    tasks.append(async_micro_task("extract_entities", 75))
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and package results
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    return {
        "successful_tasks": successful,
        "failed_tasks": len(failed),
        "total_duration_ms": max(r.get("result", "0ms") for r in successful if isinstance(r, dict))
    }

# Sequential vs Parallel comparison
async def compare_execution_patterns(macro_output: MacroOutput):
    import time
    
    # Sequential execution
    start = time.time()
    seq_results = []
    seq_results.append(await async_micro_task("extract_dates", 50))
    seq_results.append(await async_micro_task("check_availability", 200))
    seq_results.append(await async_micro_task("sentiment_analysis", 100))
    seq_results.append(await async_micro_task("extract_entities", 75))
    sequential_time = time.time() - start
    
    # Parallel execution
    start = time.time()
    parallel_results = await parallel_micro_execution(macro_output)
    parallel_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.3f}s")
    print(f"Parallel: {parallel_time:.3f}s")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")

# Expected output:
# Sequential: 0.425s
# Parallel: 0.200s
# Speedup: 2.12x
```

**Real Constraints:** Parallelization helps only when micro tasks involve I/