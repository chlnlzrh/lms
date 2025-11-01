# Advanced Orchestration: Building Reliable Multi-Step LLM Systems

## Core Concepts

Orchestration in LLM systems means coordinating multiple AI calls, data transformations, and decision points into reliable workflows that accomplish complex tasks. Unlike simple prompt-response patterns, orchestration handles branching logic, error recovery, state management, and coordination between multiple models or tools.

### Traditional vs. Orchestrated Approaches

```python
# Traditional: Single-shot approach (limited, brittle)
def analyze_customer_feedback_simple(feedback: str) -> dict:
    prompt = f"Analyze this feedback and categorize it: {feedback}"
    response = llm.generate(prompt)
    return {"analysis": response}

# Problem: No validation, no multi-step reasoning, no error handling
# Fails on: ambiguous feedback, multiple issues, need for data lookup

# Orchestrated: Multi-step workflow (robust, capable)
from typing import List, Dict, Optional
import json

def analyze_customer_feedback_orchestrated(feedback: str) -> dict:
    # Step 1: Extract structured information
    extraction_prompt = """Extract from this feedback:
    - sentiment (positive/negative/neutral)
    - issues mentioned (list)
    - urgency (low/medium/high)
    
    Feedback: {feedback}
    Return valid JSON only."""
    
    extracted = llm.generate(extraction_prompt.format(feedback=feedback))
    data = json.loads(extracted)
    
    # Step 2: Validate extraction - retry if needed
    if not all(key in data for key in ['sentiment', 'issues', 'urgency']):
        # Retry with more explicit instructions
        data = retry_with_clarification(feedback, data)
    
    # Step 3: For each issue, determine category and owner
    categorized_issues = []
    for issue in data['issues']:
        category_prompt = f"""Based on this issue: "{issue}"
        Return the most specific category: technical|billing|feature_request|support"""
        category = llm.generate(category_prompt).strip()
        
        # Step 4: Look up responsible team (external data integration)
        owner = get_team_for_category(category)
        categorized_issues.append({
            'issue': issue,
            'category': category,
            'owner': owner
        })
    
    # Step 5: Generate response based on urgency
    if data['urgency'] == 'high':
        response = generate_urgent_response(data, categorized_issues)
    else:
        response = generate_standard_response(data, categorized_issues)
    
    return {
        'sentiment': data['sentiment'],
        'issues': categorized_issues,
        'urgency': data['urgency'],
        'suggested_response': response,
        'metadata': {'steps_completed': 5, 'retries': 0}
    }
```

**Key Engineering Insight:** Orchestration transforms unreliable single LLM calls into reliable systems through decomposition, validation, and recovery mechanisms. The same principles that make distributed systems reliable apply here: idempotency, retries, timeouts, and state management.

### Why This Matters Now

Three convergent factors make orchestration critical:

1. **Cost-Complexity Trade-off**: Large models are expensive. Orchestrating smaller models for specific sub-tasks can reduce costs by 70-90% while maintaining quality.

2. **Reliability Requirements**: Production systems need >99% reliability. Single LLM calls achieve 60-80% task completion. Orchestration with validation and retry brings this to 95-99%.

3. **Capability Composition**: No single model excels at everything. Orchestration lets you route tasks to specialized models—using fast models for classification, accurate models for generation, and code-specific models for technical tasks.

## Technical Components

### 1. State Management and Context Propagation

Orchestrated workflows maintain state across multiple steps, deciding what context each step needs.

**Technical Explanation:** Each orchestration step generates outputs that become inputs for subsequent steps. Managing this state efficiently—deciding what to pass forward, what to store, and what to discard—determines both reliability and cost.

```python
from dataclasses import dataclass, field
from typing import Any, Dict
from datetime import datetime

@dataclass
class OrchestrationState:
    """Immutable state object passed between steps"""
    task_id: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def with_output(self, step_name: str, output: Any) -> 'OrchestrationState':
        """Create new state with additional output (immutable pattern)"""
        new_outputs = {**self.outputs, step_name: output}
        return OrchestrationState(
            task_id=self.task_id,
            inputs=self.inputs,
            outputs=new_outputs,
            metadata=self.metadata
        )
    
    def get_context_for_step(self, step_name: str, 
                            required_outputs: List[str]) -> Dict[str, Any]:
        """Extract only needed context for a step"""
        context = {'original_input': self.inputs}
        for output_key in required_outputs:
            if output_key in self.outputs:
                context[output_key] = self.outputs[output_key]
        return context

# Usage in orchestration
def research_and_write_article(topic: str) -> str:
    state = OrchestrationState(
        task_id=f"article_{datetime.now().timestamp()}",
        inputs={'topic': topic}
    )
    
    # Step 1: Research - needs only original topic
    research_context = state.get_context_for_step('research', [])
    research_results = research_topic(research_context['original_input']['topic'])
    state = state.with_output('research', research_results)
    
    # Step 2: Outline - needs research results
    outline_context = state.get_context_for_step('outline', ['research'])
    outline = create_outline(outline_context)
    state = state.with_output('outline', outline)
    
    # Step 3: Write - needs outline and research, not intermediate steps
    write_context = state.get_context_for_step('write', ['research', 'outline'])
    article = write_article(write_context)
    state = state.with_output('final_article', article)
    
    return article
```

**Practical Implications:** 
- Token costs scale with context size. Passing full conversation history to each step can increase costs 10x.
- Large contexts increase latency and reduce model accuracy (lost-in-the-middle problem).
- Selective context propagation reduces tokens by 60-80% in typical workflows.

**Trade-offs:**
- More selective context = lower cost but risk missing information
- Immutable state patterns are safer but use more memory
- JSON serialization of state enables persistence but adds overhead

### 2. Error Handling and Retry Logic

LLMs fail unpredictably: malformed outputs, refusals, rate limits, timeouts. Production orchestration must handle these systematically.

```python
from typing import Callable, TypeVar, Optional
import time
import logging
from enum import Enum

T = TypeVar('T')

class FailureMode(Enum):
    MALFORMED_OUTPUT = "malformed"
    CONTENT_POLICY = "content_policy"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNEXPECTED = "unexpected"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    backoff_base: float = 2.0
    timeout_seconds: float = 30.0
    retry_on: List[FailureMode] = field(default_factory=lambda: [
        FailureMode.MALFORMED_OUTPUT,
        FailureMode.RATE_LIMIT,
        FailureMode.TIMEOUT
    ])

def with_retry(
    func: Callable[..., T],
    config: RetryConfig,
    validation_func: Optional[Callable[[T], bool]] = None
) -> T:
    """Execute function with retry logic and validation"""
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            # Execute with timeout
            result = func()
            
            # Validate output if validator provided
            if validation_func and not validation_func(result):
                logging.warning(f"Validation failed on attempt {attempt + 1}")
                last_exception = ValueError("Output validation failed")
                
                # Wait before retry
                if attempt < config.max_attempts - 1:
                    sleep_time = config.backoff_base ** attempt
                    time.sleep(sleep_time)
                continue
            
            return result
            
        except json.JSONDecodeError as e:
            failure_mode = FailureMode.MALFORMED_OUTPUT
            last_exception = e
            
        except TimeoutError as e:
            failure_mode = FailureMode.TIMEOUT
            last_exception = e
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                failure_mode = FailureMode.RATE_LIMIT
                # Longer backoff for rate limits
                sleep_time = config.backoff_base ** (attempt + 2)
            else:
                failure_mode = FailureMode.UNEXPECTED
                sleep_time = config.backoff_base ** attempt
            
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                logging.info(f"Retry {attempt + 1} after {failure_mode.value}")
                time.sleep(sleep_time)
    
    raise Exception(f"Failed after {config.max_attempts} attempts: {last_exception}")

# Usage example
def extract_json_with_retry(prompt: str, llm_client) -> dict:
    def validate_has_required_fields(data: dict) -> bool:
        required = ['sentiment', 'category', 'priority']
        return all(field in data for field in required)
    
    def call_llm():
        response = llm_client.generate(prompt)
        return json.loads(response)
    
    config = RetryConfig(max_attempts=3, backoff_base=1.5)
    return with_retry(call_llm, config, validate_has_required_fields)
```

**Real Constraints:**
- Each retry adds 1-3 seconds latency plus backoff time
- Retry budgets matter: 3 steps × 3 retries = 9 potential LLM calls
- Validation functions must be fast (<100ms) or they become bottlenecks

### 3. Conditional Routing and Dynamic Workflows

Not all inputs need the same processing path. Routing decisions based on input characteristics or intermediate results optimize cost and quality.

```python
from abc import ABC, abstractmethod
from typing import Protocol

class WorkflowStep(Protocol):
    """Interface for workflow steps"""
    def execute(self, state: OrchestrationState) -> OrchestrationState:
        ...
    
    def should_execute(self, state: OrchestrationState) -> bool:
        """Determine if this step should run based on current state"""
        ...

class Router:
    """Routes execution through different paths based on conditions"""
    
    def __init__(self):
        self.routes: Dict[str, List[WorkflowStep]] = {}
    
    def add_route(self, condition_name: str, steps: List[WorkflowStep]):
        self.routes[condition_name] = steps
    
    def route(self, state: OrchestrationState) -> str:
        """Determine which route to take based on state"""
        # Example: Route based on input complexity
        input_text = state.inputs.get('text', '')
        
        if len(input_text) < 100:
            return 'simple'
        elif len(input_text) < 1000:
            return 'standard'
        else:
            return 'complex'

# Concrete example: Content moderation with routing
class QuickFilter(WorkflowStep):
    """Fast, simple check using pattern matching"""
    def execute(self, state: OrchestrationState) -> OrchestrationState:
        text = state.inputs['text']
        # Fast regex or keyword check
        has_obvious_issues = quick_pattern_check(text)
        return state.with_output('quick_filter', {
            'passed': not has_obvious_issues,
            'confidence': 'high' if has_obvious_issues else 'low'
        })
    
    def should_execute(self, state: OrchestrationState) -> bool:
        return True  # Always run first

class MLModeration(WorkflowStep):
    """Medium-cost ML model for ambiguous cases"""
    def execute(self, state: OrchestrationState) -> OrchestrationState:
        text = state.inputs['text']
        ml_score = ml_classifier.predict(text)
        return state.with_output('ml_moderation', {
            'score': ml_score,
            'passed': ml_score < 0.5
        })
    
    def should_execute(self, state: OrchestrationState) -> bool:
        # Only run if quick filter had low confidence
        quick_result = state.outputs.get('quick_filter', {})
        return quick_result.get('confidence') == 'low'

class LLMModeration(WorkflowStep):
    """Expensive LLM for nuanced judgment"""
    def execute(self, state: OrchestrationState) -> OrchestrationState:
        text = state.inputs['text']
        prompt = f"""Analyze if this content violates policies.
        Consider context and nuance. Return JSON with:
        {{"violates": boolean, "reason": string, "severity": 1-5}}
        
        Content: {text}"""
        
        result = llm.generate(prompt)
        data = json.loads(result)
        return state.with_output('llm_moderation', data)
    
    def should_execute(self, state: OrchestrationState) -> bool:
        # Only run if ML was uncertain (score near 0.5)
        ml_result = state.outputs.get('ml_moderation', {})
        if not ml_result:
            return False
        score = ml_result.get('score', 0)
        return 0.3 < score < 0.7

def moderate_content(text: str) -> dict:
    """Orchestrated moderation with conditional routing"""
    state = OrchestrationState(
        task_id=f"mod_{hash(text)}",
        inputs={'text': text}
    )
    
    # Define workflow steps
    steps = [
        QuickFilter(),
        MLModeration(),
        LLMModeration()
    ]
    
    # Execute only necessary steps
    for step in steps:
        if step.should_execute(state):
            state = step.execute(state)
            
            # Early exit if we have high-confidence result
            if 'quick_filter' in state.outputs:
                if state.outputs['quick_filter']['confidence'] == 'high':
                    break
    
    # Determine final result based on executed steps
    return compile_moderation_result(state)
```

**Practical Implications:**
- 80% of content fails quick filter (cost: ~$0.00001 per check)
- 15% needs ML classification (cost: ~$0.0001 per check)
- 5% needs LLM reasoning (cost: ~$0.01 per check)
- Average cost per item: $0.0005 vs $0.01 if all used LLM

### 4. Parallel Execution and Aggregation

Some orchestration steps can run concurrently, reducing total latency from sequential seconds to parallel maximum.

```python
import asyncio
from typing import List, Awaitable
from concurrent.futures import ThreadPoolExecutor

async def parallel_execute(
    tasks: List[Awaitable[T]],
    max_concurrent: int = 5
) -> List[T]:
    """Execute tasks in parallel with concurrency limit"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[bounded_task(t) for t in tasks])

# Example: Multi-aspect analysis in parallel
async def analyze_document_parallel(document: str) -> dict:
    """Analyze different aspects of document concurrently