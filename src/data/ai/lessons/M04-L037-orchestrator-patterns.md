# Orchestrator Patterns: Managing Complex LLM Workflows

## Core Concepts

An **orchestrator** is a control layer that coordinates multiple LLM calls, tool executions, and decision points to accomplish tasks too complex for a single prompt-response cycle. Think of it as the conductor of an orchestra—it doesn't play every instrument, but it decides when each plays, how they harmonize, and what the final performance sounds like.

### Traditional vs. Orchestrated Approach

**Traditional Single-Call Pattern:**
```python
from typing import Dict
import anthropic

def simple_report(data: Dict[str, any]) -> str:
    """Generate report in one LLM call - limited by context and reasoning depth"""
    client = anthropic.Anthropic()
    
    prompt = f"""Analyze this data and create a comprehensive report:
    {data}
    
    Include: executive summary, detailed analysis, recommendations, and risks."""
    
    response = client.messages.create(
        model="claude-sonnet-4",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
```

**Orchestrated Multi-Step Pattern:**
```python
from typing import Dict, List
from dataclasses import dataclass
import anthropic

@dataclass
class AnalysisStep:
    name: str
    prompt_template: str
    dependencies: List[str]
    max_tokens: int

class ReportOrchestrator:
    """Coordinates multiple specialized LLM calls for deeper analysis"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.context: Dict[str, str] = {}
        
    def execute_step(self, step: AnalysisStep, data: Dict) -> str:
        """Execute a single analysis step with dependency resolution"""
        # Inject results from dependent steps
        context_data = {
            **data,
            **{dep: self.context[dep] for dep in step.dependencies if dep in self.context}
        }
        
        prompt = step.prompt_template.format(**context_data)
        
        response = self.client.messages.create(
            model="claude-sonnet-4",
            max_tokens=step.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text
        self.context[step.name] = result
        return result
    
    def generate_report(self, data: Dict[str, any]) -> Dict[str, str]:
        """Orchestrate multi-step analysis workflow"""
        steps = [
            AnalysisStep(
                name="data_validation",
                prompt_template="Validate this data for completeness and flag anomalies: {raw_data}",
                dependencies=[],
                max_tokens=500
            ),
            AnalysisStep(
                name="trend_analysis",
                prompt_template="Analyze trends in validated data: {raw_data}\nValidation notes: {data_validation}",
                dependencies=["data_validation"],
                max_tokens=1000
            ),
            AnalysisStep(
                name="risk_assessment",
                prompt_template="Assess risks based on: {trend_analysis}",
                dependencies=["trend_analysis"],
                max_tokens=800
            ),
            AnalysisStep(
                name="recommendations",
                prompt_template="Generate recommendations considering trends: {trend_analysis}\nAnd risks: {risk_assessment}",
                dependencies=["trend_analysis", "risk_assessment"],
                max_tokens=1000
            ),
            AnalysisStep(
                name="executive_summary",
                prompt_template="Create executive summary from: {trend_analysis}, {risk_assessment}, {recommendations}",
                dependencies=["trend_analysis", "risk_assessment", "recommendations"],
                max_tokens=600
            )
        ]
        
        for step in steps:
            self.execute_step(step, {"raw_data": str(data)})
        
        return self.context
```

**Key Difference:** The orchestrated approach breaks complex reasoning into specialized steps, each with focused objectives. This produces deeper analysis because:
- Each LLM call focuses on one type of reasoning
- Later steps build on validated earlier outputs
- You can tune token allocation per step
- Failures are isolated and recoverable

### Why Orchestration Matters Now

**Context Limitations:** Even with 200K token windows, forcing everything into one prompt creates cognitive overload. The model must simultaneously validate data, identify trends, assess risks, and synthesize recommendations—degrading quality on all fronts.

**Controllability:** Single-call approaches are black boxes. Orchestration gives you visibility into intermediate steps, allowing you to:
- Cache expensive analysis steps
- Retry failed components without recomputing everything
- A/B test different reasoning paths
- Insert human review at critical decision points

**Cost Efficiency:** A monolithic 4000-token output might cost $0.12 per request. The orchestrated approach uses 3900 total tokens across five calls, costing $0.10, but with 3-4x better reasoning quality because each step is optimized.

## Technical Components

### 1. State Management

Orchestrators maintain state across multiple LLM calls. Poor state management leads to context loss, redundant API calls, and inconsistent outputs.

**Technical Explanation:** State encompasses all intermediate results, metadata, and control flow information needed to execute the workflow. It must be serializable for debugging, resumable for error recovery, and efficient for high-throughput scenarios.

**Implementation Pattern:**
```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

@dataclass
class ExecutionState:
    """Serializable workflow state"""
    workflow_id: str
    current_step: int
    completed_steps: List[str] = field(default_factory=list)
    step_outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        """Serialize state for persistence"""
        state_dict = asdict(self)
        state_dict['created_at'] = self.created_at.isoformat()
        return json.dumps(state_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExecutionState':
        """Restore state from persistence"""
        data = json.loads(json_str)
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

class StatefulOrchestrator:
    """Orchestrator with persistent state management"""
    
    def __init__(self, workflow_id: str):
        self.state = ExecutionState(workflow_id=workflow_id)
        
    def execute_with_state(self, step_name: str, func: callable, *args, **kwargs) -> Any:
        """Execute step and update state atomically"""
        if step_name in self.state.completed_steps:
            print(f"Skipping completed step: {step_name}")
            return self.state.step_outputs[step_name]
        
        try:
            result = func(*args, **kwargs)
            self.state.step_outputs[step_name] = result
            self.state.completed_steps.append(step_name)
            self.state.current_step += 1
            
            # Persist state after each step
            self._save_checkpoint()
            
            return result
        except Exception as e:
            self.state.metadata['last_error'] = {
                'step': step_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self._save_checkpoint()
            raise
    
    def _save_checkpoint(self):
        """Save state to disk for recovery"""
        with open(f"checkpoint_{self.state.workflow_id}.json", "w") as f:
            f.write(self.state.to_json())
    
    @classmethod
    def resume(cls, workflow_id: str) -> 'StatefulOrchestrator':
        """Resume from checkpoint"""
        with open(f"checkpoint_{workflow_id}.json", "r") as f:
            state = ExecutionState.from_json(f.read())
        
        orchestrator = cls(workflow_id)
        orchestrator.state = state
        return orchestrator
```

**Trade-offs:**
- **Checkpoint frequency:** Save after each step (safe but slower) vs. batch saves (faster but risk loss)
- **State size:** Storing full outputs enables debugging but increases memory; storing only IDs requires re-computation
- **Serialization format:** JSON is human-readable; pickle is faster but version-sensitive

### 2. Control Flow Logic

Control flow determines the execution path through your workflow—sequential, parallel, conditional, or looping. This is where orchestrators differ most from simple chains.

**Technical Explanation:** Unlike linear chains where step N always follows step N-1, orchestrators implement branching logic based on intermediate results. This requires a decision framework that interprets outputs and routes execution accordingly.

**Implementation Pattern:**
```python
from typing import Callable, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

class FlowDecision(Enum):
    CONTINUE = "continue"
    SKIP = "skip"
    RETRY = "retry"
    BRANCH_A = "branch_a"
    BRANCH_B = "branch_b"
    TERMINATE = "terminate"

@dataclass
class ConditionalStep:
    name: str
    execute_fn: Callable
    condition_fn: Callable[[Dict], FlowDecision]
    retry_limit: int = 3

class ConditionalOrchestrator:
    """Orchestrator with conditional branching"""
    
    def __init__(self, client):
        self.client = client
        self.context: Dict[str, any] = {}
        self.execution_path: List[str] = []
    
    def run_workflow(self, steps: List[ConditionalStep], initial_data: Dict):
        """Execute workflow with dynamic branching"""
        step_idx = 0
        
        while step_idx < len(steps):
            step = steps[step_idx]
            retry_count = 0
            
            while retry_count < step.retry_limit:
                try:
                    # Execute step
                    result = step.execute_fn(self.client, {**initial_data, **self.context})
                    self.context[step.name] = result
                    
                    # Evaluate condition
                    decision = step.condition_fn(self.context)
                    self.execution_path.append(f"{step.name}:{decision.value}")
                    
                    if decision == FlowDecision.CONTINUE:
                        step_idx += 1
                        break
                    elif decision == FlowDecision.SKIP:
                        step_idx += 2  # Skip next step
                        break
                    elif decision == FlowDecision.RETRY:
                        retry_count += 1
                        continue
                    elif decision == FlowDecision.TERMINATE:
                        return self.context
                    elif decision == FlowDecision.BRANCH_A:
                        step_idx = self._find_step_by_name(steps, f"{step.name}_branch_a")
                        break
                    elif decision == FlowDecision.BRANCH_B:
                        step_idx = self._find_step_by_name(steps, f"{step.name}_branch_b")
                        break
                
                except Exception as e:
                    retry_count += 1
                    if retry_count >= step.retry_limit:
                        raise RuntimeError(f"Step {step.name} failed after {retry_count} retries: {e}")
        
        return self.context
    
    def _find_step_by_name(self, steps: List[ConditionalStep], name: str) -> int:
        for idx, step in enumerate(steps):
            if step.name == name:
                return idx
        raise ValueError(f"Step {name} not found")

# Example usage
def analyze_sentiment(client, data: Dict) -> str:
    response = client.messages.create(
        model="claude-sonnet-4",
        max_tokens=100,
        messages=[{"role": "user", "content": f"Classify sentiment as positive/negative/neutral: {data['text']}"}]
    )
    return response.content[0].text.lower()

def sentiment_router(context: Dict) -> FlowDecision:
    """Route based on sentiment analysis"""
    sentiment = context.get('sentiment_analysis', '')
    if 'positive' in sentiment:
        return FlowDecision.BRANCH_A  # Upsell flow
    elif 'negative' in sentiment:
        return FlowDecision.BRANCH_B  # Support flow
    else:
        return FlowDecision.CONTINUE  # Standard flow

# Build workflow
steps = [
    ConditionalStep(
        name="sentiment_analysis",
        execute_fn=analyze_sentiment,
        condition_fn=sentiment_router
    ),
    ConditionalStep(
        name="sentiment_analysis_branch_a",
        execute_fn=lambda c, d: "Generating upsell recommendations...",
        condition_fn=lambda ctx: FlowDecision.CONTINUE
    ),
    ConditionalStep(
        name="sentiment_analysis_branch_b",
        execute_fn=lambda c, d: "Generating support response...",
        condition_fn=lambda ctx: FlowDecision.CONTINUE
    )
]
```

**Real Constraint:** Branching logic increases complexity exponentially. Each conditional doubles potential execution paths, making testing difficult. Limit branching depth to 2-3 levels for maintainability.

### 3. Error Handling and Recovery

Production orchestrators must handle partial failures gracefully. Unlike single API calls where retry is simple, orchestrated workflows require sophisticated recovery strategies.

**Technical Explanation:** Errors in orchestrated workflows fall into three categories: transient (network timeouts), deterministic (invalid input), and non-deterministic (LLM refusing to follow format). Each requires different recovery strategies.

**Implementation Pattern:**
```python
from typing import Optional, Callable, Type
from dataclasses import dataclass
import time
import anthropic

@dataclass
class RetryConfig:
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0
    retriable_exceptions: tuple = (anthropic.RateLimitError, anthropic.InternalServerError)

class ResilientOrchestrator:
    """Orchestrator with sophisticated error handling"""
    
    def __init__(self, client):
        self.client = client
        self.failure_log: List[Dict] = []
    
    def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        config: Optional[RetryConfig] = None,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> any:
        """Execute with exponential backoff and fallback"""
        config = config or RetryConfig()
        last_exception = None
        
        for attempt in range(config.max_attempts):
            try:
                return func(*args, **kwargs)
            
            except config.retriable_exceptions as e:
                last_exception = e
                delay = config.initial_delay * (config.backoff_multiplier ** attempt)
                
                self.failure_log.append({
                    'function': func.__name__,
                    'attempt': attempt + 1,
                    'exception': type(e).__name__,
                    'delay': delay
                })
                
                if attempt < config.max_attempts - 1:
                    print(f"Retry {attempt + 1}/{config.max_attempts} after {delay}s")
                    time.sleep(delay)
                else:
                    # Final attempt failed, try fallback
                    if fallback:
                        print(f"All retries failed, executing fallback")
                        return fallback(*args, **kwargs)
            
            except Exception as e:
                # Non-retriable error
                self.failure_log.append({
                    'function': func.__name__,
                    'attempt': attempt + 1,
                    'exception': type(e).__name__,
                    'fatal': True
                })
                raise
        
        raise last_exception

def extract_json_with_validation(client,