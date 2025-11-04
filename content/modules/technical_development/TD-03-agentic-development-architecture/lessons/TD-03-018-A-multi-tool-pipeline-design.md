# Multi-Tool Pipeline Design

## Core Concepts

Multi-tool pipeline design is the architectural pattern of decomposing complex AI tasks into discrete steps, each handled by specialized tools, models, or processing functions that execute in a coordinated sequence or graph. Rather than forcing a single model to handle every aspect of a problem, you create a workflow where each component excels at its specific subtask.

### Traditional vs. Pipeline Approach

```python
# Traditional: Single-shot approach
# One prompt tries to do everything

import anthropic
from typing import Dict, Any

def analyze_customer_feedback_traditional(feedback: str) -> Dict[str, Any]:
    """Single LLM call attempts entire analysis."""
    client = anthropic.Anthropic()
    
    prompt = f"""Analyze this customer feedback and:
    1. Extract the customer sentiment (positive/negative/neutral)
    2. Identify the product mentioned
    3. Categorize the issue type
    4. Determine urgency level
    5. Generate a response draft
    6. Suggest internal actions
    
    Feedback: {feedback}
    
    Return as JSON."""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Hope everything parsed correctly, no validation possible
    return eval(response.content[0].text)  # Risky!


# Modern: Multi-tool pipeline approach
# Each step is specialized and validated

from typing import Literal, Optional
from pydantic import BaseModel, ValidationError
import json

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]

class ProductIdentification(BaseModel):
    product_name: str
    product_category: str
    confidence: float

class IssueClassification(BaseModel):
    issue_type: Literal["bug", "feature_request", "support", "billing"]
    urgency: Literal["low", "medium", "high", "critical"]
    requires_escalation: bool

class ResponseDraft(BaseModel):
    response_text: str
    tone: str
    estimated_resolution_time: str

def analyze_sentiment(feedback: str) -> SentimentResult:
    """Specialized tool for sentiment analysis."""
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Analyze sentiment of this feedback. Return JSON:
            {{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0, "key_phrases": []}}
            
            Feedback: {feedback}"""
        }]
    )
    
    data = json.loads(response.content[0].text)
    return SentimentResult(**data)  # Validated!

def identify_product(feedback: str) -> ProductIdentification:
    """Specialized tool for product identification."""
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Identify the product mentioned. Return JSON:
            {{"product_name": "string", "product_category": "string", "confidence": 0.0-1.0}}
            
            Feedback: {feedback}"""
        }]
    )
    
    data = json.loads(response.content[0].text)
    return ProductIdentification(**data)

def classify_issue(
    feedback: str,
    sentiment: SentimentResult,
    product: ProductIdentification
) -> IssueClassification:
    """Contextual classification using previous results."""
    client = anthropic.Anthropic()
    
    context = f"""
    Sentiment: {sentiment.sentiment} ({sentiment.confidence})
    Product: {product.product_name}
    Feedback: {feedback}
    """
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Classify this issue. Return JSON:
            {{"issue_type": "bug|feature_request|support|billing",
              "urgency": "low|medium|high|critical",
              "requires_escalation": true|false}}
            
            Context: {context}"""
        }]
    )
    
    data = json.loads(response.content[0].text)
    return IssueClassification(**data)

def generate_response(
    feedback: str,
    sentiment: SentimentResult,
    product: ProductIdentification,
    classification: IssueClassification
) -> ResponseDraft:
    """Generate response using full pipeline context."""
    client = anthropic.Anthropic()
    
    # Only generate response if appropriate
    if classification.urgency == "critical":
        return ResponseDraft(
            response_text="Escalated to priority support team.",
            tone="urgent",
            estimated_resolution_time="immediate"
        )
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Generate customer response draft.
            
            Context:
            - Sentiment: {sentiment.sentiment}
            - Product: {product.product_name}
            - Issue: {classification.issue_type}
            - Urgency: {classification.urgency}
            - Original: {feedback}
            
            Return JSON: {{"response_text": "...", "tone": "...", "estimated_resolution_time": "..."}}"""
        }]
    )
    
    data = json.loads(response.content[0].text)
    return ResponseDraft(**data)

def analyze_customer_feedback_pipeline(feedback: str) -> Dict[str, Any]:
    """Orchestrate the full pipeline."""
    try:
        # Step 1: Sentiment (parallel-capable)
        sentiment = analyze_sentiment(feedback)
        
        # Step 2: Product ID (parallel-capable)
        product = identify_product(feedback)
        
        # Step 3: Classification (depends on 1 & 2)
        classification = classify_issue(feedback, sentiment, product)
        
        # Step 4: Response (depends on all previous)
        response_draft = generate_response(feedback, sentiment, product, classification)
        
        return {
            "sentiment": sentiment.model_dump(),
            "product": product.model_dump(),
            "classification": classification.model_dump(),
            "response": response_draft.model_dump(),
            "pipeline_status": "success"
        }
    
    except ValidationError as e:
        return {"pipeline_status": "validation_failed", "error": str(e)}
    except Exception as e:
        return {"pipeline_status": "failed", "error": str(e)}
```

### Key Insights That Change Engineering Thinking

1. **Composability over complexity**: Breaking tasks into stages allows you to test, debug, and optimize each component independently. A monolithic prompt failing is a black box; a pipeline stage failing is a specific, fixable problem.

2. **Contextual accumulation**: Each stage enriches the context for subsequent stages. Later tools make better decisions because they have structured, validated information from earlier stages rather than raw text.

3. **Selective execution**: Pipelines enable conditional logic. Skip expensive operations when unnecessary, route to specialized handlers based on intermediate results, or fail fast when confidence is low.

4. **Model diversity**: Different pipeline stages can use different models. Use faster, cheaper models for simple classification, reserve powerful models for complex reasoning or generation.

### Why This Matters Now

Current LLMs excel at specific, well-defined tasks but struggle with complex, multi-faceted problems requiring different types of reasoning. Pipeline design is the difference between:

- **85% accuracy** (single prompt doing everything) vs **96% accuracy** (specialized stages)
- **$0.50 per request** (always using the most powerful model) vs **$0.08 per request** (appropriate model selection)
- **4-second latency** (sequential mega-prompt) vs **1.2-second latency** (parallel execution of independent stages)
- **Debugging nightmares** (what went wrong?) vs **observability** (exactly which stage failed and why)

## Technical Components

### 1. Stage Definition and Interface Contracts

Each pipeline stage must have clearly defined inputs, outputs, and responsibilities. This enables testing, replacement, and parallel execution.

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel

# Define input/output types
class StageInput(BaseModel):
    """Base class for stage inputs."""
    pass

class StageOutput(BaseModel):
    """Base class for stage outputs."""
    stage_name: str
    execution_time_ms: float
    confidence: Optional[float] = None

TInput = TypeVar('TInput', bound=StageInput)
TOutput = TypeVar('TOutput', bound=StageOutput)

class PipelineStage(ABC, Generic[TInput, TOutput]):
    """Abstract base for all pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, input_data: TInput) -> TOutput:
        """Execute the stage logic."""
        pass
    
    async def validate_input(self, input_data: TInput) -> bool:
        """Optional input validation."""
        return True
    
    async def handle_error(self, error: Exception, input_data: TInput) -> Optional[TOutput]:
        """Optional error handling strategy."""
        return None

# Concrete implementation
class DocumentInput(StageInput):
    document_text: str
    document_id: str
    metadata: Dict[str, Any]

class ExtractionOutput(StageOutput):
    entities: list[Dict[str, Any]]
    key_facts: list[str]
    document_id: str

class EntityExtractionStage(PipelineStage[DocumentInput, ExtractionOutput]):
    """Extract entities from documents."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__("entity_extraction")
        self.model = model
        self.client = anthropic.Anthropic()
    
    async def execute(self, input_data: DocumentInput) -> ExtractionOutput:
        import time
        start = time.time()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Extract entities and key facts from this document.
                Return JSON: {{"entities": [{{"name": "...", "type": "...", "context": "..."}}],
                             "key_facts": ["..."]}}
                
                Document: {input_data.document_text}"""
            }]
        )
        
        data = json.loads(response.content[0].text)
        execution_time = (time.time() - start) * 1000
        
        return ExtractionOutput(
            stage_name=self.name,
            execution_time_ms=execution_time,
            entities=data["entities"],
            key_facts=data["key_facts"],
            document_id=input_data.document_id
        )
```

**Practical Implications**: 
- Type-safe interfaces catch errors at development time, not runtime
- Each stage can be unit tested with mock inputs
- Stages can be swapped (e.g., replace LLM extraction with regex for simple cases)
- Execution metrics are built into every stage output

**Trade-offs**:
- More boilerplate code upfront
- Over-abstraction can make simple pipelines unnecessarily complex
- Interface changes require updating multiple stages

### 2. Execution Graph and Dependency Management

Pipelines aren't always linear. Some stages can run in parallel, others depend on multiple upstream stages, and some may be conditional.

```python
from dataclasses import dataclass, field
from enum import Enum
import asyncio

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class StageNode:
    """Represents a stage in the execution graph."""
    stage: PipelineStage
    dependencies: list[str] = field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    
    def can_execute(self, completed_stages: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_stages for dep in self.dependencies)

class PipelineExecutor:
    """Manages pipeline execution with dependency resolution."""
    
    def __init__(self):
        self.stages: Dict[str, StageNode] = {}
        self.results: Dict[str, Any] = {}
    
    def add_stage(
        self,
        name: str,
        stage: PipelineStage,
        dependencies: list[str] = None
    ):
        """Add a stage to the execution graph."""
        self.stages[name] = StageNode(
            stage=stage,
            dependencies=dependencies or []
        )
    
    async def execute_stage(
        self,
        name: str,
        stage_node: StageNode
    ) -> Optional[Any]:
        """Execute a single stage."""
        try:
            stage_node.status = StageStatus.RUNNING
            
            # Gather inputs from dependencies
            input_data = self._prepare_stage_input(stage_node)
            
            # Execute
            result = await stage_node.stage.execute(input_data)
            
            stage_node.status = StageStatus.COMPLETED
            stage_node.result = result
            self.results[name] = result
            
            return result
            
        except Exception as e:
            stage_node.status = StageStatus.FAILED
            stage_node.error = e
            
            # Try error handling
            recovery_result = await stage_node.stage.handle_error(e, input_data)
            if recovery_result:
                stage_node.status = StageStatus.COMPLETED
                stage_node.result = recovery_result
                self.results[name] = recovery_result
                return recovery_result
            
            raise
    
    def _prepare_stage_input(self, stage_node: StageNode) -> Any:
        """Gather results from dependency stages."""
        if not stage_node.dependencies:
            return None
        
        if len(stage_node.dependencies) == 1:
            return self.results[stage_node.dependencies[0]]
        
        # Multiple dependencies: return dict of results
        return {
            dep: self.results[dep]
            for dep in stage_node.dependencies
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the entire pipeline."""
        completed_stages: set[str] = set()
        failed_stages: set[str] = set()
        
        while len(completed_stages) + len(failed_stages) < len(self.stages):
            # Find stages ready to execute
            ready_stages = [
                (name, node)
                for name, node in self.stages.items()
                if node.status == StageStatus.PENDING
                and node.can_execute(completed_stages)
            ]
            
            if not ready_stages:
                # Check if we're deadlocked
                pending = [
                    name for name, node in self.stages.items()
                    if node.status == StageStatus.PENDING
                ]
                if pending:
                    raise RuntimeError(f"Pipeline deadlock. Pending stages: {pending}")
                break
            
            # Execute ready stages in parallel
            tasks = [
                self.execute_stage(name, node)
                for name, node in ready_stages
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update completed/failed sets
            for (name, node), result in zip(ready_stages, results):
                if isinstance(result, Exception):
                    failed_stages.add(name)
                else:
                    completed_stages.add(name)
        
        return {