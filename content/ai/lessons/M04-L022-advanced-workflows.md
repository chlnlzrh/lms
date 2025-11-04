# Advanced Workflows: Multi-Agent Systems and Orchestration Patterns

## Core Concepts

Advanced workflows in LLM systems represent a paradigm shift from single-shot prompt-response patterns to complex, multi-step computational graphs where language models function as composable reasoning units. Instead of treating an LLM as a monolithic oracle, advanced workflows decompose problems into specialized tasks, route information dynamically, and coordinate multiple model invocations with deterministic logic.

### The Engineering Shift

```python
# Traditional approach: Single LLM call with complex prompt
def analyze_customer_feedback_traditional(feedback: str) -> dict:
    prompt = f"""Analyze this feedback and provide:
    1. Sentiment (positive/negative/neutral)
    2. Key topics mentioned
    3. Action items for product team
    4. Priority level
    
    Feedback: {feedback}
    
    Return as JSON."""
    
    response = llm.generate(prompt)
    return parse_json(response)  # Fragile, inconsistent structure

# Advanced workflow: Orchestrated multi-step pipeline
from typing import Literal, List
from pydantic import BaseModel

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    reasoning: str

class TopicExtraction(BaseModel):
    topics: List[str]
    primary_concern: str

class ActionPlan(BaseModel):
    items: List[str]
    priority: Literal["high", "medium", "low"]
    reasoning: str

def analyze_customer_feedback_workflow(feedback: str) -> dict:
    # Step 1: Specialized sentiment analysis
    sentiment = llm.generate(
        prompt=f"Analyze sentiment: {feedback}",
        response_format=SentimentAnalysis
    )
    
    # Step 2: Conditional topic extraction with context
    if sentiment.confidence < 0.7:
        # Use more powerful model for ambiguous cases
        topics = llm_advanced.generate(
            prompt=f"Extract nuanced topics from ambiguous feedback: {feedback}",
            response_format=TopicExtraction
        )
    else:
        topics = llm.generate(
            prompt=f"Extract main topics: {feedback}",
            response_format=TopicExtraction
        )
    
    # Step 3: Priority-aware action planning
    action_plan = llm.generate(
        prompt=f"""Given sentiment ({sentiment.sentiment}, {sentiment.confidence}) 
        and primary concern ({topics.primary_concern}), generate action items.
        
        Feedback: {feedback}""",
        response_format=ActionPlan,
        temperature=0.3  # Lower temperature for consistent action items
    )
    
    # Step 4: Deterministic routing based on results
    if sentiment.sentiment == "negative" and action_plan.priority == "high":
        escalation = create_escalation_ticket(feedback, action_plan)
        return {**sentiment.dict(), **topics.dict(), **action_plan.dict(), 
                "escalation_id": escalation.id}
    
    return {**sentiment.dict(), **topics.dict(), **action_plan.dict()}
```

### Key Insights

**Decomposition over complexity**: Breaking a task into 4 specialized LLM calls with structured outputs produces more reliable results than a single complex prompt. Each step has a focused objective, making debugging and optimization tractable.

**Conditional execution paths**: Advanced workflows incorporate branching logic based on intermediate results. A low-confidence sentiment score triggers a different processing path, similar to how exception handling routes program execution.

**Type safety and validation**: Structured outputs (Pydantic models) transform LLM responses from string parsing problems into type-checked data structures, eliminating an entire class of runtime errors.

**Cost-performance tradeoffs**: Orchestration enables strategic model selection. Use efficient models for straightforward subtasks, reserving expensive models for complex reasoning steps.

### Why This Matters Now

The transition from GPT-3.5 to GPT-4 to modern models hasn't eliminated reliability issues—it's changed the economics. A single GPT-4 call might cost 10x more than GPT-3.5 but isn't 10x more reliable for decomposable tasks. Advanced workflows let you spend tokens strategically, achieving GPT-4-level results at GPT-3.5 prices by routing tasks appropriately. With structured output support now standard across providers, the infrastructure tax for building workflows has dropped dramatically.

## Technical Components

### 1. Task Decomposition and Specialization

Task decomposition transforms monolithic prompts into directed acyclic graphs (DAGs) of specialized operations. Each node performs one cognitive operation—classification, extraction, transformation, or generation—with well-defined inputs and outputs.

**Technical Explanation**: Decomposition exploits the statistical properties of language models. Models perform better on focused tasks because training data contains more examples of specific patterns (e.g., "extract entities") than complex compositions ("extract entities AND analyze sentiment AND generate recommendations"). Specialized prompts reduce the solution space, improving consistency.

**Practical Implications**: 

- Each task node should have a single primary objective measurable by a specific metric
- Task boundaries should align with natural breakpoints in your domain logic
- Intermediate outputs should be serializable for caching and debugging

**Real Constraints**:

- Decomposition adds latency (network round-trips between calls)
- Over-decomposition creates information loss at task boundaries
- Optimal granularity depends on your latency budget and accuracy requirements

**Concrete Example**:

```python
from typing import Protocol
from abc import abstractmethod

class TaskNode(Protocol):
    @abstractmethod
    async def execute(self, input_data: dict) -> dict:
        """Execute this task node and return structured output"""
        pass

class EntityExtractor(TaskNode):
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_template = """Extract named entities from the text.
        Return JSON with keys: persons, organizations, locations, dates.
        
        Text: {text}"""
    
    async def execute(self, input_data: dict) -> dict:
        text = input_data["text"]
        response = await self.llm.generate(
            self.prompt_template.format(text=text),
            response_format={"type": "json_object"}
        )
        return {"entities": response.parsed}

class RelationshipMapper(TaskNode):
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_template = """Given entities, identify relationships.
        Format: [entity1, relationship_type, entity2]
        
        Entities: {entities}
        Original text: {text}"""
    
    async def execute(self, input_data: dict) -> dict:
        response = await self.llm.generate(
            self.prompt_template.format(
                entities=input_data["entities"],
                text=input_data["text"]
            ),
            response_format={"type": "json_object"}
        )
        return {"relationships": response.parsed}

class KnowledgeGraphBuilder(TaskNode):
    """Deterministic node - no LLM call"""
    async def execute(self, input_data: dict) -> dict:
        entities = input_data["entities"]
        relationships = input_data["relationships"]
        
        graph = {
            "nodes": [{"id": e, "type": t} 
                     for t, entities_list in entities.items() 
                     for e in entities_list],
            "edges": [{"source": r[0], "target": r[2], "type": r[1]}
                     for r in relationships]
        }
        return {"knowledge_graph": graph}

# Workflow orchestrator
class WorkflowEngine:
    def __init__(self):
        self.tasks = {}
        self.dag = {}  # task_name -> list of dependent task names
    
    def register_task(self, name: str, task: TaskNode, depends_on: List[str] = None):
        self.tasks[name] = task
        self.dag[name] = depends_on or []
    
    async def execute(self, initial_input: dict) -> dict:
        results = {"input": initial_input}
        executed = set()
        
        async def run_task(task_name: str):
            if task_name in executed:
                return
            
            # Execute dependencies first
            for dep in self.dag[task_name]:
                await run_task(dep)
            
            # Gather inputs from dependencies
            task_input = {**initial_input}
            for dep in self.dag[task_name]:
                task_input.update(results[dep])
            
            # Execute task
            result = await self.tasks[task_name].execute(task_input)
            results[task_name] = result
            executed.add(task_name)
        
        # Execute all tasks respecting dependencies
        for task_name in self.tasks:
            await run_task(task_name)
        
        return results

# Usage
async def extract_knowledge_graph(text: str):
    engine = WorkflowEngine()
    engine.register_task("extract", EntityExtractor(llm))
    engine.register_task("map", RelationshipMapper(llm), depends_on=["extract"])
    engine.register_task("build", KnowledgeGraphBuilder(), depends_on=["extract", "map"])
    
    results = await engine.execute({"text": text})
    return results["build"]["knowledge_graph"]
```

### 2. Dynamic Routing and Conditional Execution

Dynamic routing selects execution paths based on runtime conditions—intermediate results, confidence scores, or input characteristics. This enables workflows to adapt behavior without hardcoding every scenario.

**Technical Explanation**: Routing logic combines deterministic control flow with probabilistic model outputs. You extract decision variables from LLM responses (confidence scores, classification labels) and apply traditional conditional logic to determine the next step.

**Practical Implications**:

- Routing decisions should be based on structured, validated outputs, not raw text
- Implement fallback paths for every routing decision
- Log routing decisions for debugging and optimization

**Real Constraints**:

- Each branch adds complexity to testing and validation
- Routing based on confidence scores requires calibration (model confidence != true probability)
- Deep branching creates exponential state space—keep routing depth ≤ 3

**Concrete Example**:

```python
from enum import Enum
from typing import Union

class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class QueryClassification(BaseModel):
    complexity: ComplexityLevel
    requires_external_data: bool
    confidence: float

class SimpleAnswer(BaseModel):
    answer: str
    source: Literal["direct_knowledge"]

class ResearchAnswer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class MultiStepAnswer(BaseModel):
    reasoning_steps: List[str]
    final_answer: str
    assumptions: List[str]

async def adaptive_question_answering(question: str) -> Union[SimpleAnswer, ResearchAnswer, MultiStepAnswer]:
    # Step 1: Classify query complexity
    classification = await llm.generate(
        prompt=f"""Classify this question's complexity and data needs:
        Question: {question}
        
        Consider:
        - Does it require factual lookup or reasoning?
        - Can it be answered directly or needs decomposition?
        - Does it need external/current data?""",
        response_format=QueryClassification,
        temperature=0.1
    )
    
    # Step 2: Route based on classification
    if classification.complexity == ComplexityLevel.SIMPLE and not classification.requires_external_data:
        # Fast path: direct answer from model knowledge
        return await llm.generate(
            prompt=f"Answer concisely: {question}",
            response_format=SimpleAnswer,
            max_tokens=150
        )
    
    elif classification.requires_external_data:
        # Research path: retrieve + synthesize
        search_results = await search_knowledge_base(question)
        
        return await llm.generate(
            prompt=f"""Answer using provided sources:
            Question: {question}
            Sources: {search_results}
            
            Cite sources and provide confidence level.""",
            response_format=ResearchAnswer,
            temperature=0.2
        )
    
    else:  # Complex reasoning required
        # Multi-step reasoning path
        if classification.confidence < 0.7:
            # Use more powerful model for uncertain cases
            llm_advanced = get_advanced_model()
            response = await llm_advanced.generate(
                prompt=f"""Break down this complex question into reasoning steps:
                Question: {question}
                
                Show your work step-by-step, then provide final answer.""",
                response_format=MultiStepAnswer,
                temperature=0.3
            )
        else:
            response = await llm.generate(
                prompt=f"""Answer with step-by-step reasoning:
                Question: {question}""",
                response_format=MultiStepAnswer,
                temperature=0.3
            )
        
        # Validation step: check reasoning consistency
        is_valid = await validate_reasoning(response.reasoning_steps, response.final_answer)
        if not is_valid:
            # Fallback: retry with more explicit constraints
            response = await llm.generate(
                prompt=f"""Previous attempt had logical inconsistencies.
                Answer carefully with explicit reasoning:
                Question: {question}""",
                response_format=MultiStepAnswer,
                temperature=0.1
            )
        
        return response

async def validate_reasoning(steps: List[str], conclusion: str) -> bool:
    """Lightweight validation of reasoning chain"""
    validation = await llm.generate(
        prompt=f"""Check if these reasoning steps logically lead to the conclusion.
        Steps: {steps}
        Conclusion: {conclusion}
        
        Return 'valid' or 'invalid' with brief explanation.""",
        max_tokens=50
    )
    return "valid" in validation.lower()
```

### 3. State Management and Context Propagation

Advanced workflows maintain state across multiple LLM calls, selectively passing information forward to minimize token usage while preserving necessary context.

**Technical Explanation**: Context propagation is an optimization problem: each LLM call has a token budget, and you must decide which information from previous steps to include. Naive approaches (pass everything forward) waste tokens and degrade performance as irrelevant information confuses the model.

**Practical Implications**:

- Design explicit state schemas—don't pass arbitrary dictionaries
- Implement context summarization for long-running workflows
- Use external storage (databases, vector stores) for large state objects

**Real Constraints**:

- Context window limits are hard constraints (typically 8k-128k tokens)
- Each token passed forward costs money on every subsequent call
- Information compression (summarization) introduces lossy transformations

**Concrete Example**:

```python
from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class WorkflowState:
    """Explicit state schema for multi-step workflow"""
    input_text: str
    current_step: str
    
    # Step outputs
    extracted_entities: Optional[dict] = None
    sentiment: Optional[str] = None
    key_insights: Optional[List[str]] = None
    final_report: Optional[str] = None
    
    # Metadata
    token_usage: dict = field(default_factory=dict)
    step_history: List[str] = field(default_factory=list)
    
    def add_step(self, step_name: str, tokens_used: int):
        self.step_history.append(step_name)
        self.token_usage[step_name] = tokens_used
        self.current_step = step_name
    
    def get_context_for_step(self, step_name: str) -> dict:
        """Return only relevant context for specific step"""
        contexts = {
            "sentiment_analysis": {
                "text": self.input_text
            },
            "insight_extraction": {
                "text": self.input_text,
                "sentiment": self.sentiment,
                "entities": self.extracted_entities
            },
            "report_generation": {
                "sentiment": self.sentiment,
                "insights": self.key_insights,
                # Note: NOT including full text to save tokens
            }
        }
        return contexts.get(step_name, {})
    
    def total_tokens(self) -> int:
        return sum(self.token_usage.values())

class StatefulWorkflow:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def analyze_document(self, text: str) -> WorkflowState:
        state = Workflow