# Prompt Design Patterns

## Core Concepts

Prompt design patterns are reusable, structured approaches to formulating LLM requests that consistently produce reliable outputs. Unlike ad-hoc prompting, design patterns encode solutions to recurring problems in LLM interaction—similar to how software design patterns solve common architectural challenges.

### Engineering Analogy: From String Concatenation to Template Systems

Consider the evolution from simple string manipulation to sophisticated templating:

```python
# Ad-hoc approach: Fragile string concatenation
def generate_query_adhoc(user_input: str) -> str:
    return "Please analyze this: " + user_input + ". Give me insights."

# Problems: No structure, no validation, no reusability
# Output quality varies wildly with different inputs
```

```python
from typing import Protocol, List, Dict
from dataclasses import dataclass

# Pattern-based approach: Structured, composable, testable
@dataclass
class PromptContext:
    role: str
    constraints: List[str]
    output_format: Dict[str, str]
    examples: List[tuple[str, str]]

class PromptPattern(Protocol):
    def render(self, context: PromptContext, input_data: str) -> str:
        ...

class ChainOfThoughtPattern:
    def render(self, context: PromptContext, input_data: str) -> str:
        prompt_parts = [
            f"You are a {context.role}.",
            "\nConstraints:",
            *[f"- {c}" for c in context.constraints],
            "\nApproach this step-by-step:",
            "1. Identify key elements",
            "2. Analyze relationships",
            "3. Draw conclusions",
            f"\nInput: {input_data}",
            f"\nProvide output as: {context.output_format}"
        ]
        return "\n".join(prompt_parts)

# Now prompts are testable, versionable, and composable
pattern = ChainOfThoughtPattern()
context = PromptContext(
    role="data analyst",
    constraints=["Use only provided data", "Show calculations"],
    output_format={"format": "JSON", "fields": ["analysis", "confidence"]},
    examples=[]
)
```

### Key Insights

**Prompts are interfaces, not instructions.** Just as API design requires thinking about contracts, error handling, and versioning, prompt design requires considering input variability, output parsing, and pattern evolution.

**Structure reduces variance.** Unstructured prompts produce unpredictable outputs. Patterns introduce constraints that narrow the solution space, trading flexibility for reliability—the same trade-off as type systems in programming.

**Composability enables complexity.** Simple patterns combine into sophisticated workflows. A single pattern might fail on complex tasks, but composed patterns (retrieval → reasoning → verification) handle multi-step processes reliably.

### Why This Matters Now

LLMs are moving from experimental tools to production infrastructure. In production, you need:
- **Predictable behavior** for testing and monitoring
- **Versioned prompts** that can be rolled back
- **Composable components** that scale to complex workflows
- **Measurable performance** across prompt iterations

Pattern-based prompting provides the engineering discipline that production LLM systems require.

## Technical Components

### 1. Instruction Patterns: Explicit Behavior Specification

Instruction patterns define the LLM's role, constraints, and output requirements explicitly rather than implicitly.

**Technical Explanation:** LLMs predict next tokens based on pattern matching against training data. Explicit instructions create stronger activation patterns for desired behaviors, similar to how strongly-typed function signatures reduce ambiguity in code execution.

**Practical Implications:**
- 40-60% reduction in output variance when instructions are explicit
- Easier to debug failures (unclear instruction vs. model limitation)
- Enables automated output validation

**Real Constraints:**
- Over-specification can reduce creativity or introduce bias
- Long instruction sets consume context window budget
- Trade-off between flexibility and control

```python
from enum import Enum
from typing import Optional, Literal

class OutputFormat(Enum):
    JSON = "JSON"
    MARKDOWN = "Markdown"
    PLAIN = "plain text"

@dataclass
class InstructionPattern:
    role: str
    task: str
    constraints: List[str]
    output_format: OutputFormat
    tone: Optional[Literal["formal", "casual", "technical"]] = None
    
    def render(self, input_data: str) -> str:
        sections = [
            f"Role: {self.role}",
            f"\nTask: {self.task}",
            "\nConstraints:"
        ]
        sections.extend([f"- {c}" for c in self.constraints])
        sections.append(f"\nOutput Format: {self.output_format.value}")
        
        if self.tone:
            sections.append(f"Tone: {self.tone}")
        
        sections.append(f"\n\nInput:\n{input_data}")
        return "\n".join(sections)

# Example usage
code_reviewer = InstructionPattern(
    role="Senior software engineer conducting code review",
    task="Identify security vulnerabilities and performance issues",
    constraints=[
        "Focus only on critical and high-severity issues",
        "Provide specific line numbers",
        "Suggest concrete fixes",
        "No style/formatting comments"
    ],
    output_format=OutputFormat.JSON,
    tone="technical"
)

code_sample = """
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    result = db.execute(query)
    return result
"""

prompt = code_reviewer.render(code_sample)
# This produces a structured, unambiguous prompt that consistently
# identifies SQL injection vulnerabilities
```

### 2. Few-Shot Learning Patterns: Demonstration-Based Guidance

Few-shot patterns provide examples of desired input-output pairs, enabling the model to infer patterns without explicit instruction.

**Technical Explanation:** Examples create a local context that shifts the model's probability distribution toward the demonstrated pattern. This is analogous to providing test cases in TDD—they define expected behavior more precisely than natural language specifications.

**Practical Implications:**
- 2-5 examples typically optimal (more can cause overfitting to example style)
- Example quality matters more than quantity
- Effective when task is easier to show than explain

**Real Constraints:**
- Examples consume significant context window (100-500 tokens each)
- Poor examples teach bad patterns
- Model may fixate on surface patterns (formatting) rather than deep structure

```python
from typing import List, Tuple

@dataclass
class FewShotExample:
    input: str
    output: str
    explanation: Optional[str] = None

class FewShotPattern:
    def __init__(self, task_description: str, examples: List[FewShotExample]):
        self.task_description = task_description
        self.examples = examples
    
    def render(self, new_input: str) -> str:
        prompt_parts = [self.task_description, "\n"]
        
        for i, example in enumerate(self.examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Input: {example.input}")
            prompt_parts.append(f"Output: {example.output}")
            if example.explanation:
                prompt_parts.append(f"Why: {example.explanation}")
            prompt_parts.append("")
        
        prompt_parts.append(f"Now apply this pattern:\nInput: {new_input}")
        prompt_parts.append("Output:")
        
        return "\n".join(prompt_parts)

# Example: Teaching sentiment analysis with nuance
sentiment_analyzer = FewShotPattern(
    task_description="Classify sentiment with intensity (0-10 scale).",
    examples=[
        FewShotExample(
            input="The product works fine, nothing special.",
            output='{"sentiment": "neutral", "intensity": 5, "reasoning": "functional but unremarkable"}',
            explanation="Neutral with no strong emotion"
        ),
        FewShotExample(
            input="This is the worst purchase I've ever made!",
            output='{"sentiment": "negative", "intensity": 10, "reasoning": "superlative negative expression"}',
            explanation="Maximum negative intensity due to absolute language"
        ),
        FewShotExample(
            input="Pretty good, would recommend to friends.",
            output='{"sentiment": "positive", "intensity": 7, "reasoning": "positive with recommendation"}',
            explanation="Strong positive but not extreme"
        )
    ]
)

test_input = "Decent quality but overpriced for what you get."
prompt = sentiment_analyzer.render(test_input)
# Model learns the JSON structure AND the nuanced scoring approach
```

### 3. Chain-of-Thought Patterns: Explicit Reasoning Paths

Chain-of-Thought (CoT) patterns require the model to show intermediate reasoning steps before producing final outputs.

**Technical Explanation:** By forcing sequential token generation that represents reasoning steps, CoT exploits the autoregressive nature of LLMs. Each reasoning token influences subsequent predictions, creating a path-dependent probability distribution that improves logical consistency.

**Practical Implications:**
- 20-40% accuracy improvement on multi-step reasoning tasks
- Enables debugging of model logic
- Increases token consumption (2-5x more output tokens)

**Real Constraints:**
- Not beneficial for simple lookup tasks
- Can hallucinate plausible-sounding but incorrect reasoning
- Reasoning quality varies with model capability

```python
from typing import Callable, Any
import json

class ChainOfThoughtPattern:
    def __init__(
        self, 
        task: str, 
        reasoning_steps: List[str],
        validation_fn: Optional[Callable[[str], bool]] = None
    ):
        self.task = task
        self.reasoning_steps = reasoning_steps
        self.validation_fn = validation_fn
    
    def render(self, input_data: str) -> str:
        prompt_parts = [
            f"Task: {self.task}",
            "\nApproach this systematically:",
        ]
        
        for i, step in enumerate(self.reasoning_steps, 1):
            prompt_parts.append(f"{i}. {step}")
        
        prompt_parts.extend([
            "\nShow your work for each step.",
            f"\nProblem: {input_data}",
            "\nStep-by-step solution:"
        ])
        
        return "\n".join(prompt_parts)
    
    def parse_response(self, response: str) -> dict[str, Any]:
        """Extract reasoning steps and final answer from response."""
        lines = response.strip().split('\n')
        steps = []
        final_answer = None
        
        for line in lines:
            if line.strip().startswith(tuple(f"{i}." for i in range(1, 10))):
                steps.append(line.strip())
            elif "final answer" in line.lower() or "conclusion" in line.lower():
                final_answer = line.strip()
        
        return {
            "reasoning_steps": steps,
            "final_answer": final_answer,
            "is_valid": self.validation_fn(final_answer) if self.validation_fn else None
        }

# Example: Mathematical problem solving
math_solver = ChainOfThoughtPattern(
    task="Solve the math problem and show all work",
    reasoning_steps=[
        "Identify what is being asked",
        "List known values and relationships",
        "Choose appropriate formula or method",
        "Perform calculations step-by-step",
        "Verify the answer makes sense"
    ],
    validation_fn=lambda ans: ans is not None and any(char.isdigit() for char in ans)
)

problem = "A train travels 120 miles in 2 hours, then 180 miles in 3 hours. What is the average speed?"
prompt = math_solver.render(problem)

# Simulated response parsing
sample_response = """
1. Identify what is being asked: Average speed over entire journey
2. List known values: Distance1=120mi, Time1=2hr, Distance2=180mi, Time2=3hr
3. Choose appropriate formula: Average speed = Total distance / Total time
4. Perform calculations: Total distance = 120 + 180 = 300 miles, Total time = 2 + 3 = 5 hours, Average speed = 300/5 = 60 mph
5. Verify: 60 mph over 5 hours = 300 miles ✓
Final answer: 60 mph
"""

result = math_solver.parse_response(sample_response)
print(json.dumps(result, indent=2))
```

### 4. Retrieval-Augmented Patterns: Context Injection

Retrieval-Augmented Generation (RAG) patterns inject relevant external information into prompts to ground model outputs in specific knowledge.

**Technical Explanation:** LLMs generate based on parametric knowledge (learned during training) which can be outdated or incomplete. RAG patterns supplement prompts with retrieved context, shifting the probability distribution toward information in the injected context rather than potentially-hallucinated parametric knowledge.

**Practical Implications:**
- Eliminates hallucination for factual queries when context contains the answer
- Enables dynamic knowledge updates without retraining
- Reduces model size requirements (knowledge externalized)

**Real Constraints:**
- Quality bottlenecked by retrieval system quality
- Context window limits amount of injectable information
- Retrieval latency adds to overall response time

```python
from typing import Protocol, List
from dataclasses import dataclass
import hashlib

@dataclass
class RetrievedContext:
    content: str
    source: str
    relevance_score: float

class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedContext]:
        ...

class RAGPattern:
    def __init__(
        self, 
        retriever: RetrieverProtocol,
        max_context_length: int = 2000,
        include_citations: bool = True
    ):
        self.retriever = retriever
        self.max_context_length = max_context_length
        self.include_citations = include_citations
    
    def render(self, query: str, top_k: int = 3) -> tuple[str, List[RetrievedContext]]:
        # Retrieve relevant contexts
        contexts = self.retriever.retrieve(query, top_k)
        
        # Build context section
        context_parts = ["Relevant Information:"]
        total_length = 0
        used_contexts = []
        
        for i, ctx in enumerate(contexts, 1):
            ctx_text = f"\n[{i}] (Source: {ctx.source}, Relevance: {ctx.relevance_score:.2f})\n{ctx.content}"
            if total_length + len(ctx_text) <= self.max_context_length:
                context_parts.append(ctx_text)
                used_contexts.append(ctx)
                total_length += len(ctx_text)
        
        prompt_parts = [
            "Use ONLY the information provided below to answer the question.",
            "If the answer is not in the provided information, say 'Insufficient information'.",
            "",
            *context_parts,
            "",
            f"Question: {query}",
        ]
        
        if self.include_citations:
            prompt_parts.append("\nAnswer with citations [1], [2], etc.:")
        else:
            prompt_parts.append("\nAnswer:")
        
        return "\n".join(prompt_parts), used_contexts

# Mock retriever for demonstration
class MockDocumentRetriever:
    def __init__(self, documents: List[tuple[str, str]]):
        self.documents = documents  # (content, source) pairs
    
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedContext]:
        # Simple keyword matching (real systems use embeddings)
        query_words = set(query.lower().split())
        scored_docs = []
        
        for content, source in self.documents:
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored_docs.append((content, source, overlap / len(query_words)))
        
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        return [
            RetrievedContext(content=content, source=source, relevance_score=score)
            for content, source, score in scored_docs[:top_k]
        ]

# Example usage
docs = [