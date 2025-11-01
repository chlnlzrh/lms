# Prompt Design Patterns & Templates

## Core Concepts

### Technical Definition

Prompt design patterns are reusable, structured approaches to crafting LLM inputs that consistently produce desired outputs. Like software design patterns (Strategy, Factory, Observer), prompt patterns encode proven solutions to recurring problems—but instead of organizing code, they organize natural language instructions to control AI behavior.

A prompt template is the concrete implementation: a parameterized structure with placeholders for variable content, analogous to function signatures or API contracts.

### Engineering Analogy: From Ad-Hoc Scripts to Structured APIs

**Traditional Approach (Ad-Hoc Prompting):**
```python
# Brittle, unreliable, hard to maintain
def get_summary(text: str) -> str:
    prompt = f"Summarize this: {text}"
    return llm.generate(prompt)

# Every variation requires complete rewrite
def get_technical_summary(text: str) -> str:
    prompt = f"Summarize this technical document: {text}"
    return llm.generate(prompt)
```

**Pattern-Based Approach:**
```python
from typing import Protocol, Dict, Any
from dataclasses import dataclass

@dataclass
class PromptPattern:
    """Reusable prompt structure with clear contract"""
    template: str
    required_params: list[str]
    output_format: str
    
class ChainOfThoughtPattern(PromptPattern):
    def __init__(self):
        super().__init__(
            template="""Task: {task}

Approach this systematically:
1. Break down the problem
2. Solve each component
3. Synthesize the solution

Input: {input_data}

Think step-by-step:""",
            required_params=["task", "input_data"],
            output_format="structured_reasoning"
        )
    
    def apply(self, **kwargs) -> str:
        missing = [p for p in self.required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        return self.template.format(**kwargs)

# Consistent, testable, composable
pattern = ChainOfThoughtPattern()
prompt = pattern.apply(
    task="Calculate compound interest",
    input_data="Principal: $10,000, Rate: 5%, Time: 3 years"
)
```

The difference: one is a string concatenation hack, the other is an engineered component with contracts, validation, and predictable behavior.

### Key Insights

**1. Prompts are interfaces, not magic spells.** They define contracts between your code and the LLM. Poor interface design causes the same problems in prompts as in APIs: ambiguity, brittleness, unpredictable failures.

**2. Context structure matters more than content volume.** Engineers instinctively dump more information into prompts when results are poor. This is like passing a giant unstructured dict to a function instead of designing proper parameters. A well-structured 200-token prompt outperforms a rambling 1000-token one.

**3. Patterns compose.** Chain-of-Thought + Few-Shot + Output Formatting creates more reliable results than any single technique. Like middleware in web frameworks, patterns stack to modify behavior systematically.

### Why This Matters Now

Production LLM systems are moving from "throw text at GPT and pray" to deterministic, testable components. Companies report 40-60% reduction in API costs and 3x improvement in output consistency by migrating from ad-hoc prompts to pattern-based architectures. The engineers who understand prompt patterns as rigorously as they understand REST API design will build the reliable AI systems that actually ship.

## Technical Components

### 1. Structural Patterns: Controlling Information Flow

**Technical Explanation:**
Structural patterns organize how information is presented to the LLM, similar to how HTTP request structure (headers, body, query params) affects API behavior. The pattern controls attention, context utilization, and output formatting.

**Critical Sub-Patterns:**

```python
from enum import Enum
from typing import Optional, List

class StructuralPattern(Enum):
    INSTRUCTION_CONTEXT_OUTPUT = "ico"
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHT = "tot"
    REACT = "react"

class ICOPattern:
    """Instruction-Context-Output: The fundamental structure"""
    
    @staticmethod
    def build(
        instruction: str,
        context: Dict[str, Any],
        output_spec: str,
        constraints: Optional[List[str]] = None
    ) -> str:
        prompt_parts = [
            "# Instruction",
            instruction,
            "",
            "# Context",
            "\n".join(f"{k}: {v}" for k, v in context.items()),
            "",
            "# Output Format",
            output_spec
        ]
        
        if constraints:
            prompt_parts.extend([
                "",
                "# Constraints",
                "\n".join(f"- {c}" for c in constraints)
            ])
        
        return "\n".join(prompt_parts)

# Usage
prompt = ICOPattern.build(
    instruction="Extract all monetary values and their associated entities",
    context={
        "document_type": "financial_report",
        "text": "Q3 revenue reached $2.4M, up from $1.8M in Q2..."
    },
    output_spec="JSON array: [{\"entity\": str, \"amount\": float, \"currency\": str}]",
    constraints=["Include only explicitly stated amounts", "Convert all to USD"]
)
```

**Practical Implications:**
- 30-40% improvement in instruction-following accuracy vs. unstructured prompts
- Enables automated validation: you can programmatically verify the output matches the spec
- Facilitates A/B testing: change one section while keeping others constant

**Real Constraints:**
- Overhead: ~50-100 extra tokens per request
- Not all LLMs respect structural cues equally (smaller models often ignore them)
- Over-structuring can reduce creativity in open-ended tasks

### 2. Few-Shot Patterns: Teaching by Example

**Technical Explanation:**
Few-shot prompting provides input-output examples that demonstrate the desired behavior, similar to training data but evaluated at inference time. The LLM uses in-context learning to pattern-match against examples.

```python
from typing import List, Tuple
import random

class FewShotPattern:
    """Dynamically construct few-shot prompts with example selection"""
    
    def __init__(self, examples: List[Tuple[str, str]], task_description: str):
        self.examples = examples
        self.task_description = task_description
    
    def build(
        self,
        input_data: str,
        num_examples: int = 3,
        selection_strategy: str = "random"
    ) -> str:
        selected = self._select_examples(input_data, num_examples, selection_strategy)
        
        prompt_parts = [self.task_description, ""]
        
        for i, (example_input, example_output) in enumerate(selected, 1):
            prompt_parts.extend([
                f"Example {i}:",
                f"Input: {example_input}",
                f"Output: {example_output}",
                ""
            ])
        
        prompt_parts.extend([
            "Now process this input:",
            f"Input: {input_data}",
            "Output:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _select_examples(
        self,
        input_data: str,
        num_examples: int,
        strategy: str
    ) -> List[Tuple[str, str]]:
        if strategy == "random":
            return random.sample(self.examples, min(num_examples, len(self.examples)))
        elif strategy == "similarity":
            # In production, use embedding similarity
            # Simplified here for clarity
            return self.examples[:num_examples]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

# Usage with real example
examples = [
    ("The server crashed at 3am", "SEVERITY: high, CATEGORY: infrastructure"),
    ("User reported slow page load", "SEVERITY: medium, CATEGORY: performance"),
    ("Typo in button label", "SEVERITY: low, CATEGORY: ui")
]

classifier = FewShotPattern(
    examples=examples,
    task_description="Classify incident reports by severity and category"
)

prompt = classifier.build("Database connection pool exhausted", num_examples=2)
```

**Practical Implications:**
- 2-5x reduction in errors for structured extraction tasks
- Reduces need for fine-tuning in many scenarios
- Examples act as implicit validation—if output doesn't match example format, something's wrong

**Real Constraints:**
- Token cost scales linearly with examples (3 examples ≈ 200-400 tokens)
- Example selection matters enormously: poor examples actively harm performance
- Diminishing returns after 5-7 examples for most tasks

**Advanced: Semantic Example Selection**

```python
import numpy as np
from typing import Callable

class SemanticFewShotPattern(FewShotPattern):
    """Select most relevant examples using embedding similarity"""
    
    def __init__(
        self,
        examples: List[Tuple[str, str]],
        task_description: str,
        embedding_fn: Callable[[str], np.ndarray]
    ):
        super().__init__(examples, task_description)
        self.embedding_fn = embedding_fn
        # Pre-compute example embeddings
        self.example_embeddings = np.array([
            embedding_fn(inp) for inp, _ in examples
        ])
    
    def _select_examples(
        self,
        input_data: str,
        num_examples: int,
        strategy: str
    ) -> List[Tuple[str, str]]:
        if strategy != "similarity":
            return super()._select_examples(input_data, num_examples, strategy)
        
        input_embedding = self.embedding_fn(input_data)
        
        # Cosine similarity
        similarities = np.dot(self.example_embeddings, input_embedding) / (
            np.linalg.norm(self.example_embeddings, axis=1) * np.linalg.norm(input_embedding)
        )
        
        top_indices = np.argsort(similarities)[-num_examples:][::-1]
        return [self.examples[i] for i in top_indices]
```

### 3. Chain Patterns: Multi-Step Reasoning

**Technical Explanation:**
Chain patterns decompose complex tasks into sequential steps, where each step's output becomes the next step's input. This mirrors functional composition or Unix pipes: `task | step1 | step2 | step3`.

```python
from typing import Callable, Any, List
from dataclasses import dataclass

@dataclass
class ChainStep:
    name: str
    prompt_template: str
    parser: Callable[[str], Any]

class ChainPattern:
    """Execute multi-step reasoning with intermediate outputs"""
    
    def __init__(self, steps: List[ChainStep]):
        self.steps = steps
    
    def execute(
        self,
        initial_input: str,
        llm_call: Callable[[str], str],
        verbose: bool = False
    ) -> Dict[str, Any]:
        context = {"input": initial_input}
        results = {}
        
        for step in self.steps:
            # Build prompt with all previous context
            prompt = step.prompt_template.format(**context)
            
            if verbose:
                print(f"\n=== {step.name} ===")
                print(f"Prompt:\n{prompt[:200]}...")
            
            # Execute LLM call
            raw_output = llm_call(prompt)
            
            # Parse and store
            parsed_output = step.parser(raw_output)
            results[step.name] = parsed_output
            context[step.name] = parsed_output
            
            if verbose:
                print(f"Output: {parsed_output}")
        
        return results

# Example: Multi-step code review
import json

def parse_issues(output: str) -> List[str]:
    """Extract issue list from LLM output"""
    lines = output.strip().split("\n")
    return [line.strip("- ").strip() for line in lines if line.strip().startswith("-")]

def parse_severity(output: str) -> Dict[str, str]:
    """Parse JSON severity mapping"""
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {}

def parse_recommendations(output: str) -> str:
    return output.strip()

code_review_chain = ChainPattern([
    ChainStep(
        name="identify_issues",
        prompt_template="""Review this code for issues:

```python
{input}
```

List all issues as bullet points:""",
        parser=parse_issues
    ),
    ChainStep(
        name="assess_severity",
        prompt_template="""Given these code issues:
{identify_issues}

Rate each as: critical, high, medium, or low.
Output as JSON: {{"issue": "severity"}}""",
        parser=parse_severity
    ),
    ChainStep(
        name="recommend_fixes",
        prompt_template="""Issues and severities:
{assess_severity}

Provide specific fix recommendations for critical and high severity items:""",
        parser=parse_recommendations
    )
])

# Execute (mock LLM for demonstration)
def mock_llm(prompt: str) -> str:
    if "List all issues" in prompt:
        return "- Missing error handling\n- SQL injection vulnerability\n- Unused variable"
    elif "Rate each" in prompt:
        return '{"Missing error handling": "high", "SQL injection vulnerability": "critical"}'
    else:
        return "1. Add try-except blocks\n2. Use parameterized queries"

results = code_review_chain.execute(
    "def query(user_input): cursor.execute(f'SELECT * FROM users WHERE id={user_input}')",
    mock_llm,
    verbose=True
)
```

**Practical Implications:**
- 50-70% improvement on complex reasoning tasks vs. single-shot prompts
- Intermediate outputs enable debugging: you can see where chains fail
- Each step can use different models (cheap model for classification, expensive for generation)

**Real Constraints:**
- Latency: N steps = N sequential API calls (can't parallelize without complex orchestration)
- Cost multiplier: 3-step chain ≈ 3x base cost
- Error propagation: mistakes in step 1 cascade through entire chain

### 4. Role & Persona Patterns: Behavioral Framing

**Technical Explanation:**
Role patterns prime the LLM's behavior by establishing a persona with specific expertise, constraints, and communication style. This works because LLMs are trained on diverse text where roles correlate with specific patterns (doctors use medical terminology, lawyers cite precedents, etc.).

```python
from typing import Optional, Dict

class RolePattern:
    """Define behavioral framing for LLM responses"""
    
    def __init__(
        self,
        role: str,
        expertise: List[str],
        constraints: List[str],
        communication_style: str
    ):
        self.role = role
        self.expertise = expertise
        self.constraints = constraints
        self.communication_style = communication_style
    
    def build_system_message(self) -> str:
        return f"""You are {self.role}.

Expertise:
{chr(10).join(f"- {e}" for e in self.expertise)}

Constraints:
{chr(10).join(f"- {c}" for c in self.constraints)}

Communication style: {self.communication_style}"""
    
    def wrap_user_message(self, content: str) -> str:
        return f"{content}\n\nRespond according to your role and constraints."

# Example: Technical code reviewer
code_reviewer = RolePattern(
    role="a senior software engineer conducting code review",
    expertise=[
        "Performance optimization",
        "Security best practices",
        "Maintainability and readability"
    ],
    constraints=[
        "Only flag genuine issues, not stylistic preferences",
        "Provide specific line references",
        "Suggest concrete improvements, not vague advice"
    ],
    communication_style="Direct and constructive, focus on impact"
)

# Example: Data extraction specialist
data_extractor = RolePattern(
    role="a data extraction specialist",
    expertise=[
        "Structured data parsing",
        "Entity recognition",
        "Handling ambiguous or incomplete information"
    ],
    constraints=[
        "Never infer information not present in the source