# Documentation & Reference Materials for LLM Systems

## Core Concepts

Documentation for LLM-powered applications differs fundamentally from traditional software documentation. When you write documentation for a REST API, you describe deterministic endpoints with fixed schemas. When you document an LLM system, you're describing probabilistic behavior, context-sensitive outputs, and emergent capabilities that may change with model updates.

Consider this comparison:

```python
# Traditional API Documentation Approach
def get_user(user_id: int) -> User:
    """
    Retrieves user by ID.
    
    Args:
        user_id: Integer ID of user (1-999999)
    
    Returns:
        User object with fields: id, name, email, created_at
    
    Raises:
        UserNotFoundError: If user_id doesn't exist
        ValidationError: If user_id < 1
    """
    pass

# LLM System Documentation Approach
def analyze_sentiment(text: str, llm_client: LLMClient) -> dict:
    """
    Analyzes sentiment using LLM with structured output.
    
    Args:
        text: Input text (optimal: 10-500 words, max: 3000 tokens)
        llm_client: Configured LLM client (tested with GPT-4, Claude 3)
    
    Returns:
        dict with 'sentiment' (positive/negative/neutral), 
        'confidence' (0.0-1.0), 'reasoning' (str)
    
    Behavior Notes:
        - Confidence typically >0.8 for clear sentiment
        - May return 'neutral' with low confidence for ambiguous text
        - Performance degrades with text >2000 tokens
        - Sarcasm detection accuracy ~70% in testing
    
    Prompt Strategy:
        Uses chain-of-thought with examples (see PROMPT_TEMPLATE)
        Temperature: 0.3 for consistency
        Fallback: Rule-based classifier if LLM unavailable
    
    Cost: ~$0.002 per call (4k context average)
    Latency: p50=800ms, p95=2.1s
    """
    pass
```

The traditional documentation describes a contract. The LLM documentation describes behavior patterns, operating characteristics, and decision rationale. You need to document not just what your system does, but how it thinks, when it fails, and why you made specific prompt engineering choices.

**Why This Matters Now**

Three forces make documentation critical for LLM systems today:

1. **Team collaboration on prompt engineering**: Unlike traditional code where logic is explicit, prompts encode domain knowledge and behavioral expectations. A poorly documented prompt is archaeological work for the next engineer.

2. **Model version migrations**: When you upgrade from GPT-4 to GPT-4.5 or switch providers, documented behavior baselines let you measure regression. Without them, you're flying blind.

3. **Debugging non-deterministic failures**: When an LLM produces unexpected output 2% of the time, documentation of expected behavior patterns, edge cases, and testing methodology is your primary debugging tool.

The key insight: **LLM system documentation is executable knowledge**, not just reference material. It should enable an engineer to understand, modify, debug, and validate the system's behavior with minimal trial and error.

## Technical Components

### 1. Prompt Documentation Structure

Prompts are code. Document them like you'd document complex algorithms, with rationale and examples.

```python
from typing import TypedDict, Literal
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """
    Contract extraction prompt for legal documents.
    
    Purpose: Extract key terms from service agreements with high precision.
    Optimized for: Contracts 2-20 pages, standard commercial terms.
    
    Performance Characteristics:
        - Precision: 94% on test set (n=500)
        - Recall: 89% (misses non-standard clause types)
        - Cost: ~$0.05 per document (avg 8k tokens)
        
    Version History:
        v3: Added few-shot examples for payment terms (2024-01)
        v2: Switched to structured output format (2023-12)
        v1: Initial zero-shot implementation (2023-11)
    """
    
    system_message: str = """You are a legal contract analyzer specializing in commercial agreements.

Extract key terms with high precision. If uncertain, mark confidence as 'low'.
Focus on: payment terms, termination clauses, liability caps, renewal terms.

Output Format:
{
  "payment_terms": {"value": "...", "confidence": "high|medium|low"},
  "termination_notice_days": {"value": 30, "confidence": "high|medium|low"},
  ...
}"""

    few_shot_examples: list[dict] = None
    
    temperature: float = 0.2  # Low for consistency
    max_tokens: int = 2000
    
    def __post_init__(self):
        self.few_shot_examples = [
            {
                "input": "Payment due within 30 days of invoice date...",
                "output": {
                    "payment_terms": {
                        "value": "Net 30",
                        "confidence": "high"
                    }
                }
            },
            # Additional examples...
        ]

# Usage example with inline documentation
def extract_contract_terms(
    contract_text: str,
    model: str = "gpt-4-turbo"
) -> dict:
    """
    Extract structured terms from contract.
    
    Known Limitations:
        - Struggles with handwritten amendments
        - May miss implied terms (not explicitly stated)
        - Non-English contracts require translation first
    
    Tested Edge Cases:
        ✓ Multi-party agreements (extracts primary parties)
        ✓ Amended contracts (uses most recent terms)
        ✗ Contracts with conflicting clauses (returns first found)
    
    If adding new term types:
        1. Add to PromptTemplate.system_message format spec
        2. Add 2-3 few-shot examples
        3. Update test suite in test_contract_extraction.py
        4. Document expected precision/recall in this docstring
    """
    template = PromptTemplate()
    # Implementation...
```

**Practical Implications**: When a new engineer needs to modify extraction logic, they have performance baselines, known limitations, and guidance on the testing process. This reduces "trial and error" cycles from days to hours.

**Trade-offs**: Comprehensive prompt documentation takes 2-3x longer to write initially but saves 10x debugging time later. Balance depth with maintenance burden—focus on prompts that are complex, business-critical, or frequently modified.

### 2. Behavioral Testing Documentation

Traditional unit tests verify deterministic outputs. LLM systems require behavioral tests that verify output characteristics.

```python
import pytest
from typing import Callable
import json

class BehavioralTest:
    """
    Framework for documenting and testing LLM behavioral expectations.
    
    Purpose: Capture expected behavior patterns for regression testing
    across model versions or prompt changes.
    """
    
    def __init__(self, test_name: str, description: str):
        self.test_name = test_name
        self.description = description
        self.test_cases = []
    
    def add_case(
        self,
        input_data: dict,
        expected_pattern: Callable[[dict], bool],
        rationale: str
    ):
        """
        Add a behavioral test case.
        
        Args:
            input_data: Input to the LLM system
            expected_pattern: Function that validates output characteristics
            rationale: Why this behavior is expected (for documentation)
        """
        self.test_cases.append({
            'input': input_data,
            'validator': expected_pattern,
            'rationale': rationale
        })

# Example: Document expected sentiment analysis behavior
sentiment_tests = BehavioralTest(
    test_name="sentiment_analysis_edge_cases",
    description="""
    Tests sentiment analysis on ambiguous or complex inputs.
    
    Behavioral Expectations:
    1. Sarcasm: Should detect negative sentiment in obvious sarcasm
    2. Mixed sentiment: Should identify dominant sentiment or report 'mixed'
    3. Short text: Should return low confidence for <5 words
    4. Neutral business text: Should identify as neutral, not force positive/negative
    """
)

# Test case 1: Sarcasm detection
sentiment_tests.add_case(
    input_data={"text": "Oh great, another meeting that could have been an email"},
    expected_pattern=lambda output: (
        output['sentiment'] == 'negative' and 
        output['confidence'] > 0.6
    ),
    rationale="Obvious sarcasm with negative undertone. We accept 60%+ confidence "
              "because sarcasm is inherently ambiguous, but expect negative classification."
)

# Test case 2: Very short ambiguous text
sentiment_tests.add_case(
    input_data={"text": "OK"},
    expected_pattern=lambda output: (
        output['sentiment'] == 'neutral' and
        output['confidence'] < 0.7
    ),
    rationale="Single-word responses lack context. Expect neutral with low confidence. "
              "This documents that we prefer 'unsure' over 'wrong' for ambiguous inputs."
)

# Test case 3: Mixed sentiment with dominant tone
sentiment_tests.add_case(
    input_data={"text": "The product is excellent but the customer service was terrible and shipping took forever"},
    expected_pattern=lambda output: (
        output['sentiment'] in ['negative', 'mixed'] and
        'customer service' in output.get('reasoning', '').lower()
    ),
    rationale="Multiple strong sentiments. Accept 'negative' (more negative signals) "
              "or 'mixed'. Must mention customer service in reasoning to show it identified "
              "the key pain point, not just the positive product mention."
)

def run_behavioral_tests(
    test_suite: BehavioralTest,
    system_under_test: Callable
) -> dict:
    """
    Run behavioral tests and document results.
    
    Returns:
        dict with pass/fail status, failure details, and baseline measurements
    """
    results = {
        'test_name': test_suite.test_name,
        'description': test_suite.description,
        'passed': 0,
        'failed': 0,
        'failures': []
    }
    
    for i, case in enumerate(test_suite.test_cases):
        try:
            output = system_under_test(case['input'])
            if case['validator'](output):
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append({
                    'case_index': i,
                    'input': case['input'],
                    'output': output,
                    'expected_pattern': case['rationale']
                })
        except Exception as e:
            results['failed'] += 1
            results['failures'].append({
                'case_index': i,
                'error': str(e),
                'rationale': case['rationale']
            })
    
    return results
```

**Practical Implications**: When you switch from GPT-4 to Claude or update your prompt, run the behavioral test suite. A drop from 85% to 65% pass rate signals regression. The documented rationale helps you understand *why* behavior changed.

**Constraints**: Behavioral tests are more brittle than traditional unit tests. Expect to update 10-20% of tests with each major model version. Focus on high-value behaviors (business logic, safety constraints) rather than exhaustive coverage.

### 3. Context and Configuration Documentation

LLM systems have numerous configuration parameters that dramatically affect behavior. Document the decision space.

```python
from enum import Enum
from typing import Optional

class ModelStrategy(Enum):
    """
    Model selection strategies for different use cases.
    
    Decision Framework:
    - FAST: Real-time user-facing features (<500ms p95)
    - BALANCED: Background processing, user can wait 1-3s
    - QUALITY: Batch processing, accuracy critical
    - COST_OPTIMIZED: High-volume, quality adequate at lower tiers
    """
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    COST_OPTIMIZED = "cost_optimized"

class LLMConfig:
    """
    Configuration for LLM API calls with documented decision rationale.
    
    Configuration Philosophy:
    - Default to BALANCED strategy for new features
    - Use FAST only when latency measured as user pain point
    - Use QUALITY when errors have high business impact
    - Measure actual cost before using COST_OPTIMIZED
    """
    
    def __init__(
        self,
        strategy: ModelStrategy = ModelStrategy.BALANCED,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        self.strategy = strategy
        
        # Strategy-specific defaults with rationale
        if strategy == ModelStrategy.FAST:
            # Smaller, faster models
            self.model = "gpt-3.5-turbo"
            self.temperature = temperature if temperature is not None else 0.3
            self.max_tokens = max_tokens if max_tokens is not None else 500
            self.timeout_seconds = 5
            """
            FAST Strategy Rationale:
            - Model: 3.5-turbo gives 80% of 4.0 quality at 3x speed, 1/20th cost
            - Temperature: 0.3 for consistency without being robotic
            - Max tokens: 500 cap prevents runaway costs on user-generated prompts
            - Timeout: Fail fast for real-time UX
            
            Use Cases: 
            - Chat autocomplete
            - Real-time content suggestions
            - Quick sentiment checks
            
            NOT Suitable For:
            - Complex reasoning tasks
            - Long-form content generation
            - Tasks requiring deep domain knowledge
            """
            
        elif strategy == ModelStrategy.QUALITY:
            self.model = "gpt-4-turbo"
            self.temperature = temperature if temperature is not None else 0.2
            self.max_tokens = max_tokens if max_tokens is not None else 4000
            self.timeout_seconds = 30
            """
            QUALITY Strategy Rationale:
            - Model: GPT-4 for complex reasoning, nuanced understanding
            - Temperature: 0.2 for high consistency (deterministic when possible)
            - Max tokens: 4000 allows detailed responses
            - Timeout: 30s accommodates longer processing
            
            Use Cases:
            - Legal document analysis
            - Technical code review
            - Customer escalation analysis
            - Content moderation appeals
            
            Cost Impact: 20x more expensive than FAST strategy
            Measure: Only ~5% of requests typically need QUALITY tier
            """
            
        # Additional strategies...
        
    def get_retry_config(self) -> dict:
        """
        Retry configuration based on strategy.
        
        Documented Trade-offs:
        - FAST: Fail fast, don't retry (user experience > reliability)
        - QUALITY: Aggressive retries with exponential backoff
        - BALANCED: Retry transient failures only
        """
        if self.strategy == ModelStrategy.FAST:
            return {
                'max_retries': 1,
                'backoff_factor': 1,
                'timeout_seconds': 5,
                'rationale': 'User-facing: fail fast rather than make user wait'
            }
        elif self.strategy == ModelStrategy.QUALITY:
            return {
                'max_retries': 5,
                'backoff_factor': 2,
                'timeout_seconds': 30,
                'rationale': 'Batch processing: reliability > speed, retry aggressively'
            }
        # Additional strategies...
```

**Practical Implications**: When a product manager asks "why is this slow?", you can point to the documented strategy and show the speed/quality/cost tradeoff. When an on-call engineer sees high costs, they can identify which features use QUALITY strategy and whether it's justified.

**Trade-offs**: Over-documenting configuration creates maintenance burden. Focus on parameters that: (1) significantly affect behavior, (2) aren't obvious, (3) resulted from measured decisions rather than guesses.

### 4. Failure Mode Documentation

LLM failures are often subtle. Document known failure patterns and diagnostic approaches.

```python
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class FailurePattern:
    """
    Documents a known failure mode with diagnostic guidance.
    """
    name: str
    description: str
    symptoms: List[str]
    detection_method: Callable
    mitigation: str
    example_input