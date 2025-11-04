# Outcome-Based Pricing for LLM Applications

## Core Concepts

Outcome-based pricing shifts the economic model for LLM applications from metering computational resources (tokens, API calls) to charging based on delivered value. Instead of billing per token processed, you charge for successful outcomes: a completed analysis, a verified code generation, a satisfied user query, or a business transaction enabled.

This pricing model transforms how you architect LLM systems. Traditional token-based pricing encourages minimizing context and output length. Outcome-based pricing incentivizes reliability, quality, and user success—even if that requires multiple attempts, larger contexts, or chain-of-thought reasoning.

### Engineering Analogy: Infrastructure Cost Abstraction

Consider traditional vs. outcome-based approaches:

**Traditional Token-Based Approach:**

```python
from typing import Optional
import anthropic

class TokenBasedService:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cost_per_input_token = 0.000003  # $3 per MTok
        self.cost_per_output_token = 0.000015  # $15 per MTok
    
    def analyze_document(self, document: str) -> dict:
        """User pays per token, regardless of quality."""
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,  # Artificially limited to control costs
            messages=[{
                "role": "user",
                "content": f"Analyze: {document[:2000]}"  # Truncated
            }]
        )
        
        cost = (
            response.usage.input_tokens * self.cost_per_input_token +
            response.usage.output_tokens * self.cost_per_output_token
        )
        
        return {
            "analysis": response.content[0].text,
            "cost": cost,
            "quality": "unknown"  # User pays even if garbage
        }
```

**Outcome-Based Approach:**

```python
from typing import Optional, Literal
import anthropic
from dataclasses import dataclass
from enum import Enum

class OutcomeQuality(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"

@dataclass
class OutcomeResult:
    analysis: str
    outcome_quality: OutcomeQuality
    confidence: float
    attempts: int
    total_tokens: int
    billable: bool

class OutcomeBasedService:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.price_per_success = 0.50  # Fixed price per successful analysis
    
    def analyze_document(
        self, 
        document: str,
        quality_threshold: float = 0.8,
        max_attempts: int = 3
    ) -> OutcomeResult:
        """User pays only for successful outcomes."""
        total_tokens = 0
        
        for attempt in range(max_attempts):
            # No artificial limits - optimize for quality
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,  # Allow full analysis
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze this document and provide:
1. Key insights (bullet points)
2. Risk assessment
3. Confidence score (0-1) for your analysis

Document: {document}"""
                    }
                ]
            )
            
            total_tokens += response.usage.input_tokens + response.usage.output_tokens
            analysis = response.content[0].text
            
            # Verify quality before charging
            confidence = self._extract_confidence(analysis)
            
            if confidence >= quality_threshold:
                return OutcomeResult(
                    analysis=analysis,
                    outcome_quality=OutcomeQuality.SUCCESS,
                    confidence=confidence,
                    attempts=attempt + 1,
                    total_tokens=total_tokens,
                    billable=True
                )
            
            # Retry with refined prompt if quality insufficient
            if attempt < max_attempts - 1:
                document = self._add_context_from_failure(document, analysis)
        
        # Failed to achieve quality - don't charge
        return OutcomeResult(
            analysis=analysis,
            outcome_quality=OutcomeQuality.FAILURE,
            confidence=confidence,
            attempts=max_attempts,
            total_tokens=total_tokens,
            billable=False
        )
    
    def _extract_confidence(self, analysis: str) -> float:
        """Extract confidence score from analysis."""
        # Implementation would parse confidence from structured output
        import re
        match = re.search(r'confidence.*?(\d+\.?\d*)', analysis.lower())
        return float(match.group(1)) if match else 0.0
    
    def _add_context_from_failure(self, document: str, previous: str) -> str:
        """Enhance context for retry."""
        return f"{document}\n\nPrevious incomplete analysis: {previous}"
```

### Key Insights for Engineering Thinking

**1. Cost Becomes a Guarantee, Not a Variable:** You absorb token-level volatility and expose only outcome-level pricing. This requires sophisticated retry logic, quality verification, and cost modeling.

**2. Quality Verification is Infrastructure:** You must programmatically determine if an outcome succeeded. This means structured outputs, confidence scoring, and validation layers become first-class architecture concerns.

**3. Economic Incentives Align with Technical Decisions:** Multi-shot prompting, chain-of-thought, and retrieval augmentation are now cost-neutral to users, enabling better solutions.

**4. Risk Transfer Changes System Design:** You now own the risk of model failures, prompt sensitivity, and API instability. This demands defensive architecture: circuit breakers, fallbacks, and cost caps.

### Why This Matters Now

**Economic Pressure:** Token costs vary 100x across models (GPT-4 vs. GPT-3.5-turbo). Outcome pricing lets you optimize model selection transparently while maintaining stable user pricing.

**Product Differentiation:** Enterprise customers prefer predictable costs tied to value. A $50/user/month seat license is easier to budget than "unknown tokens consumed."

**Competitive Moat:** Reliable outcome delivery requires engineering sophistication. It's harder to replicate than simple token passthrough.

**Model Independence:** Abstract away model pricing changes. When a model gets 2x cheaper, you improve margins rather than repricing customers.

## Technical Components

### 1. Outcome Definition and Verification

**Technical Explanation:**

An "outcome" must be programmatically verifiable. Unlike subjective quality, outcomes require boolean or scored validation: Did the code compile? Was the email sentiment positive? Did the extraction match schema?

**Practical Implementation:**

```python
from typing import Protocol, TypeVar, Generic
from abc import abstractmethod
from pydantic import BaseModel, Field
import json

T = TypeVar('T')

class OutcomeValidator(Protocol[T]):
    """Protocol for outcome verification."""
    
    @abstractmethod
    def validate(self, result: str) -> tuple[bool, T, float]:
        """
        Returns:
            (is_valid, parsed_result, confidence_score)
        """
        ...

class CodeGenerationOutcome(BaseModel):
    """Structured outcome for code generation."""
    code: str
    language: str
    test_cases: list[str]
    
class CodeGenerationValidator:
    def validate(self, result: str) -> tuple[bool, CodeGenerationOutcome, float]:
        """Validates generated code compiles and has tests."""
        try:
            # Extract structured JSON from response
            parsed = json.loads(result)
            outcome = CodeGenerationOutcome(**parsed)
            
            # Verify code syntax
            compile(outcome.code, '<string>', 'exec')
            
            # Check test coverage
            has_tests = len(outcome.test_cases) >= 2
            
            # Calculate confidence
            confidence = 0.6  # Base
            if has_tests:
                confidence += 0.3
            if len(outcome.code) > 50:  # Not trivial
                confidence += 0.1
            
            is_valid = has_tests and confidence >= 0.8
            
            return is_valid, outcome, confidence
            
        except (json.JSONDecodeError, SyntaxError, KeyError) as e:
            # Invalid outcome - no charge
            return False, None, 0.0

class DataExtractionOutcome(BaseModel):
    """Outcome for structured extraction."""
    entities: list[dict[str, str]]
    completeness: float = Field(ge=0.0, le=1.0)

class DataExtractionValidator:
    def __init__(self, required_fields: set[str]):
        self.required_fields = required_fields
    
    def validate(self, result: str) -> tuple[bool, DataExtractionOutcome, float]:
        """Validates extraction completeness."""
        try:
            parsed = json.loads(result)
            outcome = DataExtractionOutcome(**parsed)
            
            # Check all entities have required fields
            valid_entities = [
                e for e in outcome.entities
                if self.required_fields.issubset(e.keys())
            ]
            
            completeness = len(valid_entities) / len(outcome.entities) if outcome.entities else 0
            confidence = completeness * outcome.completeness
            
            is_valid = completeness >= 0.9 and confidence >= 0.8
            
            return is_valid, outcome, confidence
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return False, None, 0.0
```

**Real Constraints:**

- **Verification Cost:** Validation logic runs on every attempt. Keep it fast (<100ms) or it becomes a bottleneck.
- **False Negatives:** Overly strict validation rejects good outcomes, increasing retry costs.
- **Subjectivity:** Some outcomes (creative writing, tone) resist objective validation. Consider human-in-the-loop or user acceptance as the outcome.

### 2. Retry and Fallback Architecture

**Technical Explanation:**

Since you absorb cost risk, you need strategies to achieve outcomes within budget constraints. This requires retry policies, model cascading, and graceful degradation.

**Practical Implementation:**

```python
from typing import Callable, Optional, Any
from dataclasses import dataclass
import asyncio
from enum import Enum

class ModelTier(Enum):
    PREMIUM = "claude-3-5-sonnet-20241022"
    STANDARD = "claude-3-haiku-20240307"
    FAST = "claude-3-haiku-20240307"

@dataclass
class ModelConfig:
    name: str
    cost_per_token: float
    max_tokens: int
    timeout: float

@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0
    model_cascade: list[ModelTier] = None

class OutcomeOrchestrator:
    def __init__(self, api_key: str, cost_cap_per_outcome: float = 0.50):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cost_cap = cost_cap_per_outcome
        
        self.models = {
            ModelTier.PREMIUM: ModelConfig("claude-3-5-sonnet-20241022", 0.000015, 4096, 30.0),
            ModelTier.STANDARD: ModelConfig("claude-3-haiku-20240307", 0.000005, 2048, 15.0),
            ModelTier.FAST: ModelConfig("claude-3-haiku-20240307", 0.000005, 1024, 10.0),
        }
    
    async def execute_with_fallback(
        self,
        prompt: str,
        validator: OutcomeValidator,
        policy: RetryPolicy
    ) -> OutcomeResult:
        """Execute with model cascading and retries."""
        
        cumulative_cost = 0.0
        total_tokens = 0
        
        model_tiers = policy.model_cascade or [ModelTier.PREMIUM, ModelTier.STANDARD, ModelTier.FAST]
        
        for tier_idx, tier in enumerate(model_tiers):
            model = self.models[tier]
            attempts = policy.max_attempts if tier_idx == 0 else 2  # Fewer retries on fallbacks
            
            for attempt in range(attempts):
                # Check cost cap
                if cumulative_cost >= self.cost_cap:
                    return OutcomeResult(
                        analysis="Cost cap exceeded",
                        outcome_quality=OutcomeQuality.FAILURE,
                        confidence=0.0,
                        attempts=attempt + 1,
                        total_tokens=total_tokens,
                        billable=False
                    )
                
                try:
                    response = await asyncio.wait_for(
                        self._execute_model(model, prompt),
                        timeout=model.timeout
                    )
                    
                    tokens = response.usage.input_tokens + response.usage.output_tokens
                    cost = tokens * model.cost_per_token
                    
                    cumulative_cost += cost
                    total_tokens += tokens
                    
                    # Validate outcome
                    is_valid, parsed, confidence = validator.validate(response.content[0].text)
                    
                    if is_valid:
                        return OutcomeResult(
                            analysis=response.content[0].text,
                            outcome_quality=OutcomeQuality.SUCCESS,
                            confidence=confidence,
                            attempts=attempt + 1,
                            total_tokens=total_tokens,
                            billable=True
                        )
                    
                    # Retry with delay
                    if attempt < attempts - 1:
                        delay = policy.initial_delay * (policy.backoff_multiplier ** attempt)
                        await asyncio.sleep(delay)
                
                except asyncio.TimeoutError:
                    # Try faster model
                    break
                except Exception as e:
                    # Log and retry
                    if attempt == attempts - 1:
                        break
        
        # All attempts exhausted
        return OutcomeResult(
            analysis="Failed to achieve outcome within constraints",
            outcome_quality=OutcomeQuality.FAILURE,
            confidence=0.0,
            attempts=policy.max_attempts * len(model_tiers),
            total_tokens=total_tokens,
            billable=False
        )
    
    async def _execute_model(self, model: ModelConfig, prompt: str):
        """Execute single model call."""
        return self.client.messages.create(
            model=model.name,
            max_tokens=model.max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
```

**Real Constraints:**

- **Latency vs. Cost Trade-off:** Retries increase success rate but add latency. Set aggressive timeouts (5-15s per attempt).
- **Cost Cap Breaches:** 5-10% of requests may exceed caps. Decide: eat the cost, or fail gracefully?
- **Model Compatibility:** Cascading to cheaper models assumes prompt compatibility. Haiku may fail where Sonnet succeeds.

### 3. Cost Modeling and Margin Management

**Technical Explanation:**

To price outcomes profitably, model expected token usage, success rates, and retry costs across your request distribution.

**Practical Implementation:**

```python
from collections import defaultdict
from datetime import datetime, timedelta
import statistics

class CostModel:
    def __init__(self):
        self.outcomes = defaultdict(list)
        self.costs = defaultdict(list)
    
    def record_outcome(
        self, 
        outcome_type: str,
        success: bool,
        tokens_used: int,
        attempts: int,
        actual_cost: float
    ):
        """Record outcome metrics for analysis."""
        self.outcomes[outcome_type].append({
            "timestamp": datetime.utcnow(),
            "success": success,
            "tokens": tokens_used,
            "attempts": attempts,
            "cost": actual_cost
        })
    
    def calculate_target_price(
        self, 
        outcome_type: str,
        target_margin: float = 0.40  # 40% margin
    ) -> dict[str