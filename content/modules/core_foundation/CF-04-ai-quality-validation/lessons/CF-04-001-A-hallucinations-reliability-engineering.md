# Hallucinations & Reliability Engineering

## Core Concepts

### Technical Definition

Hallucinations in language models are outputs that appear syntactically valid and semantically coherent but contain factually incorrect information, logical inconsistencies, or fabricated references. Unlike traditional software bugs that manifest deterministically, hallucinations emerge from the probabilistic nature of token prediction—the model generates plausible text based on learned patterns rather than verified knowledge retrieval.

This isn't a bug to be "fixed" in the traditional sense. It's an inherent characteristic of how autoregressive language models work: they predict the most probable next token given the context, without a built-in mechanism to verify whether that prediction corresponds to factual reality.

### Engineering Analogy: Traditional vs. LLM-Based Systems

```python
from typing import Optional, Dict, List
import hashlib

# Traditional Deterministic System
class TraditionalDatabase:
    def __init__(self):
        self.facts: Dict[str, str] = {
            "python_release": "1991",
            "creator": "Guido van Rossum"
        }
    
    def query(self, key: str) -> Optional[str]:
        """Returns exact value or None - no hallucination possible"""
        result = self.facts.get(key)
        if result is None:
            raise KeyError(f"No data for key: {key}")
        return result
    
    def verify(self, key: str, value: str) -> bool:
        """Deterministic verification"""
        return self.facts.get(key) == value

# LLM-Based System (Simplified Representation)
class LLMSystem:
    def __init__(self):
        # Model has learned statistical associations, not verified facts
        self.learned_patterns: Dict[str, List[tuple[str, float]]] = {
            "python_release": [
                ("1991", 0.85),  # High probability, correct
                ("1989", 0.10),  # Lower probability, incorrect
                ("1990", 0.05)   # Even lower, incorrect
            ]
        }
    
    def query(self, key: str, temperature: float = 0.7) -> str:
        """Returns probabilistic answer - hallucination possible"""
        import random
        
        if key not in self.learned_patterns:
            # Model doesn't "know" it doesn't know - will generate plausible answer
            return self._generate_plausible_response(key)
        
        options = self.learned_patterns[key]
        # Temperature affects randomness - higher = more creative/risky
        if temperature == 0:
            return options[0][0]  # Most probable
        else:
            # Weighted random selection - can return incorrect answer
            weights = [prob ** (1/temperature) for _, prob in options]
            return random.choices([val for val, _ in options], weights=weights)[0]
    
    def _generate_plausible_response(self, key: str) -> str:
        """When model has no strong pattern, generates based on surface patterns"""
        # This is where hallucinations often occur
        return f"Based on context patterns, approximately 1995"  # Wrong but plausible
    
    def verify(self, key: str, value: str) -> Optional[bool]:
        """Cannot deterministically verify - no ground truth"""
        # LLM has no access to absolute truth
        confidence = self._check_pattern_strength(key, value)
        return None  # Cannot provide boolean certainty


# Demonstration
if __name__ == "__main__":
    trad_db = TraditionalDatabase()
    llm_sys = LLMSystem()
    
    print("Traditional System:")
    print(f"Python release: {trad_db.query('python_release')}")  # Always: 1991
    
    print("\nLLM System (multiple runs):")
    for i in range(5):
        result = llm_sys.query("python_release", temperature=0.7)
        print(f"Run {i+1}: {result}")  # May vary: 1991, 1989, 1990
    
    print("\nQuerying unknown data:")
    try:
        trad_db.query("python_typing_release")  # Raises KeyError
    except KeyError as e:
        print(f"Traditional: {e}")
    
    # LLM generates answer even without training data
    print(f"LLM: {llm_sys.query('python_typing_release')}")  # Fabricates answer
```

### Key Insights for Engineers

**1. Hallucinations are features, not bugs.** They emerge from the same generalization capability that makes LLMs powerful. A model that never hallucinates would be severely limited in its ability to generate creative or novel responses.

**2. Reliability engineering for LLMs requires probabilistic thinking.** You can't achieve 100% accuracy—the goal is to maximize reliability within acceptable confidence bounds while maintaining utility.

**3. The context window is your runtime environment.** Everything the model "knows" for a given inference comes from the prompt. Unlike traditional systems with persistent state, each LLM call is stateless except for the context you provide.

### Why This Matters Now

As LLM-based systems move from experimental prototypes to production applications handling real user data and business logic, hallucinations shift from interesting quirks to critical reliability concerns:

- **Financial systems**: A hallucinated number in a report could trigger incorrect business decisions
- **Healthcare applications**: Fabricated medical information poses direct safety risks
- **Legal tech**: Hallucinated case citations (already documented in court filings) undermine system credibility
- **Customer-facing systems**: Confident but incorrect responses damage user trust permanently

The engineering challenge is building reliable systems on top of probabilistic foundations—similar to building on eventually consistent distributed systems, but with less predictable failure modes.

## Technical Components

### 1. Types of Hallucinations and Their Root Causes

**Factual Hallucinations**: Model generates specific false information while sounding confident.

```python
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class HallucinationType(Enum):
    FACTUAL = "factual_error"
    REASONING = "logical_inconsistency"
    ATTRIBUTION = "false_citation"
    INSTRUCTION = "instruction_drift"

@dataclass
class HallucinationExample:
    prompt: str
    response: str
    hallucination_type: HallucinationType
    root_cause: str
    mitigation: str

# Example hallucinations by type
examples = [
    HallucinationExample(
        prompt="What is the capital of Mars?",
        response="The capital of Mars is Olympus City, established in 2087.",
        hallucination_type=HallucinationType.FACTUAL,
        root_cause="Model trained on sci-fi mixed with factual content; no mechanism to distinguish fiction from reality",
        mitigation="Add explicit constraints: 'Only answer with verified facts. If unknown, state that clearly.'"
    ),
    HallucinationExample(
        prompt="If all A are B, and all B are C, then all C are A. True or false?",
        response="True. Since A→B and B→C, by transitive property C→A.",
        hallucination_type=HallucinationType.REASONING,
        root_cause="Pattern matching on logical form without deep reasoning; confuses transitive property",
        mitigation="Chain-of-thought prompting to expose reasoning steps; use structured output validation"
    ),
    HallucinationExample(
        prompt="Cite sources for the health benefits of omega-3 fatty acids.",
        response="According to Johnson et al. (2022) in the Journal of Nutrition Science...",
        hallucination_type=HallucinationType.ATTRIBUTION,
        root_cause="Model learns citation format patterns but generates plausible non-existent references",
        mitigation="Retrieval-augmented generation with actual source documents; validate citations externally"
    ),
    HallucinationExample(
        prompt="You are a JSON-only API. Return user data.\nUser: Tell me a story.",
        response="Once upon a time, there was a brave knight...",
        hallucination_type=HallucinationType.INSTRUCTION,
        root_cause="User prompt overrides system instruction; insufficient reinforcement of output format",
        mitigation="Use structured output schemas; validate format before returning; stronger system prompts"
    )
]

def analyze_hallucination_risk(prompt: str, response: str) -> Dict[str, float]:
    """
    Simple heuristic-based hallucination risk assessment.
    Production systems would use trained classifiers.
    """
    risks = {
        "specificity_without_source": 0.0,
        "logical_complexity": 0.0,
        "format_deviation": 0.0,
        "confidence_indicators": 0.0
    }
    
    # High specificity (dates, names, numbers) without citations = higher risk
    specific_patterns = ["in 2", "19", "20", "Dr.", "Professor", "%", "$"]
    citation_patterns = ["according to", "source:", "retrieved from"]
    
    has_specifics = any(pattern in response for pattern in specific_patterns)
    has_citations = any(pattern in response.lower() for pattern in citation_patterns)
    
    if has_specifics and not has_citations:
        risks["specificity_without_source"] = 0.7
    
    # Logical chains without verification steps
    if any(word in response.lower() for word in ["therefore", "thus", "consequently"]):
        if "let's verify" not in response.lower() and "checking:" not in response.lower():
            risks["logical_complexity"] = 0.5
    
    # Confidence without hedging
    confident_phrases = ["definitely", "certainly", "always", "never", "impossible"]
    if any(phrase in response.lower() for phrase in confident_phrases):
        risks["confidence_indicators"] = 0.6
    
    return risks

# Test the analyzer
for example in examples[:2]:
    print(f"\nAnalyzing: {example.hallucination_type.value}")
    print(f"Response: {example.response[:100]}...")
    risks = analyze_hallucination_risk(example.prompt, example.response)
    print(f"Risk scores: {risks}")
    print(f"Mitigation: {example.mitigation}")
```

**Practical Implications:**
- Different hallucination types require different mitigation strategies
- Factual hallucinations can be reduced with retrieval augmentation
- Reasoning hallucinations need explicit step verification
- Attribution hallucinations require external validation pipelines
- Instruction drift needs stronger prompt engineering and format validation

### 2. Confidence Calibration and Uncertainty Quantification

LLMs express linguistic confidence but this doesn't correlate reliably with factual accuracy. A model can be confidently wrong.

```python
import re
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ConfidenceAnalysis:
    linguistic_confidence: float  # How confident the language sounds
    factual_reliability: float    # Actual likelihood of correctness
    uncertainty_markers: List[str]
    recommendation: str

def analyze_response_confidence(response: str) -> ConfidenceAnalysis:
    """
    Analyze both linguistic confidence and reliability indicators.
    """
    # Linguistic confidence markers (high confidence language)
    high_confidence_markers = [
        r'\bdefinitely\b', r'\bcertainly\b', r'\balways\b', 
        r'\bnever\b', r'\bimpossible\b', r'\bobviously\b',
        r'\bclearly\b', r'\bundoubtedly\b'
    ]
    
    # Uncertainty/hedging markers (appropriate epistemic humility)
    uncertainty_markers = [
        r'\bmight\b', r'\bcould\b', r'\bpossibly\b', r'\blikely\b',
        r'\bprobably\b', r'\bseems\b', r'\bappears\b', r'\bmay\b',
        r'\bin my understanding\b', r'\bI believe\b'
    ]
    
    # Fact-checking indicators (good practices)
    verification_markers = [
        r'\baccording to\b', r'\bsource:\b', r'\bas of\b',
        r'\blast verified\b', r'\bcitation:\b'
    ]
    
    response_lower = response.lower()
    
    high_conf_count = sum(1 for pattern in high_confidence_markers 
                          if re.search(pattern, response_lower))
    uncertainty_count = sum(1 for pattern in uncertainty_markers 
                           if re.search(pattern, response_lower))
    verification_count = sum(1 for pattern in verification_markers 
                            if re.search(pattern, response_lower))
    
    # Calculate linguistic confidence (0-1)
    linguistic_confidence = min(1.0, high_conf_count * 0.3 + 
                               (1 - uncertainty_count * 0.2))
    
    # Calculate factual reliability estimate (0-1)
    # High verification markers = higher reliability
    # High confidence without verification = lower reliability
    if verification_count > 0:
        factual_reliability = min(0.9, 0.5 + verification_count * 0.2)
    elif high_conf_count > 2 and uncertainty_count == 0:
        factual_reliability = 0.3  # Confident without backing = risky
    else:
        factual_reliability = 0.5  # Neutral baseline
    
    # Generate recommendation
    if linguistic_confidence > 0.7 and factual_reliability < 0.5:
        recommendation = "HIGH RISK: Confident language without verification. Requires fact-checking."
    elif factual_reliability > 0.7:
        recommendation = "ACCEPTABLE: Contains verification markers. Still validate critical facts."
    elif uncertainty_count > 2:
        recommendation = "CAUTIOUS: Model expressing uncertainty. Verify before using."
    else:
        recommendation = "MEDIUM RISK: Standard output. Apply standard validation."
    
    found_uncertainty = [pattern for pattern in uncertainty_markers 
                        if re.search(pattern, response_lower)]
    
    return ConfidenceAnalysis(
        linguistic_confidence=linguistic_confidence,
        factual_reliability=factual_reliability,
        uncertainty_markers=found_uncertainty[:5],  # Top 5
        recommendation=recommendation
    )

# Test cases
test_responses = [
    "Python was definitely released in 1991 by Guido van Rossum. This is a well-documented fact.",
    "Python was released around 1991, according to historical records. The exact month may vary in sources.",
    "Python was certainly invented in 1985 and is the oldest programming language still in use.",
    "According to the Python Software Foundation documentation (last verified 2024), Python was released in 1991."
]

print("CONFIDENCE CALIBRATION ANALYSIS\n")
for i, response in enumerate(test_responses, 1):
    analysis = analyze_response_confidence(response)
    print(f"Response {i}:")
    print(f"  Text: {response[:80]}...")
    print(f"  Linguistic Confidence: {analysis.linguistic_confidence:.2f}")
    print(f"  Factual Reliability Estimate: {analysis.factual_reliability:.2f}")
    print(f"  Recommendation: {analysis.recommendation}")
    print()
```

**Real Constraints:**
- Linguistic confidence ≠ factual accuracy
- Token-level probabilities (logprobs) are available via API but don't directly translate to statement-level accuracy
- Multiple high-probability tokens can combine into a low-probability (hallucinated) fact
- Temperature settings affect confidence distribution but not accuracy

### 3. Retrieval-Augmented Generation (RAG) Architecture

RAG reduces hallucinations by grounding responses in retrieved facts, but introduces new failure modes.

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, str]
    relevance_score: float = 0.0

class SimpleRAGSystem:
    """
    Simplified RAG architecture demonstrating key components.
    Production systems would use vector databases and LLM APIs.
    """
    
    def __init__(self):
        # Simulated knowledge base
        self.knowledge_base: List[Document] = [
            Document(
                id="doc1",
                content="Python was released in 1991 by Guido van Rossum.",
                metadata={"source": "python.org", "date": "2024",