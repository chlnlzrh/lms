# LLM Output Consistency Testing

## Core Concepts

LLM output consistency testing is the systematic validation of model response stability across identical or semantically equivalent inputs. Unlike deterministic systems where `f(x)` always returns the same output, LLMs introduce controlled randomness through temperature sampling, making each inference potentially unique. Consistency testing quantifies this variability and establishes acceptable variance bounds for production systems.

### Engineering Analogy: Deterministic vs. Probabilistic Systems

```python
# Traditional deterministic API
def calculate_discount(price: float, customer_tier: str) -> float:
    """Always returns identical output for same inputs"""
    discounts = {"bronze": 0.05, "silver": 0.10, "gold": 0.15}
    return price * (1 - discounts.get(customer_tier, 0))

# Deterministic behavior
assert calculate_discount(100.0, "gold") == 85.0
assert calculate_discount(100.0, "gold") == 85.0  # Always true

# LLM-based system
import anthropic
from typing import Dict

def extract_discount_llm(order_description: str) -> Dict[str, float]:
    """Returns potentially different outputs despite same input"""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        temperature=0.7,  # Introduces randomness
        messages=[{
            "role": "user",
            "content": f"Extract discount percentage from: {order_description}"
        }]
    )
    # Parse and return structured data
    return {"discount": parse_percentage(response.content[0].text)}

# Non-deterministic behavior
result1 = extract_discount_llm("Gold customer with 15% off")
result2 = extract_discount_llm("Gold customer with 15% off")
# result1 may != result2 due to sampling variance
```

The engineering challenge shifts from testing correctness to testing consistency—establishing variance thresholds and validating that outputs remain functionally equivalent even when textually different.

### Why This Matters Now

Production LLM systems face unique reliability challenges:

1. **Silent Degradation**: Model updates or prompt changes may increase variance without immediate detection
2. **Semantic Equivalence**: Different phrasings can be equally correct (e.g., "15%" vs "fifteen percent discount")
3. **Downstream Impact**: Inconsistent formatting breaks parsers; inconsistent reasoning breaks business logic
4. **Regulatory Requirements**: Financial, healthcare, and legal applications require documented stability metrics

Traditional integration tests catch complete failures but miss subtle drift in response patterns. Consistency testing detects these issues before they impact users.

## Technical Components

### 1. Variance Quantification Metrics

Measuring consistency requires metrics beyond simple string equality. Engineers need to capture both syntactic and semantic variance.

```python
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import difflib

class ConsistencyMetrics:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def exact_match_rate(self, responses: List[str]) -> float:
        """Strictest metric: percentage of identical responses"""
        if not responses:
            return 0.0
        most_common = Counter(responses).most_common(1)[0][1]
        return most_common / len(responses)
    
    def semantic_similarity_stats(self, responses: List[str]) -> dict:
        """Measure semantic consistency via embeddings"""
        embeddings = self.embedding_model.encode(responses)
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
        
        return {
            "mean_similarity": float(np.mean(upper_tri)),
            "min_similarity": float(np.min(upper_tri)),
            "std_similarity": float(np.std(upper_tri)),
            "variance": float(np.var(upper_tri))
        }
    
    def structural_consistency(self, responses: List[str]) -> dict:
        """Analyze structural patterns (length, format)"""
        lengths = [len(r) for r in responses]
        has_numbers = [bool(any(c.isdigit() for c in r)) for r in responses]
        
        return {
            "length_cv": np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0,
            "format_consistency": sum(has_numbers) / len(has_numbers),
            "avg_edit_distance": self._avg_edit_distance(responses)
        }
    
    def _avg_edit_distance(self, responses: List[str]) -> float:
        """Calculate normalized average Levenshtein distance"""
        if len(responses) < 2:
            return 0.0
        
        distances = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                ratio = difflib.SequenceMatcher(None, responses[i], responses[j]).ratio()
                distances.append(1 - ratio)  # Convert similarity to distance
        
        return float(np.mean(distances))

# Example usage
metrics = ConsistencyMetrics()
responses = [
    "The discount is 15%",
    "Discount: 15 percent",
    "15% discount applied",
    "You receive a 15% discount"
]

print(f"Exact match: {metrics.exact_match_rate(responses):.2%}")
print(f"Semantic: {metrics.semantic_similarity_stats(responses)}")
print(f"Structural: {metrics.structural_consistency(responses)}")
```

**Practical Implications**: Different metrics serve different purposes. Exact match detects prompt sensitivity. Semantic similarity catches meaning drift. Structural consistency validates parsing reliability. Production systems typically monitor all three with different thresholds.

**Trade-offs**: Embedding-based metrics require additional inference costs (~10-50ms per comparison). They also can mask critical differences—two semantically similar responses might have opposite sentiment or inverted numbers.

### 2. Temperature and Sampling Strategy Impact

Temperature controls the randomness distribution during token selection. Understanding its impact is critical for consistency testing strategy.

```python
import anthropic
from typing import List, Dict
import json

class SamplingExperiment:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def test_temperature_impact(
        self,
        prompt: str,
        temperatures: List[float],
        n_samples: int = 10
    ) -> Dict[float, dict]:
        """Measure variance across temperature settings"""
        results = {}
        
        for temp in temperatures:
            responses = []
            for _ in range(n_samples):
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    temperature=temp,
                    messages=[{"role": "user", "content": prompt}]
                )
                responses.append(response.content[0].text)
            
            metrics = ConsistencyMetrics()
            results[temp] = {
                "responses": responses,
                "exact_match": metrics.exact_match_rate(responses),
                "semantic": metrics.semantic_similarity_stats(responses),
                "structural": metrics.structural_consistency(responses)
            }
        
        return results
    
    def analyze_top_p_impact(
        self,
        prompt: str,
        top_p_values: List[float],
        temperature: float = 1.0,
        n_samples: int = 10
    ) -> Dict[float, dict]:
        """Nucleus sampling consistency analysis"""
        results = {}
        
        for top_p in top_p_values:
            responses = []
            for _ in range(n_samples):
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    temperature=temperature,
                    top_p=top_p,
                    messages=[{"role": "user", "content": prompt}]
                )
                responses.append(response.content[0].text)
            
            metrics = ConsistencyMetrics()
            results[top_p] = {
                "variance": np.var([len(r) for r in responses]),
                "semantic_std": metrics.semantic_similarity_stats(responses)["std_similarity"]
            }
        
        return results

# Example experiment
experiment = SamplingExperiment(api_key="your-key")
prompt = "Extract the dollar amount from: 'Customer paid $1,234.56 for premium service'"

temp_results = experiment.test_temperature_impact(
    prompt=prompt,
    temperatures=[0.0, 0.3, 0.7, 1.0],
    n_samples=20
)

for temp, data in temp_results.items():
    print(f"\nTemperature {temp}:")
    print(f"  Exact match: {data['exact_match']:.2%}")
    print(f"  Semantic similarity: {data['semantic']['mean_similarity']:.3f}")
    print(f"  Length variance: {data['structural']['length_cv']:.3f}")
```

**Practical Implications**: 
- Temperature 0.0 doesn't guarantee identical outputs due to floating-point operations, but variance is minimal
- Temperature > 0.5 often introduces significant response diversity
- Top-p (nucleus sampling) provides better control than temperature for bounded variance
- For production extraction tasks, temperature ≤ 0.3 typically required

**Constraints**: Lower temperatures reduce creativity and may cause mode collapse on certain prompts. Always test with actual use-case prompts, not synthetic examples.

### 3. Prompt Sensitivity Analysis

Small prompt variations can cause disproportionate output variance. Systematic testing identifies fragile prompt structures.

```python
from typing import List, Callable
import itertools

class PromptSensitivityTester:
    def __init__(self, llm_function: Callable[[str], str]):
        self.llm_function = llm_function
        self.metrics = ConsistencyMetrics()
    
    def test_phrasing_variants(
        self,
        base_prompt: str,
        variations: List[str],
        n_samples: int = 5
    ) -> dict:
        """Test semantically equivalent prompt phrasings"""
        all_results = {}
        
        for variant in variations:
            responses = [self.llm_function(variant) for _ in range(n_samples)]
            all_results[variant] = responses
        
        # Cross-variant consistency
        all_responses = list(itertools.chain(*all_results.values()))
        
        return {
            "per_variant": {
                k: self.metrics.semantic_similarity_stats(v)
                for k, v in all_results.items()
            },
            "cross_variant": self.metrics.semantic_similarity_stats(all_responses),
            "variance_ratio": self._calculate_variance_ratio(all_results)
        }
    
    def test_instruction_ordering(
        self,
        instructions: List[str],
        context: str,
        n_samples: int = 5
    ) -> dict:
        """Test if instruction order affects consistency"""
        results = {}
        
        # Test all permutations
        for perm in itertools.permutations(instructions):
            prompt = "\n".join(perm) + f"\n\n{context}"
            responses = [self.llm_function(prompt) for _ in range(n_samples)]
            results[str(perm)] = responses
        
        return {
            "ordering_matters": self._measure_ordering_impact(results),
            "optimal_order": self._find_most_consistent_order(results)
        }
    
    def test_delimiter_sensitivity(
        self,
        content: str,
        delimiters: List[Tuple[str, str]]
    ) -> dict:
        """Test consistency across different delimiter styles"""
        results = {}
        
        for open_delim, close_delim in delimiters:
            prompt = f"Extract data from {open_delim}{content}{close_delim}"
            responses = [self.llm_function(prompt) for _ in range(5)]
            results[f"{open_delim}...{close_delim}"] = {
                "responses": responses,
                "consistency": self.metrics.semantic_similarity_stats(responses)["mean_similarity"]
            }
        
        return results
    
    def _calculate_variance_ratio(self, results: dict) -> float:
        """Ratio of between-variant to within-variant variance"""
        within_var = np.mean([
            self.metrics.semantic_similarity_stats(responses)["variance"]
            for responses in results.values()
        ])
        
        all_responses = list(itertools.chain(*results.values()))
        total_var = self.metrics.semantic_similarity_stats(all_responses)["variance"]
        
        return float(total_var / within_var) if within_var > 0 else 0.0
    
    def _measure_ordering_impact(self, results: dict) -> float:
        """Quantify if order affects outputs"""
        all_responses = list(itertools.chain(*results.values()))
        return self.metrics.semantic_similarity_stats(all_responses)["std_similarity"]
    
    def _find_most_consistent_order(self, results: dict) -> str:
        """Identify instruction order with lowest variance"""
        consistencies = {
            order: self.metrics.semantic_similarity_stats(responses)["mean_similarity"]
            for order, responses in results.items()
        }
        return max(consistencies.items(), key=lambda x: x[1])[0]

# Example usage
def mock_llm(prompt: str) -> str:
    # Replace with actual LLM call
    return "Extracted: $1,234.56"

tester = PromptSensitivityTester(mock_llm)

# Test phrasing variants
variants = [
    "Extract the dollar amount from the text below:",
    "Please extract the dollar amount from:",
    "Find the dollar amount in:",
    "What is the dollar amount in the following text:"
]

phrasing_results = tester.test_phrasing_variants("Sample text: $1,234.56", variants)
print(f"Cross-variant consistency: {phrasing_results['cross_variant']['mean_similarity']:.3f}")
print(f"Variance ratio: {phrasing_results['variance_ratio']:.3f}")
```

**Practical Implications**: Prompts with variance ratio > 2.0 indicate high sensitivity—small phrasings cause large output changes. These prompts need stabilization through more explicit instructions or examples.

**Trade-offs**: Testing all permutations is exponential. Focus on high-impact variations: instruction order, delimiter styles, politeness levels, explicit vs. implicit instructions.

### 4. Regression Detection Frameworks

Production systems need continuous monitoring to detect consistency degradation over time.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import sqlite3
import hashlib

@dataclass
class ConsistencyBaseline:
    prompt_hash: str
    temperature: float
    mean_semantic_similarity: float
    std_semantic_similarity: float
    exact_match_rate: float
    sample_size: int
    timestamp: datetime

class ConsistencyRegressionDetector:
    def __init__(self, db_path: str = "consistency_baselines.db"):
        self.db_path = db_path
        self.metrics = ConsistencyMetrics()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for baseline storage"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                prompt_hash TEXT,
                temperature REAL,
                mean_similarity REAL,
                std_similarity REAL,
                exact_match_rate REAL,
                sample_size INTEGER,
                timestamp TEXT,
                PRIMARY KEY (prompt_hash, temperature, timestamp)
            )
        """)
        conn.commit()
        conn.close()
    
    def establish_baseline(
        self,
        prompt