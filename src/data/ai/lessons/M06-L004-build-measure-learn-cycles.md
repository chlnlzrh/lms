# Build-Measure-Learn Cycles for LLM Applications

## Core Concepts

Build-Measure-Learn (BML) cycles represent a systematic approach to developing LLM-powered systems through rapid iteration, quantitative feedback, and empirical validation. Unlike traditional software where behavior is deterministic and testable through unit tests, LLM applications produce probabilistic outputs that require statistical evaluation across diverse inputs.

### Engineering Analogy: Traditional vs. LLM Development

**Traditional API Development:**
```python
# Traditional: Deterministic, test-driven
def calculate_discount(price: float, customer_tier: str) -> float:
    """Pure function with predictable outputs."""
    discount_rates = {"gold": 0.2, "silver": 0.1, "bronze": 0.05}
    return price * (1 - discount_rates.get(customer_tier, 0))

# Test once, works forever
assert calculate_discount(100, "gold") == 80.0
```

**LLM Application Development:**
```python
import anthropic
from typing import List, Dict
import json
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    input: str
    output: str
    score: float
    metadata: Dict

# Non-deterministic, requires statistical validation
def extract_entities(text: str, client: anthropic.Anthropic) -> Dict:
    """Probabilistic function requiring empirical validation."""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Extract person names, companies, and dates from:\n{text}\nReturn JSON."
        }]
    )
    return json.loads(response.content[0].text)

# Must test across distribution of inputs
test_cases = [
    "John Smith joined Microsoft on Jan 5, 2024.",
    "Dr. María García founded TechCorp in 2020.",
    "The CEO, whose name wasn't disclosed, resigned yesterday."
]

# Single test proves nothing—need statistical confidence
results: List[EvaluationResult] = []
for case in test_cases:
    output = extract_entities(case, client)
    score = evaluate_extraction(output, ground_truth[case])
    results.append(EvaluationResult(case, output, score, {}))

print(f"Mean accuracy: {sum(r.score for r in results) / len(results):.2%}")
```

### Key Insights

**1. Shift from Binary Pass/Fail to Distribution Analysis**

Traditional software uses assertions. LLM applications require measuring performance distributions, identifying edge cases through statistics, and understanding failure modes empirically.

**2. Measurement Infrastructure Becomes Primary Engineering Artifact**

Your evaluation harness is more valuable than initial prompts. Robust measurement enables rapid experimentation. Without metrics, you're optimizing blind.

**3. Learning Velocity Determines Competitive Advantage**

The team that can run 100 experiments in a week beats the team running 10, even if their initial approach is inferior. Automation of the BML cycle is critical infrastructure.

### Why This Matters Now

LLM capabilities evolve monthly. Your prompt that achieved 85% accuracy in January may hit 92% with a new model in March—or drop to 78% due to behavior changes. BML cycles provide:

- **Adaptation velocity**: Quickly validate whether new models improve your specific use case
- **Regression detection**: Catch when updates break existing workflows
- **Cost optimization**: Empirically determine whether a cheaper model maintains quality
- **Reliability confidence**: Quantify system behavior under production conditions

## Technical Components

### 1. Evaluation Dataset Construction

**Technical Explanation:**

An evaluation dataset is a versioned collection of input-output pairs with ground truth labels or scoring criteria. Quality datasets are small (50-500 examples), representative of production distribution, and include edge cases discovered through failure analysis.

**Practical Implementation:**

```python
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

@dataclass
class EvalExample:
    id: str
    input: str
    expected_output: str
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            # Generate deterministic ID from content
            content = f"{self.input}{self.expected_output}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class EvalDataset:
    name: str
    examples: List[EvalExample]
    version: str
    
    def add_production_failure(self, input: str, actual: str, 
                              corrected: str, metadata: Dict):
        """Continuously improve dataset from production errors."""
        example = EvalExample(
            id="",
            input=input,
            expected_output=corrected,
            metadata={**metadata, "source": "production_failure", 
                     "failed_output": actual}
        )
        self.examples.append(example)
    
    def stratified_sample(self, n: int, key: str) -> 'EvalDataset':
        """Sample while maintaining distribution of metadata key."""
        from collections import defaultdict
        import random
        
        strata = defaultdict(list)
        for ex in self.examples:
            strata[ex.metadata.get(key, "unknown")].append(ex)
        
        samples = []
        per_stratum = n // len(strata)
        for examples in strata.values():
            samples.extend(random.sample(examples, 
                          min(per_stratum, len(examples))))
        
        return EvalDataset(f"{self.name}_sample", samples, self.version)
    
    def to_jsonl(self, path: str):
        """Serialize for version control."""
        with open(path, 'w') as f:
            for ex in self.examples:
                f.write(json.dumps({
                    'id': ex.id,
                    'input': ex.input,
                    'expected_output': ex.expected_output,
                    'metadata': ex.metadata,
                    'created_at': ex.created_at.isoformat()
                }) + '\n')

# Usage
dataset = EvalDataset("entity_extraction", [], "v1.2")

# Start with hand-crafted examples covering key scenarios
dataset.examples.extend([
    EvalExample("", 
        "CEO Jane Doe joined Acme Corp on 3/15/2024",
        '{"people": ["Jane Doe"], "companies": ["Acme Corp"], "dates": ["3/15/2024"]}',
        {"complexity": "simple", "entities": 3}
    ),
    EvalExample("",
        "The executive, whose identity remains confidential, resigned.",
        '{"people": [], "companies": [], "dates": []}',
        {"complexity": "ambiguous", "entities": 0}
    )
])

# Add production failures
dataset.add_production_failure(
    input="Dr. O'Brien met with representatives from AT&T.",
    actual='{"people": ["Dr. O", "Brien"], "companies": ["AT", "T"]}',
    corrected='{"people": ["Dr. O\'Brien"], "companies": ["AT&T"]}',
    metadata={"failure_type": "punctuation_handling"}
)
```

**Constraints & Trade-offs:**

- **Size vs. Cost**: 500 examples at $0.01/eval = $5/run. Acceptable for hourly cycles, prohibitive for per-commit CI.
- **Representativeness vs. Availability**: Early-stage products lack production data. Start with synthetic examples, continuously replace with real failures.
- **Stability vs. Evolution**: Datasets must be versioned. Changing evaluation data invalidates comparisons.

### 2. Automated Scoring Functions

**Technical Explanation:**

Scoring functions convert model outputs into numerical metrics. Types include: exact match (deterministic), semantic similarity (embedding-based), LLM-as-judge (flexible but costly), and rule-based heuristics (domain-specific).

**Implementation:**

```python
from typing import Protocol, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class ScoringFunction(Protocol):
    def score(self, predicted: str, expected: str) -> float:
        """Return score in [0, 1]."""
        ...

class ExactMatchScorer:
    """Fast, deterministic, but brittle."""
    
    def score(self, predicted: str, expected: str) -> float:
        return 1.0 if predicted.strip() == expected.strip() else 0.0

class EmbeddingScorer:
    """Semantic similarity, handles paraphrasing."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model_name
    
    def _embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def score(self, predicted: str, expected: str) -> float:
        pred_emb = self._embed(predicted)
        exp_emb = self._embed(expected)
        similarity = cosine_similarity([pred_emb], [exp_emb])[0][0]
        return float(np.clip(similarity, 0, 1))

class LLMJudgeScorer:
    """Flexible evaluation for complex criteria."""
    
    def __init__(self, client: anthropic.Anthropic, criteria: str):
        self.client = client
        self.criteria = criteria
    
    def score(self, predicted: str, expected: str) -> float:
        prompt = f"""Score the predicted output against expected output.
        
Criteria: {self.criteria}

Expected: {expected}
Predicted: {predicted}

Return only a number between 0-100."""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        score_text = response.content[0].text.strip()
        score = float(re.search(r'\d+', score_text).group())
        return score / 100.0

class CompositeScorer:
    """Combine multiple scoring functions with weights."""
    
    def __init__(self, scorers: List[tuple[ScoringFunction, float]]):
        self.scorers = scorers
        total_weight = sum(w for _, w in scorers)
        self.scorers = [(s, w/total_weight) for s, w in scorers]
    
    def score(self, predicted: str, expected: str) -> float:
        return sum(
            scorer.score(predicted, expected) * weight
            for scorer, weight in self.scorers
        )

# Example: Multi-faceted evaluation
evaluator = CompositeScorer([
    (ExactMatchScorer(), 0.3),  # Reward exact matches highly
    (EmbeddingScorer(), 0.5),   # Primary: semantic correctness
    (LLMJudgeScorer(client, "Factual accuracy and completeness"), 0.2)
])
```

**Constraints & Trade-offs:**

- **Exact match**: Free, instant, but misses equivalent answers ("CEO" vs "Chief Executive Officer")
- **Embeddings**: $0.0001/eval, 50ms latency, handles synonyms but misses factual errors
- **LLM judge**: $0.01/eval, 2s latency, flexible but adds variance and cost

**Best Practice**: Start with cheap scorers (exact, embedding), add LLM judges only for ambiguous cases flagged by low confidence.

### 3. Experiment Tracking Infrastructure

**Technical Explanation:**

Experiment tracking captures every parameter, output, and metric across all runs, enabling reproducibility, comparison, and analysis. Critical for understanding what changes improved performance and by how much.

**Implementation:**

```python
from typing import Any, Dict, Optional
import sqlite3
from contextlib import contextmanager
import json
from datetime import datetime

class ExperimentTracker:
    """Lightweight experiment tracking without external dependencies."""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    config TEXT NOT NULL,
                    metrics TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    example_id TEXT,
                    input TEXT,
                    predicted TEXT,
                    expected TEXT,
                    score REAL,
                    latency_ms INTEGER,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
    
    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def start_experiment(self, name: str, config: Dict[str, Any], 
                        metadata: Optional[Dict] = None) -> int:
        """Begin new experiment, return experiment ID."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO experiments (name, timestamp, config, metadata) VALUES (?, ?, ?, ?)",
                (name, datetime.now().isoformat(), json.dumps(config), 
                 json.dumps(metadata or {}))
            )
            return cursor.lastrowid
    
    def log_prediction(self, experiment_id: int, example_id: str,
                      input: str, predicted: str, expected: str,
                      score: float, latency_ms: int):
        """Record individual prediction result."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO predictions 
                   (experiment_id, example_id, input, predicted, expected, score, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (experiment_id, example_id, input, predicted, expected, score, latency_ms)
            )
    
    def finish_experiment(self, experiment_id: int, metrics: Dict[str, float]):
        """Complete experiment with aggregate metrics."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE experiments SET metrics = ? WHERE id = ?",
                (json.dumps(metrics), experiment_id)
            )
    
    def compare_experiments(self, exp_ids: List[int]) -> Dict:
        """Compare metrics across experiments."""
        with self._get_conn() as conn:
            results = {}
            for exp_id in exp_ids:
                row = conn.execute(
                    "SELECT name, config, metrics FROM experiments WHERE id = ?",
                    (exp_id,)
                ).fetchone()
                
                if row:
                    results[exp_id] = {
                        'name': row['name'],
                        'config': json.loads(row['config']),
                        'metrics': json.loads(row['metrics'])
                    }
            return results
    
    def get_failed_predictions(self, experiment_id: int, 
                              threshold: float = 0.5) -> List[Dict]:
        """Retrieve low-scoring predictions for analysis."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT example_id, input, predicted, expected, score
                   FROM predictions 
                   WHERE experiment_id = ? AND score < ?
                   ORDER BY score ASC""",
                (experiment_id, threshold)
            ).fetchall()
            
            return [dict(row) for row in rows]

# Usage in BML cycle
tracker = ExperimentTracker()

config = {
    "model": "claude-3-5-sonnet-20241022",
    "temperature