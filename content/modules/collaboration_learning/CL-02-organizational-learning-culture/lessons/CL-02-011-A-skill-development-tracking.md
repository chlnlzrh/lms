# Skill Development Tracking with LLMs

## Core Concepts

**Technical Definition:** Skill development tracking uses LLMs to analyze learning artifacts—code submissions, text responses, project outputs—and extract structured assessments of competency progression over time. Unlike traditional rule-based assessment systems that match answers against fixed rubrics, LLM-based tracking interprets nuanced evidence of understanding, maps it to skill taxonomies, and identifies growth patterns through semantic analysis of work products.

**Engineering Analogy: Traditional vs. LLM-Based Tracking**

Traditional approach (rule-based assessment):

```python
# Traditional: Rigid pattern matching
def assess_sql_skill(query: str) -> dict[str, bool]:
    """Basic pattern matching for SQL competency"""
    skills = {
        'uses_join': 'JOIN' in query.upper(),
        'uses_aggregation': any(fn in query.upper() 
                                for fn in ['SUM', 'COUNT', 'AVG']),
        'uses_subquery': query.count('SELECT') > 1,
        'uses_groupby': 'GROUP BY' in query.upper()
    }
    return skills

# Result: Binary checklist, no depth understanding
student_query = "SELECT AVG(price) FROM products WHERE category='electronics'"
print(assess_sql_skill(student_query))
# {'uses_join': False, 'uses_aggregation': True, 
#  'uses_subquery': False, 'uses_groupby': False}
```

LLM-based approach (semantic interpretation):

```python
from anthropic import Anthropic
from typing import TypedDict
import json

class SkillAssessment(TypedDict):
    skill: str
    proficiency_level: str  # novice, developing, proficient, advanced
    evidence: str
    growth_areas: list[str]

def assess_sql_skill_llm(query: str, context: str) -> list[SkillAssessment]:
    """Semantic analysis of SQL competency"""
    client = Anthropic()
    
    prompt = f"""Analyze this SQL query for skill demonstration:

Query: {query}
Context: {context}

Evaluate across these dimensions:
1. Query construction (syntax, structure)
2. Data manipulation understanding (functions, operations)
3. Performance awareness (indexing implications, efficiency)
4. Problem-solving approach (solution elegance)

For each skill area, provide:
- Proficiency level (novice/developing/proficient/advanced)
- Specific evidence from the query
- Growth areas for next level

Return JSON array of assessments."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)

# Result: Nuanced, contextual assessment
context = "Task: Calculate average product price for electronics category"
assessments = assess_sql_skill_llm(student_query, context)
# Returns detailed proficiency mapping with growth trajectory
```

**Key Insight:** Traditional systems track *what* features appear in work; LLM systems understand *how well* concepts are applied in context. This shift from detection to interpretation enables tracking genuine understanding rather than surface-level pattern completion.

**Why This Matters Now:**

1. **Scale-Quality Trade-off Solved:** Human expert assessment provides depth but doesn't scale; automated systems scale but lack depth. LLMs provide expert-level interpretation at automated-system scale—critical as remote learning and async education grow.

2. **Competency-Based Learning Infrastructure:** As industries shift from credential-based to skill-based hiring, organizations need systems that track granular competency development with evidence trails. LLMs make this technically feasible.

3. **Continuous Professional Development:** Engineers learn constantly through work artifacts (code reviews, design docs, incident reports). LLMs can transform existing work streams into skill development data without additional assessment overhead.

## Technical Components

### 1. Artifact Analysis & Feature Extraction

**Technical Explanation:** The foundation of skill tracking is extracting meaningful features from learning artifacts. Unlike structured test data, real learning artifacts—code, essays, designs—require semantic parsing to identify skill signals. LLMs act as feature extractors, transforming unstructured work products into structured skill evidence.

**Practical Implications:** You must design prompts that consistently extract comparable features across diverse artifacts. The challenge is maintaining assessment consistency while handling artifact variability.

**Real Constraints:**
- Context window limits restrict how much artifact history you can analyze simultaneously
- Different artifact types (code vs. text vs. diagrams) require different analysis approaches
- Feature extraction quality depends heavily on prompt engineering

**Concrete Example:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

@dataclass
class LearningArtifact:
    content: str
    artifact_type: Literal['code', 'writing', 'diagram_description']
    timestamp: datetime
    context: str

@dataclass
class SkillEvidence:
    skill_id: str
    skill_name: str
    proficiency_score: float  # 0-1 scale
    evidence_snippets: list[str]
    reasoning: str

def extract_skill_evidence(
    artifact: LearningArtifact,
    skill_taxonomy: dict[str, str]
) -> list[SkillEvidence]:
    """Extract structured skill evidence from learning artifact"""
    
    client = Anthropic()
    
    # Build taxonomy description for prompt
    taxonomy_desc = "\n".join([
        f"- {skill_id}: {description}" 
        for skill_id, description in skill_taxonomy.items()
    ])
    
    prompt = f"""Analyze this learning artifact for skill demonstration.

ARTIFACT TYPE: {artifact.artifact_type}
CONTEXT: {artifact.context}

CONTENT:
{artifact.content}

SKILL TAXONOMY:
{taxonomy_desc}

For each skill demonstrated in this artifact:
1. Identify which taxonomy skill it maps to
2. Rate proficiency (0.0-1.0): 0.0-0.3=novice, 0.3-0.6=developing, 
   0.6-0.8=proficient, 0.8-1.0=advanced
3. Extract specific evidence snippets from content
4. Explain your reasoning

Return JSON array:
[
  {{
    "skill_id": "string",
    "skill_name": "string", 
    "proficiency_score": float,
    "evidence_snippets": ["string"],
    "reasoning": "string"
  }}
]

Only include skills with clear evidence. Be conservative with proficiency scores."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    evidence_list = json.loads(response.content[0].text)
    
    return [
        SkillEvidence(
            skill_id=ev['skill_id'],
            skill_name=ev['skill_name'],
            proficiency_score=ev['proficiency_score'],
            evidence_snippets=ev['evidence_snippets'],
            reasoning=ev['reasoning']
        )
        for ev in evidence_list
    ]

# Example usage
artifact = LearningArtifact(
    content="""
def calculate_moving_average(prices: list[float], window: int) -> list[float]:
    if window > len(prices):
        raise ValueError(f"Window size {window} exceeds data length {len(prices)}")
    
    result = []
    for i in range(len(prices) - window + 1):
        window_avg = sum(prices[i:i+window]) / window
        result.append(window_avg)
    
    return result
""",
    artifact_type='code',
    timestamp=datetime.now(),
    context="Exercise: Implement moving average calculation for time series data"
)

skill_taxonomy = {
    'python_basic': 'Basic Python syntax, control flow, data structures',
    'python_typing': 'Type hints and type safety practices',
    'error_handling': 'Exception handling and input validation',
    'algorithm_design': 'Problem-solving approach and algorithm selection',
    'code_efficiency': 'Performance considerations and optimization'
}

evidence = extract_skill_evidence(artifact, skill_taxonomy)
for ev in evidence:
    print(f"{ev.skill_name}: {ev.proficiency_score:.2f}")
    print(f"  Evidence: {ev.evidence_snippets[0][:80]}...")
    print()
```

### 2. Temporal Progression Modeling

**Technical Explanation:** Skill development is inherently temporal—understanding grows over time through repeated practice. Effective tracking requires comparing artifacts across time to identify growth trajectories, plateaus, and regressions. This involves maintaining skill state history and computing progression metrics.

**Practical Implications:** You need data structures that efficiently store skill assessments over time and algorithms that compute meaningful progression signals from noisy data (skill demonstrations vary based on task difficulty, not just competency).

**Real Constraints:**
- Early assessments have high uncertainty (few data points)
- Task difficulty variation creates noise in proficiency signals
- Storage costs scale with artifact volume and retention period

**Concrete Example:**

```python
from collections import defaultdict
from statistics import mean, stdev
import numpy as np

@dataclass
class SkillSnapshot:
    timestamp: datetime
    proficiency_score: float
    artifact_id: str
    task_difficulty: float  # 0-1 scale

@dataclass
class ProgressionMetrics:
    current_proficiency: float
    growth_rate: float  # proficiency points per day
    consistency: float  # 0-1, based on score variance
    trajectory: Literal['improving', 'stable', 'declining', 'insufficient_data']
    confidence: float  # 0-1

class SkillTracker:
    def __init__(self):
        self.skill_history: dict[str, list[SkillSnapshot]] = defaultdict(list)
    
    def record_assessment(
        self, 
        skill_id: str, 
        snapshot: SkillSnapshot
    ) -> None:
        """Add new skill assessment to history"""
        self.skill_history[skill_id].append(snapshot)
        # Sort by timestamp to maintain chronological order
        self.skill_history[skill_id].sort(key=lambda s: s.timestamp)
    
    def compute_progression(
        self, 
        skill_id: str,
        lookback_days: int = 30
    ) -> ProgressionMetrics:
        """Compute skill progression metrics over recent period"""
        
        snapshots = self.skill_history[skill_id]
        
        if len(snapshots) < 2:
            return ProgressionMetrics(
                current_proficiency=snapshots[0].proficiency_score if snapshots else 0.0,
                growth_rate=0.0,
                consistency=0.0,
                trajectory='insufficient_data',
                confidence=0.0
            )
        
        # Filter to lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent = [s for s in snapshots if s.timestamp >= cutoff_date]
        
        if len(recent) < 2:
            recent = snapshots[-2:]  # Use last 2 if insufficient recent data
        
        # Current proficiency: weighted average of recent assessments
        # Weight by recency and inverse of task difficulty (harder tasks = stronger signal)
        weights = []
        scores = []
        for s in recent[-5:]:  # Last 5 assessments
            days_old = (datetime.now() - s.timestamp).days
            recency_weight = np.exp(-days_old / 10)  # Exponential decay
            difficulty_weight = 0.5 + (s.task_difficulty * 0.5)  # 0.5-1.0 range
            
            weight = recency_weight * difficulty_weight
            weights.append(weight)
            scores.append(s.proficiency_score)
        
        current = np.average(scores, weights=weights)
        
        # Growth rate: linear regression over time
        days_elapsed = (recent[-1].timestamp - recent[0].timestamp).days + 1
        score_change = recent[-1].proficiency_score - recent[0].proficiency_score
        growth_rate = score_change / days_elapsed if days_elapsed > 0 else 0.0
        
        # Consistency: inverse of coefficient of variation
        score_mean = mean(s.proficiency_score for s in recent)
        score_std = stdev(s.proficiency_score for s in recent) if len(recent) > 1 else 0.0
        consistency = 1.0 - min(score_std / score_mean if score_mean > 0 else 1.0, 1.0)
        
        # Trajectory classification
        if growth_rate > 0.01:  # >1% proficiency per day
            trajectory = 'improving'
        elif growth_rate < -0.01:
            trajectory = 'declining'
        else:
            trajectory = 'stable'
        
        # Confidence based on sample size and recency
        confidence = min(len(recent) / 10, 1.0) * 0.7 + \
                    (1.0 - min((datetime.now() - recent[-1].timestamp).days / 30, 1.0)) * 0.3
        
        return ProgressionMetrics(
            current_proficiency=current,
            growth_rate=growth_rate,
            consistency=consistency,
            trajectory=trajectory,
            confidence=confidence
        )

# Example usage
tracker = SkillTracker()

# Simulate skill assessments over time
base_date = datetime.now() - timedelta(days=45)
for day_offset in [0, 7, 14, 21, 28, 35, 42]:
    tracker.record_assessment(
        'python_typing',
        SkillSnapshot(
            timestamp=base_date + timedelta(days=day_offset),
            proficiency_score=0.4 + (day_offset * 0.008) + np.random.normal(0, 0.05),
            artifact_id=f"artifact_{day_offset}",
            task_difficulty=0.5 + (day_offset * 0.01)
        )
    )

metrics = tracker.compute_progression('python_typing')
print(f"Current Proficiency: {metrics.current_proficiency:.3f}")
print(f"Growth Rate: {metrics.growth_rate:.4f} points/day")
print(f"Trajectory: {metrics.trajectory} (confidence: {metrics.confidence:.2f})")
```

### 3. Skill Taxonomy Mapping

**Technical Explanation:** Raw LLM assessments describe skill demonstrations in natural language. To aggregate, compare, and track skills systematically, you must map these descriptions to a structured skill taxonomy. This involves semantic matching between free-form skill observations and canonical skill definitions.

**Practical Implications:** Taxonomy design significantly impacts tracking granularity and actionability. Too coarse (e.g., "programming skills") provides little actionable insight; too fine (e.g., "uses list comprehensions with nested conditionals") creates fragmentation and sparse data.

**Real Constraints:**
- Taxonomies must balance comprehensiveness with maintainability
- Semantic matching has inherent ambiguity (is "error handling" part of "defensive programming"?)
- Taxonomy evolution over time creates versioning challenges

**Concrete Example:**

```python
from typing import Optional

@dataclass
class SkillNode:
    skill_id: str
    name: str
    description: str
    parent_id: Optional[str]
    level: int  # 0=root, 1=category, 2=skill, 3=sub-skill

class SkillTaxonomy:
    def __init__(self):
        self.nodes: dict[str, SkillNode] = {}
        self._build_default_taxonomy()
    
    def _build_default_taxonomy(self) -> None:
        """Build example software engineering skill taxonomy"""
        taxonomy_def = [
            # Level 0: Root
            ('engineering', 'Software Engineering', 'Core software engineering competencies', None, 0),
            
            # Level 1: Categories
            ('programming', 'Programming', 'Code writing and implementation skills', 'engineering', 1),
            ('architecture', 'Architecture & Design', 'System design and architecture', 'engineering', 1),
            
            # Level 2: Skills
            ('python', 'Python Programming', 'Python language proficiency', 'programming', 2),
            ('error_handling',