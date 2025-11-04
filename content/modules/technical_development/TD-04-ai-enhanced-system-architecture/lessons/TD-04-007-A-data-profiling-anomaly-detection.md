# Data Profiling & Anomaly Detection with LLMs

## Core Concepts

### Technical Definition

Data profiling is the systematic examination of datasets to understand their structure, content, quality, and statistical properties. Anomaly detection identifies data points that deviate significantly from expected patterns. In the LLM era, these traditionally rule-based, statistical processes can leverage language models' pattern recognition capabilities to understand semantic meaning, context-dependent abnormalities, and complex multi-dimensional relationships that traditional methods miss.

### Engineering Analogy: Traditional vs. LLM-Enhanced Approaches

**Traditional Statistical Approach:**

```python
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

def traditional_anomaly_detection(
    data: pd.DataFrame,
    column: str,
    threshold: float = 3.0
) -> Tuple[List[int], Dict[str, float]]:
    """Traditional z-score based anomaly detection."""
    values = data[column].dropna()
    mean = values.mean()
    std = values.std()
    z_scores = np.abs(stats.zscore(values))
    
    anomalies = data[z_scores > threshold].index.tolist()
    
    profile = {
        'mean': float(mean),
        'std': float(std),
        'min': float(values.min()),
        'max': float(values.max()),
        'anomaly_count': len(anomalies)
    }
    
    return anomalies, profile

# Example: Detecting price anomalies
data = pd.DataFrame({
    'product': ['Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
    'price': [19.99, 21.50, 20.25, 199.99, 19.75],  # One obvious outlier
    'category': ['tools', 'tools', 'tools', 'tools', 'tools']
})

anomalies, profile = traditional_anomaly_detection(data, 'price')
print(f"Statistical anomalies: {anomalies}")
print(f"Profile: {profile}")
# Output: Identifies index 3 (199.99) as statistical outlier
```

**LLM-Enhanced Semantic Approach:**

```python
from typing import List, Dict, Any
import json

def llm_semantic_profiling(
    data: pd.DataFrame,
    llm_function,  # Your LLM API wrapper
    context: str = ""
) -> Dict[str, Any]:
    """
    LLM-based profiling that understands semantic relationships
    and business context.
    """
    
    # Convert data to structured representation
    data_sample = data.head(10).to_dict('records')
    
    prompt = f"""Analyze this dataset for anomalies and profile its characteristics.

Dataset sample:
{json.dumps(data_sample, indent=2)}

Context: {context}

Provide:
1. Data profile (types, patterns, ranges)
2. Anomalies with reasoning
3. Quality issues
4. Semantic inconsistencies

Format as JSON with keys: profile, anomalies, quality_issues, recommendations"""

    response = llm_function(prompt)
    return json.loads(response)

# Example usage with semantic understanding
data_semantic = pd.DataFrame({
    'product': ['USB Cable 6ft', 'USB Cable 3ft', 'HDMI Cable 6ft', 
                'USB Cable 6ft Premium', 'USB Cabl 6ft'],  # Typo
    'price': [8.99, 5.99, 12.99, 89.99, 8.99],  # Premium variant, not anomaly
    'category': ['cables', 'cables', 'cables', 'cables', 'cables']
})

# LLM can identify:
# - "USB Cabl 6ft" is likely a typo, not a different product
# - $89.99 for "Premium" variant might be justified, not anomalous
# - Naming patterns and inconsistencies
# - Price-to-feature relationships
```

### Key Insights That Change Engineering Thinking

**1. Context-Aware Anomalies Replace Fixed Thresholds**

Traditional methods flag statistical outliers. LLMs understand that "$10,000 laptop" is normal while "$10,000 USB cable" is suspicious, even if both fall within historical price ranges.

**2. Schema-Free Profiling Handles Unstructured Data**

Traditional profiling requires structured schemas. LLMs can profile free-text fields, mixed formats, and nested JSON structures without predefined rules.

**3. Compound Anomalies Across Dimensions**

A single field might be normal, but combinations reveal issues. LLMs can detect "entry-level position requiring 15 years experience" or "vegan beef burger" without explicit rules for each combination.

### Why This Matters Now

**Data Quality Crisis at Scale:** Organizations process billions of records from APIs, user inputs, and integrations. Traditional rule-based validation breaks down when:
- Data sources are heterogeneous and constantly changing
- Business rules are implicit and context-dependent
- Anomalies are subtle semantic contradictions, not statistical outliers

**Cost of Bad Data:** Invalid data in production systems causes failed transactions, incorrect analytics, and poor ML model performance. LLM-enhanced profiling catches semantic issues before they propagate.

**Reduction in Manual Rule Maintenance:** Instead of maintaining thousands of validation rules across evolving schemas, LLM-based systems adapt to new patterns and contexts.

## Technical Components

### 1. Semantic Pattern Recognition

**Technical Explanation:**

LLMs identify patterns in data by understanding semantic relationships rather than just statistical distributions. They recognize that "Senior Engineer - 1 year experience" is inconsistent even though both fields individually are valid.

**Implementation:**

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import anthropic
import os

@dataclass
class SemanticAnomaly:
    record_id: int
    field: str
    value: Any
    reason: str
    severity: str  # 'high', 'medium', 'low'
    suggestion: Optional[str] = None

def detect_semantic_anomalies(
    records: List[Dict[str, Any]],
    domain_context: str,
    api_key: str
) -> List[SemanticAnomaly]:
    """
    Detect anomalies using semantic understanding.
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # Batch records for efficiency (10 at a time)
    batch_size = 10
    all_anomalies = []
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        
        prompt = f"""Analyze these records for semantic anomalies and inconsistencies.

Domain: {domain_context}

Records:
{json.dumps(batch, indent=2)}

Identify:
1. Internal inconsistencies (e.g., contradicting fields)
2. Semantic impossibilities (e.g., "waterproof water")
3. Likely data entry errors
4. Business rule violations

Return JSON array with format:
[{{
  "record_index": 0,
  "field": "field_name",
  "value": "actual_value",
  "reason": "why this is anomalous",
  "severity": "high|medium|low",
  "suggestion": "proposed correction or null"
}}]

Only flag genuine issues, not preferences. Return [] if no anomalies."""

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        try:
            anomalies_data = json.loads(response_text)
            for anom in anomalies_data:
                all_anomalies.append(SemanticAnomaly(
                    record_id=i + anom['record_index'],
                    field=anom['field'],
                    value=anom['value'],
                    reason=anom['reason'],
                    severity=anom['severity'],
                    suggestion=anom.get('suggestion')
                ))
        except json.JSONDecodeError:
            print(f"Failed to parse response for batch {i}")
            continue
    
    return all_anomalies

# Example usage
job_postings = [
    {
        "title": "Senior Software Engineer",
        "experience_required": "1 year",
        "salary_range": "$180,000 - $220,000",
        "level": "senior"
    },
    {
        "title": "Junior Developer",
        "experience_required": "0-2 years",
        "salary_range": "$70,000 - $90,000",
        "level": "entry"
    },
    {
        "title": "Lead Architect",
        "experience_required": "2 years",  # Suspicious
        "salary_range": "$45,000 - $55,000",  # Too low
        "level": "lead"
    }
]

# anomalies = detect_semantic_anomalies(
#     job_postings,
#     "Software engineering job postings",
#     os.environ['ANTHROPIC_API_KEY']
# )
```

**Practical Implications:**

- Catches 30-40% more issues than statistical methods alone
- Requires API calls, adding latency (200-500ms per batch)
- Works on first pass without training data or historical patterns

**Trade-offs:**

- **Cost:** $0.003 per 1K input tokens, $0.015 per 1K output tokens (Claude 3.5 Sonnet)
- **Latency:** Batch processing required for datasets with >1000 records
- **Determinism:** Temperature=0 provides consistency, but edge cases may vary slightly

### 2. Automated Schema Discovery and Profiling

**Technical Explanation:**

Instead of manually defining expected schemas, LLMs infer structure, types, patterns, and constraints from data samples. This is particularly powerful for JSON, nested structures, and free-text fields.

**Implementation:**

```python
from typing import Dict, Any, List
from collections import defaultdict

def profile_dataset_with_llm(
    data: pd.DataFrame,
    sample_size: int = 100,
    api_key: str
) -> Dict[str, Any]:
    """
    Generate comprehensive data profile using LLM analysis.
    """
    client = anthropic.Anthropic(api_key=api_key)
    
    # Sample data for analysis
    sample = data.sample(min(sample_size, len(data)))
    
    # Get basic statistics
    basic_stats = {
        'row_count': len(data),
        'column_count': len(data.columns),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'dtypes': data.dtypes.astype(str).to_dict()
    }
    
    # Prepare sample for LLM
    sample_records = sample.to_dict('records')[:20]
    
    prompt = f"""Profile this dataset comprehensively.

Sample records ({len(sample_records)} of {len(data)} total):
{json.dumps(sample_records, indent=2, default=str)}

Basic stats:
{json.dumps(basic_stats, indent=2)}

Provide JSON profile with:
{{
  "inferred_schema": {{
    "column_name": {{
      "type": "inferred type",
      "constraints": ["constraint1", "constraint2"],
      "pattern": "regex or description",
      "nullable": true/false
    }}
  }},
  "data_quality": {{
    "completeness": 0.0-1.0,
    "consistency": 0.0-1.0,
    "issues": ["issue descriptions"]
  }},
  "business_insights": ["insight1", "insight2"],
  "relationships": ["detected relationships between fields"],
  "recommendations": ["actionable suggestions"]
}}"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=3000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    profile = json.loads(message.content[0].text)
    profile['basic_statistics'] = basic_stats
    
    return profile

# Example with complex nested data
complex_data = pd.DataFrame([
    {
        'user_id': 'USR001',
        'email': 'john@example.com',
        'metadata': '{"signup_date": "2024-01-15", "plan": "premium"}',
        'tags': 'active,verified,subscriber'
    },
    {
        'user_id': 'USR002',
        'email': 'invalid-email',  # Issue
        'metadata': '{"signup_date": "2024-02-20", "plan": "free"}',
        'tags': 'active'
    },
    {
        'user_id': 'USR003',
        'email': 'jane@example.com',
        'metadata': 'invalid json',  # Issue
        'tags': None  # Missing
    }
])

# profile = profile_dataset_with_llm(complex_data, api_key=os.environ['ANTHROPIC_API_KEY'])
# LLM identifies:
# - metadata should be valid JSON
# - email format issues
# - tags are comma-separated lists
# - missing data patterns
```

**Practical Implications:**

- Discovers implicit business rules without documentation
- Identifies data type mismatches (e.g., numbers stored as strings)
- Detects nested structures and suggests normalization

**Real Constraints:**

- Requires representative sample (at least 50-100 records)
- May miss rare edge cases present in full dataset
- Best combined with traditional statistical profiling for numeric validation

### 3. Contextual Anomaly Scoring

**Technical Explanation:**

Rather than binary anomaly detection, assign context-aware severity scores. A price of $0.01 might be normal for a digital download but critical for a laptop.

**Implementation:**

```python
from enum import Enum
from typing import List, Tuple

class AnomalyType(Enum):
    STATISTICAL = "statistical"
    SEMANTIC = "semantic"
    BUSINESS_RULE = "business_rule"
    CONSISTENCY = "consistency"

@dataclass
class ScoredAnomaly:
    record_id: int
    anomaly_type: AnomalyType
    field: str
    value: Any
    score: float  # 0.0 (normal) to 1.0 (critical)
    explanation: str
    context: str

def contextual_anomaly_scoring(
    data: pd.DataFrame,
    context: str,
    api_key: str,
    statistical_threshold: float = 3.0
) -> List[ScoredAnomaly]:
    """
    Hybrid approach: statistical + LLM contextual scoring.
    """
    client = anthropic.Anthropic(api_key=api_key)
    anomalies = []
    
    # Step 1: Statistical anomaly detection (fast, cheap)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    statistical_outliers = []
    
    for col in numeric_columns:
        z_scores = np.abs(stats.zscore(data[col].dropna()))
        outlier_indices = np.where(z_scores > statistical_threshold)[0]
        for idx in outlier_indices:
            statistical_outliers.append({
                'record_id': int(idx),
                'field': col,
                'value': float(data.iloc[idx][col]),
                'z_score': float(z_scores[idx])
            })
    
    # Step 2: LLM contextual evaluation (slower, but precise)
    if statistical_outliers:
        # Get full records for outliers
        outlier_records = []
        for outlier in statistical_outliers:
            record = data.iloc[outlier['record_id']].to_dict()
            record['_anomaly_field'] = outlier['field']
            record['_z_score'] = outlier['z_score']
            outlier_records.append(record)
        
        prompt = f"""Evaluate these statistical outliers in context.

Domain: {context}

Outliers detected:
{json.dumps(outlier_records[:10], indent=2, default=str)}

For each outlier, determine if it's:
1.