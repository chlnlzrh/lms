# OKR & KPI Tracking Dashboards with LLMs

## Core Concepts

### Technical Definition

OKR (Objectives and Key Results) and KPI (Key Performance Indicator) tracking dashboards traditionally aggregate structured data from databases and APIs, applying predefined formulas and visualizations. LLM-enhanced dashboards introduce a fundamentally different architecture: they transform unstructured data sources into structured metrics, generate dynamic insights through semantic analysis, and provide natural language interfaces for metric exploration.

The core technical shift is from **static schema-bound aggregation** to **dynamic semantic extraction and synthesis**. Instead of writing explicit ETL pipelines for each data source and metric, you configure LLMs to understand business context and extract relevant signals from varied formats—Slack conversations, documentation updates, customer support tickets, meeting transcripts.

### Engineering Analogy: Traditional vs. LLM-Enhanced Approach

```python
# Traditional Approach: Rigid Schema Dependencies
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TraditionalMetric:
    name: str
    value: float
    timestamp: datetime

class TraditionalDashboard:
    """Requires explicit schema mapping for every data source"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_customer_satisfaction(self) -> float:
        # Must know exact table, column names, aggregation logic
        query = """
            SELECT AVG(rating) 
            FROM customer_surveys 
            WHERE survey_type = 'NPS' 
            AND timestamp > NOW() - INTERVAL '30 days'
        """
        result = self.db.execute(query)
        return result[0][0] if result else 0.0
    
    def get_feature_adoption(self, feature_name: str) -> float:
        # Requires events to be instrumented in exact format
        query = """
            SELECT COUNT(DISTINCT user_id)::float / 
                   (SELECT COUNT(*) FROM active_users) * 100
            FROM feature_events
            WHERE feature = %s AND event_type = 'used'
            AND timestamp > NOW() - INTERVAL '7 days'
        """
        result = self.db.execute(query, (feature_name,))
        return result[0][0] if result else 0.0

# Problem: Adding new metric requires schema changes, ETL updates, deployment
```

```python
# LLM-Enhanced Approach: Semantic Extraction
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import anthropic
import json

@dataclass
class SemanticMetric:
    name: str
    value: Optional[float]
    confidence: float
    extracted_from: List[str]
    reasoning: str
    timestamp: datetime

class LLMEnhancedDashboard:
    """Extracts metrics from unstructured sources through semantic understanding"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.context_window = []
    
    def extract_metric(
        self, 
        metric_definition: str,
        data_sources: List[str],
        context: Optional[str] = None
    ) -> SemanticMetric:
        """
        Extract any metric from unstructured sources without schema changes.
        
        Args:
            metric_definition: Natural language description of what to measure
            data_sources: List of text content (Slack messages, docs, tickets)
            context: Optional business context for interpretation
        """
        prompt = self._build_extraction_prompt(
            metric_definition, 
            data_sources, 
            context
        )
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_metric_response(response.content[0].text)
    
    def _build_extraction_prompt(
        self, 
        metric_def: str, 
        sources: List[str], 
        context: Optional[str]
    ) -> str:
        context_str = f"\n\nBusiness Context:\n{context}" if context else ""
        
        sources_str = "\n\n".join([
            f"Source {i+1}:\n{source}" 
            for i, source in enumerate(sources)
        ])
        
        return f"""Extract the following metric from the provided data sources:

Metric Definition: {metric_def}
{context_str}

Data Sources:
{sources_str}

Analyze the sources and provide:
1. Numeric value for the metric (or null if insufficient data)
2. Confidence score (0-1)
3. Which sources you used
4. Brief reasoning explaining your extraction

Format as JSON:
{{
    "value": <number or null>,
    "confidence": <0-1>,
    "sources_used": [<source indices>],
    "reasoning": "<explanation>"
}}"""
    
    def _parse_metric_response(self, response: str) -> SemanticMetric:
        # Extract JSON from response (handle markdown code blocks)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        data = json.loads(response[json_start:json_end])
        
        return SemanticMetric(
            name="extracted_metric",
            value=data.get("value"),
            confidence=data["confidence"],
            extracted_from=[f"source_{i}" for i in data["sources_used"]],
            reasoning=data["reasoning"],
            timestamp=datetime.now()
        )

# Advantage: Add new metrics without code changes or schema updates
```

### Key Engineering Insights

**1. Metrics Become Queries, Not Schemas**

Traditional dashboards require metrics to be pre-defined in database schemas. LLM-enhanced systems treat metrics as semantic queries that can be defined ad-hoc. This fundamentally changes velocity: instead of weeks to add a metric (schema design → ETL → testing → deployment), you can define new metrics in minutes through natural language.

**2. Unstructured Data Becomes First-Class**

Approximately 80% of business-critical information lives in unstructured formats—meeting notes, customer conversations, engineering discussions. Traditional BI tools ignore this entirely. LLMs make this data queryable, dramatically expanding what you can measure.

**3. Context Drift Requires Continuous Validation**

Unlike SQL queries with deterministic outputs, LLM extraction results vary based on prompt phrasing, model version, and input variation. This introduces a new category of engineering challenge: semantic result validation. You need automated testing that verifies extraction quality, not just code correctness.

### Why This Matters Now

The convergence of three technical factors makes this practical in 2024:

1. **Context windows of 200K+ tokens**: You can fit entire quarters of Slack channels, multiple repositories of documentation, or comprehensive customer feedback datasets in a single API call.

2. **Sub-$1 cost per analysis**: Claude 3.5 Sonnet processes 100K input tokens for ~$0.30. Analyzing a week of team communications costs less than querying a traditional analytics warehouse.

3. **Structured output reliability**: Modern LLMs with JSON mode and few-shot prompting achieve 95%+ accuracy on extraction tasks, making them production-viable.

## Technical Components

### Component 1: Semantic Extraction Pipeline

**Technical Explanation**

The extraction pipeline transforms unstructured text into structured metric values through a three-stage process:

1. **Source aggregation**: Collect relevant text data with temporal boundaries
2. **Semantic analysis**: Apply LLM to extract signals matching metric definitions
3. **Validation & storage**: Verify confidence thresholds and persist results

The critical engineering challenge is managing the context window: you need enough data for accurate extraction but must stay within token limits and control costs.

**Practical Implementation**

```python
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import anthropic
import json

@dataclass
class DataSource:
    """Represents a source of unstructured data"""
    source_id: str
    content: str
    timestamp: datetime
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractionResult:
    metric_name: str
    value: Optional[float]
    confidence: float
    sources_used: List[str]
    reasoning: str
    warnings: List[str]
    timestamp: datetime

class SemanticExtractionPipeline:
    """Production-ready extraction pipeline with cost and quality controls"""
    
    def __init__(
        self, 
        api_key: str,
        max_tokens_per_request: int = 100000,  # Leave room for output
        min_confidence: float = 0.7
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_tokens_per_request = max_tokens_per_request
        self.min_confidence = min_confidence
    
    def extract_metric(
        self,
        metric_name: str,
        metric_description: str,
        sources: List[DataSource],
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Extract a metric with automatic source selection and validation.
        
        Args:
            metric_name: Identifier for the metric
            metric_description: Natural language definition
            sources: Available data sources
            validation_rules: Optional rules like {"min": 0, "max": 100}
        
        Returns:
            ExtractionResult with value, confidence, and diagnostics
        """
        # Stage 1: Select relevant sources within token budget
        selected_sources = self._select_sources(sources)
        
        # Stage 2: Perform extraction
        raw_result = self._extract(
            metric_name,
            metric_description,
            selected_sources
        )
        
        # Stage 3: Validate and warn
        validated_result = self._validate(raw_result, validation_rules)
        
        return validated_result
    
    def _select_sources(
        self, 
        sources: List[DataSource]
    ) -> List[DataSource]:
        """
        Select sources that fit within token budget.
        
        Strategy: Most recent first, up to limit. More sophisticated
        approaches could use relevance scoring or stratified sampling.
        """
        sources_sorted = sorted(
            sources, 
            key=lambda s: s.timestamp, 
            reverse=True
        )
        
        selected = []
        total_tokens = 0
        
        for source in sources_sorted:
            if total_tokens + source.token_count <= self.max_tokens_per_request:
                selected.append(source)
                total_tokens += source.token_count
            else:
                break
        
        return selected
    
    def _extract(
        self,
        metric_name: str,
        metric_description: str,
        sources: List[DataSource]
    ) -> Dict[str, Any]:
        """Perform LLM extraction with structured output"""
        
        sources_text = "\n\n".join([
            f"[Source {s.source_id} | {s.timestamp.isoformat()}]\n{s.content}"
            for s in sources
        ])
        
        prompt = f"""You are analyzing data to extract a specific business metric.

METRIC TO EXTRACT:
Name: {metric_name}
Definition: {metric_description}

DATA SOURCES:
{sources_text}

TASK:
Extract the metric value from the sources. Provide your response as JSON:

{{
    "value": <numeric value or null>,
    "confidence": <0.0 to 1.0>,
    "sources_used": [<list of source IDs that informed your answer>],
    "reasoning": "<brief explanation of how you derived this value>",
    "assumptions": [<list any assumptions you made>]
}}

GUIDELINES:
- Only extract values explicitly supported by the data
- If data is ambiguous or contradictory, lower confidence accordingly
- If insufficient data exists, set value to null
- Be conservative: err toward lower confidence if uncertain
"""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        response_text = response.content[0].text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        return json.loads(response_text[json_start:json_end])
    
    def _validate(
        self,
        raw_result: Dict[str, Any],
        validation_rules: Optional[Dict[str, Any]]
    ) -> ExtractionResult:
        """Apply validation rules and generate warnings"""
        
        warnings = []
        value = raw_result.get("value")
        confidence = raw_result["confidence"]
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            warnings.append(
                f"Low confidence ({confidence:.2f} < {self.min_confidence})"
            )
        
        # Apply custom validation rules
        if validation_rules and value is not None:
            if "min" in validation_rules and value < validation_rules["min"]:
                warnings.append(
                    f"Value {value} below minimum {validation_rules['min']}"
                )
            if "max" in validation_rules and value > validation_rules["max"]:
                warnings.append(
                    f"Value {value} above maximum {validation_rules['max']}"
                )
        
        return ExtractionResult(
            metric_name=raw_result.get("metric_name", "unknown"),
            value=value,
            confidence=confidence,
            sources_used=raw_result["sources_used"],
            reasoning=raw_result["reasoning"],
            warnings=warnings,
            timestamp=datetime.now()
        )

# Usage example
def example_usage():
    pipeline = SemanticExtractionPipeline(
        api_key="your-api-key",
        min_confidence=0.7
    )
    
    # Simulate data sources (in practice, fetch from Slack, Jira, etc.)
    sources = [
        DataSource(
            source_id="slack_eng_1",
            content="The new API endpoint is seeing 15% error rate in prod",
            timestamp=datetime.now() - timedelta(hours=2),
            token_count=20
        ),
        DataSource(
            source_id="slack_eng_2",
            content="Error rate dropped to 3% after the hotfix",
            timestamp=datetime.now() - timedelta(hours=1),
            token_count=18
        )
    ]
    
    result = pipeline.extract_metric(
        metric_name="api_error_rate",
        metric_description="Current error rate percentage for production APIs",
        sources=sources,
        validation_rules={"min": 0, "max": 100}
    )
    
    print(f"Value: {result.value}%")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
```

**Real Constraints & Trade-offs**

- **Token costs scale with data volume**: Processing 100K tokens costs ~$0.30. For real-time dashboards with frequent updates, this can reach $100-500/month per metric.
- **Latency**: Extraction takes 2-10 seconds depending on input size. Not suitable for sub-second dashboards.
- **Determinism**: Same input may yield slightly different results across runs. Requires tolerance for ±5% variance.

### Component 2: Multi-Source Synthesis

**Technical Explanation**

Real metrics often require combining information from multiple heterogeneous sources—quantitative data from databases, qualitative signals from text, and derived insights from both. Multi-source synthesis involves:

1. **Source ranking**: Determine authority/recency of each source
2. **Conflict resolution**: Handle contradictory information
3. **Confidence aggregation**: Combine per-source confidence into overall score

The engineering challenge is building a system that transparently shows which sources contributed to the final value, enabling debugging when results seem incorrect.

**Practical Implementation**

```python
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import anthropic
import json

@dataclass
class SourcedValue:
    """A value with its source attribution"""
    value: float