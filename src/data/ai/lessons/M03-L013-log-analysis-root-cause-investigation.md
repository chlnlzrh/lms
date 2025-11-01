# Log Analysis & Root Cause Investigation with LLMs

## Core Concepts

### Technical Definition

Log analysis using LLMs transforms unstructured or semi-structured log data into actionable insights by leveraging natural language understanding to identify patterns, correlations, and anomalies that traditional regex-based or rule-based systems miss. Unlike traditional approaches that require predefined patterns and rigid parsing rules, LLM-based analysis can understand context, infer relationships across disparate log entries, and adapt to new log formats without reconfiguration.

### Traditional vs. Modern Approach

**Traditional approach:**

```python
import re
from collections import Counter
from typing import List, Dict

def traditional_log_analysis(log_lines: List[str]) -> Dict[str, int]:
    """Traditional regex-based error detection"""
    error_patterns = {
        'connection_timeout': r'Connection.*timeout',
        'null_pointer': r'NullPointerException',
        'out_of_memory': r'OutOfMemoryError',
        'database_error': r'SQLException.*connection'
    }
    
    results = Counter()
    
    for line in log_lines:
        for error_type, pattern in error_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                results[error_type] += 1
    
    return dict(results)

# Limitations:
# - Misses unknown error patterns
# - No context understanding across log entries
# - Cannot infer causality
# - Requires maintenance for new error types
# - No semantic understanding (e.g., "connection refused" vs "connection timeout")
```

**LLM-enhanced approach:**

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class LogAnalysis:
    root_cause: str
    related_entries: List[int]
    severity: str
    recommended_action: str
    confidence: float

def llm_log_analysis(
    log_lines: List[str],
    llm_client,
    context_window: int = 50
) -> LogAnalysis:
    """LLM-based contextual log analysis"""
    
    # Build temporal and semantic context
    system_prompt = """You are a log analysis expert. Analyze the provided logs to:
1. Identify the root cause of failures
2. Trace the causal chain of events
3. Assess severity based on impact and frequency
4. Recommend specific remediation steps

Return JSON with: root_cause, related_entry_indices, severity (critical/high/medium/low), 
recommended_action, confidence (0-1)."""

    user_prompt = f"""Analyze these logs and identify the root cause:

{chr(10).join(f"[{i}] {line}" for i, line in enumerate(log_lines[-context_window:]))}

Focus on causal relationships, not just pattern matching."""

    response = llm_client.complete(
        system=system_prompt,
        user=user_prompt,
        response_format="json"
    )
    
    result = json.loads(response)
    return LogAnalysis(**result)

# Advantages:
# - Discovers novel failure patterns
# - Understands temporal causality
# - Semantic understanding of related errors
# - Adapts to new log formats automatically
# - Provides actionable recommendations
```

### Key Engineering Insights

**1. Context window is your investigative radius**: Traditional tools analyze line-by-line. LLMs analyze within a context window, enabling them to correlate events separated by dozens or hundreds of log entries. This fundamentally changes root cause analysis from pattern matching to causal inference.

**2. Semantic compression reduces noise**: Logs contain massive redundancy. LLMs can semantically compress thousands of similar entries into meaningful patterns, focusing on the signal: "423 connection timeout errors to database cluster between 14:32 and 14:38, correlating with deployment event at 14:31."

**3. Implicit pattern learning vs. explicit rule writing**: Each new error type in traditional systems requires new regex patterns. LLMs generalize from examples, identifying new error classes without explicit programming.

### Why This Matters Now

Modern distributed systems generate log volumes (terabytes/day) that exceed human analysis capacity. Traditional SIEM tools produce alert fatigue—hundreds of false positives daily. LLMs bridge the gap between raw logs and actionable intelligence by:

- **Reducing MTTR (Mean Time To Resolution)**: From hours to minutes by automatically tracing causal chains
- **Handling microservices complexity**: Correlating logs across 100+ services without predefined service topology
- **Adapting to change**: New services, libraries, and error types are automatically understood without configuration updates

## Technical Components

### 1. Contextual Aggregation

**Technical Explanation**: Contextual aggregation groups related log entries based on semantic similarity and temporal proximity, rather than exact string matching. This creates "incident clusters" that represent a single underlying issue manifesting across multiple services and time periods.

**Implementation:**

```python
from typing import List, Tuple
from dataclasses import dataclass
import hashlib

@dataclass
class LogEntry:
    timestamp: str
    service: str
    level: str
    message: str
    raw: str

@dataclass
class IncidentCluster:
    representative_entry: LogEntry
    related_entries: List[LogEntry]
    semantic_summary: str
    affected_services: List[str]

def create_incident_clusters(
    logs: List[LogEntry],
    llm_client,
    similarity_threshold: float = 0.7,
    time_window_seconds: int = 300
) -> List[IncidentCluster]:
    """Group related log entries into incident clusters"""
    
    clusters: List[IncidentCluster] = []
    processed_indices = set()
    
    for i, log in enumerate(logs):
        if i in processed_indices or log.level not in ['ERROR', 'CRITICAL']:
            continue
        
        # Get temporal window
        window_logs = [
            l for j, l in enumerate(logs[max(0, i-50):i+50]) 
            if j not in processed_indices
        ]
        
        # Ask LLM to identify related entries
        prompt = f"""Given this error:
[Primary] {log.timestamp} {log.service}: {log.message}

Which of these entries are causally related or manifestations of the same issue?
{chr(10).join(f"[{j}] {l.timestamp} {l.service}: {l.message}" for j, l in enumerate(window_logs))}

Return JSON: {{"related_indices": [list], "summary": "brief explanation"}}"""

        response = llm_client.complete(system="You are a log correlation expert.", user=prompt)
        result = json.loads(response)
        
        related = [window_logs[j] for j in result['related_indices']]
        clusters.append(IncidentCluster(
            representative_entry=log,
            related_entries=related,
            semantic_summary=result['summary'],
            affected_services=list(set(l.service for l in related))
        ))
        
        processed_indices.update(result['related_indices'])
    
    return clusters
```

**Practical Implications**: A database connection timeout might manifest as: API gateway 504 errors, cache miss spikes, background job failures, and user session drops. Traditional tools see five separate issues; contextual aggregation identifies one root cause.

**Trade-offs**: 
- **Cost**: Each cluster analysis requires an LLM call (~1000-3000 tokens)
- **Latency**: Real-time analysis requires streaming and batching strategies
- **Accuracy**: Similarity threshold tuning prevents over-grouping unrelated issues

### 2. Causal Chain Reconstruction

**Technical Explanation**: Causal chain reconstruction traces the sequence of events leading to a failure by understanding temporal ordering, dependency relationships, and typical failure propagation patterns in distributed systems.

**Implementation:**

```python
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

class CausalRelation(Enum):
    DIRECT_CAUSE = "direct_cause"
    CONTRIBUTING_FACTOR = "contributing_factor"
    CONSEQUENCE = "consequence"
    UNRELATED = "unrelated"

@dataclass
class CausalLink:
    from_entry: LogEntry
    to_entry: LogEntry
    relation: CausalRelation
    confidence: float
    explanation: str

def reconstruct_causal_chain(
    incident_logs: List[LogEntry],
    llm_client
) -> List[CausalLink]:
    """Build causal graph from incident logs"""
    
    # Sort by timestamp
    sorted_logs = sorted(incident_logs, key=lambda x: x.timestamp)
    
    system_prompt = """You are analyzing distributed system failures. For each pair of log entries, 
determine if the first caused, contributed to, resulted from, or is unrelated to the second.

Consider:
- Temporal ordering (causes precede effects)
- Service dependencies (downstream depends on upstream)
- Resource constraints (memory/CPU exhaustion causes cascading failures)
- Typical failure patterns (connection pools exhaust -> timeouts -> circuit breakers open)

Return JSON: {"relation": "direct_cause|contributing_factor|consequence|unrelated", 
              "confidence": 0.0-1.0, "explanation": "brief technical reason"}"""

    causal_links: List[CausalLink] = []
    
    # Analyze consecutive and nearby pairs
    for i in range(len(sorted_logs)):
        for j in range(i + 1, min(i + 10, len(sorted_logs))):
            entry_a, entry_b = sorted_logs[i], sorted_logs[j]
            
            prompt = f"""Entry A (earlier):
{entry_a.timestamp} [{entry_a.service}] {entry_a.level}: {entry_a.message}

Entry B (later):
{entry_b.timestamp} [{entry_b.service}] {entry_b.level}: {entry_b.message}

What is the causal relationship from A to B?"""

            response = llm_client.complete(system=system_prompt, user=prompt)
            result = json.loads(response)
            
            if result['relation'] != 'unrelated' and result['confidence'] > 0.6:
                causal_links.append(CausalLink(
                    from_entry=entry_a,
                    to_entry=entry_b,
                    relation=CausalRelation(result['relation']),
                    confidence=result['confidence'],
                    explanation=result['explanation']
                ))
    
    return causal_links

def find_root_cause(causal_links: List[CausalLink]) -> Optional[LogEntry]:
    """Identify root cause from causal graph"""
    # Root cause has outgoing but few/no incoming causal links
    entries = {}
    
    for link in causal_links:
        entries.setdefault(link.from_entry, {'outgoing': 0, 'incoming': 0})
        entries.setdefault(link.to_entry, {'outgoing': 0, 'incoming': 0})
        entries[link.from_entry]['outgoing'] += 1
        entries[link.to_entry]['incoming'] += 1
    
    root_candidates = [
        (entry, stats) for entry, stats in entries.items()
        if stats['outgoing'] > 0 and stats['incoming'] == 0
    ]
    
    return root_candidates[0][0] if root_candidates else None
```

**Practical Implications**: When a production incident occurs, this automatically generates a timeline: "Deployment triggered at 14:31 → Database connection pool misconfigured → Connection acquisition timeout → API latency spike → Circuit breaker opened → User-facing errors." Engineers immediately know where to focus.

**Constraints**: 
- **Context limits**: Very long incident chains may exceed context windows (requires hierarchical summarization)
- **Ambiguity**: Concurrent failures may have multiple valid causal interpretations
- **Domain knowledge**: Accuracy improves with system-specific context in prompts

### 3. Semantic Deduplication

**Technical Explanation**: Semantic deduplication identifies functionally identical errors with different surface representations—parameter variations, timestamps, or request IDs—reducing thousands of error instances to their essential unique failure modes.

**Implementation:**

```python
from typing import List, Dict, Set
from dataclasses import dataclass
import hashlib

@dataclass
class ErrorSignature:
    canonical_message: str
    occurrence_count: int
    first_seen: str
    last_seen: str
    example_entries: List[LogEntry]
    affected_services: Set[str]

def semantic_deduplicate(
    error_logs: List[LogEntry],
    llm_client,
    batch_size: int = 20
) -> List[ErrorSignature]:
    """Deduplicate errors by semantic meaning, not string matching"""
    
    signatures: Dict[str, ErrorSignature] = {}
    
    # Process in batches to reduce API calls
    for batch_start in range(0, len(error_logs), batch_size):
        batch = error_logs[batch_start:batch_start + batch_size]
        
        system_prompt = """You are normalizing error messages. Extract the essential error type, 
removing variable data (IDs, timestamps, specific values). 

Examples:
- "Connection timeout to database-prod-3.internal after 30s" → "Database connection timeout"
- "NullPointerException at OrderService.java:142 in processOrder()" → "NullPointerException in OrderService.processOrder"
- "Failed to fetch user 12847: HTTP 404" → "User fetch failed: HTTP 404"

Return JSON array: [{"original_index": int, "canonical": "normalized message"}]"""

        user_prompt = f"""Normalize these errors:
{chr(10).join(f"[{i}] {log.message}" for i, log in enumerate(batch))}"""

        response = llm_client.complete(system=system_prompt, user=user_prompt)
        results = json.loads(response)
        
        for result in results:
            idx = result['original_index']
            canonical = result['canonical']
            log_entry = batch[idx]
            
            # Create signature hash for grouping
            sig_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
            
            if sig_hash not in signatures:
                signatures[sig_hash] = ErrorSignature(
                    canonical_message=canonical,
                    occurrence_count=0,
                    first_seen=log_entry.timestamp,
                    last_seen=log_entry.timestamp,
                    example_entries=[],
                    affected_services=set()
                )
            
            sig = signatures[sig_hash]
            sig.occurrence_count += 1
            sig.last_seen = max(sig.last_seen, log_entry.timestamp)
            if len(sig.example_entries) < 3:
                sig.example_entries.append(log_entry)
            sig.affected_services.add(log_entry.service)
    
    return sorted(signatures.values(), key=lambda s: s.occurrence_count, reverse=True)
```

**Practical Implications**: A microservices architecture with 50 services generating "connection refused" errors to various backends creates thousands of unique log lines. Semantic deduplication reveals: "3 core error types affecting 12 services, all stemming from network partition in availability zone us-east-1a."

**Trade-offs**:
- **Precision vs. recall**: Too aggressive deduplication merges distinct issues; too conservative creates noise
- **Batch size**: Larger batches reduce API calls but may miss cross-batch duplicates
- **Caching**: Canonical forms should be cached to avoid re-normalizing recurring errors

### 4. Structured Information Extraction

**Technical Explanation**: Logs contain semi-structured data embedded in free text. Structured extraction converts this into queryable fields: error codes, latencies, resource IDs, user actions, and service topology—enabling quantitative analysis and correlation with metrics.

**Implementation:**

```python
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

@dataclass
class StructuredLog:
    timestamp: str
    service: str
    level: str
    error_code: Optional[str]
    latency_ms: Optional[float]
    resource_ids: Dict[str, str]
    user_action: Optional[str]
    downstream_services: List[str]
    raw_message: str

def extract_structured_fields(
    log_entry: LogEntry,
    llm_client,
    schema: Dict[str, Any]
) -> StructuredLog:
    """Extract structured fields from unstructured log text"""
    
    system_prompt = f"""Extract structured information from logs according to this schema