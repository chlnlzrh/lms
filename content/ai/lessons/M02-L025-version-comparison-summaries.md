# Version Comparison Summaries: Automated Document Change Analysis with LLMs

## Core Concepts

Version comparison summaries leverage LLMs to automatically analyze differences between document versions and generate human-readable explanations of what changed, why it matters, and what action might be required. Unlike traditional diff tools that show line-by-line changes, LLM-based comparison provides semantic understanding of modifications.

### Traditional vs. Modern Approach

**Traditional diff approach:**

```python
import difflib
from typing import List

def traditional_diff(version1: str, version2: str) -> List[str]:
    """Line-by-line comparison showing raw changes"""
    diff = difflib.unified_diff(
        version1.splitlines(keepends=True),
        version2.splitlines(keepends=True),
        lineterm=''
    )
    return list(diff)

# Example usage
v1 = """API Rate Limits:
- Free tier: 100 requests/hour
- Premium tier: 1000 requests/hour
Contact: support@example.com"""

v2 = """API Rate Limits:
- Free tier: 60 requests/hour
- Premium tier: 1000 requests/hour
- Enterprise tier: 10000 requests/hour
Contact: api-support@example.com"""

changes = traditional_diff(v1, v2)
for line in changes:
    print(line)
```

Output:
```
--- 
+++ 
@@ -1,4 +1,5 @@
 API Rate Limits:
-- Free tier: 100 requests/hour
+- Free tier: 60 requests/hour
 - Premium tier: 1000 requests/hour
-Contact: support@example.com
+- Enterprise tier: 10000 requests/hour
+Contact: api-support@example.com
```

**LLM-based semantic comparison:**

```python
from typing import Dict, Any
import anthropic
import os

def semantic_version_comparison(
    version1: str,
    version2: str,
    context: str = ""
) -> Dict[str, Any]:
    """Generate semantic understanding of version differences"""
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    prompt = f"""Compare these two document versions and provide a structured summary.

Context: {context if context else "General document comparison"}

VERSION 1:
{version1}

VERSION 2:
{version2}

Provide your analysis in this format:
1. SUMMARY: One sentence describing the overall nature of changes
2. KEY CHANGES: List each significant change with its impact level (HIGH/MEDIUM/LOW)
3. BREAKING CHANGES: Any changes that require immediate action
4. ADDITIONS: New content or features
5. REMOVALS: Deleted content
6. MODIFICATIONS: Changed content with before/after context"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "summary": message.content[0].text,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens
    }

# Same documents
result = semantic_version_comparison(v1, v2, "API documentation update")
print(result["summary"])
```

Output:
```
SUMMARY: Rate limit reduction for free tier and contact email update, with new enterprise tier addition.

KEY CHANGES:
- Free tier rate limit reduced from 100 to 60 requests/hour (HIGH - impacts existing users)
- New Enterprise tier added at 10,000 requests/hour (MEDIUM - new offering)
- Support contact changed from support@ to api-support@ (LOW - administrative change)

BREAKING CHANGES:
- Free tier users will experience 40% reduction in available requests - requires immediate communication

ADDITIONS:
- Enterprise tier: 10,000 requests/hour option now available

REMOVALS:
- None (all previous tiers maintained)

MODIFICATIONS:
- Free tier: 100 → 60 requests/hour (40% reduction)
- Contact: support@example.com → api-support@example.com (domain unchanged, prefix updated)
```

### Key Engineering Insights

**Semantic understanding over syntactic differences:** Traditional diffs excel at showing *what* changed character-by-character but fail at explaining *why it matters*. An LLM understands that reducing rate limits is a breaking change requiring user notification, while changing an email prefix is administrative.

**Context-aware impact assessment:** The same textual change has different implications in different contexts. Changing "100" to "60" in a rate limit document is critical; in a historical reference document, it might be a correction with no operational impact.

**Multi-document synthesis:** LLMs can compare not just two versions but synthesize changes across multiple document types—API specs, changelogs, migration guides—to detect inconsistencies or missing documentation.

### Why This Matters Now

Modern software systems generate massive documentation updates: API specs change weekly, configuration files evolve with each deployment, contracts get amended. Manual review is bottlenecked and error-prone. Automated semantic comparison enables:

1. **Release management:** Automatically flag breaking changes in API specifications before deployment
2. **Compliance tracking:** Monitor contract or policy documents for material changes requiring legal review
3. **Configuration drift detection:** Identify when production configs diverge from documented standards with semantic context
4. **Knowledge base maintenance:** Keep internal documentation synchronized as code evolves

The engineering value isn't replacing human review—it's triaging what needs human attention versus what's routine.

## Technical Components

### 1. Structured Diff Input Formatting

The way you present version differences to an LLM dramatically affects output quality. Three approaches with different trade-offs:

**Side-by-side format:**
```python
def format_side_by_side(v1: str, v2: str, labels: tuple = ("V1", "V2")) -> str:
    """Present versions in parallel for direct comparison"""
    return f"""
{labels[0]}:
---
{v1}
---

{labels[1]}:
---
{v2}
---
"""
```

**Trade-offs:** Simple implementation, high token usage (full duplication), best for short documents (<500 words) where context matters.

**Inline diff format:**
```python
def format_inline_diff(v1: str, v2: str) -> str:
    """Show only changed sections with context markers"""
    import difflib
    
    v1_lines = v1.splitlines()
    v2_lines = v2.splitlines()
    
    differ = difflib.Differ()
    diff = list(differ.compare(v1_lines, v2_lines))
    
    # Keep only changed lines with 2 lines of context
    result = []
    for i, line in enumerate(diff):
        if line.startswith('- ') or line.startswith('+ '):
            # Include context
            start = max(0, i-2)
            end = min(len(diff), i+3)
            result.extend(diff[start:end])
            result.append("...")
    
    return "\n".join(result)
```

**Trade-offs:** Moderate token usage, loses document structure, best for large documents with localized changes.

**Structured JSON format (recommended):**
```python
from typing import List, Dict
import json

def format_structured_diff(v1: str, v2: str) -> str:
    """Present changes as structured data for precise analysis"""
    import difflib
    
    v1_lines = v1.splitlines()
    v2_lines = v2.splitlines()
    
    matcher = difflib.SequenceMatcher(None, v1_lines, v2_lines)
    changes: List[Dict[str, Any]] = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            changes.append({
                "type": "modification",
                "old_content": v1_lines[i1:i2],
                "new_content": v2_lines[j1:j2],
                "location": f"lines {i1+1}-{i2}"
            })
        elif tag == 'delete':
            changes.append({
                "type": "deletion",
                "content": v1_lines[i1:i2],
                "location": f"lines {i1+1}-{i2}"
            })
        elif tag == 'insert':
            changes.append({
                "type": "addition",
                "content": v2_lines[j1:j2],
                "location": f"after line {i1}"
            })
    
    return json.dumps({
        "changes": changes,
        "total_changes": len(changes)
    }, indent=2)
```

**Trade-offs:** Moderate token usage, preserves change type information, enables granular analysis. Best for most production use cases.

**Practical implication:** For documents over 2000 words, structured format reduces token usage by 40-60% compared to side-by-side while maintaining semantic fidelity.

### 2. Change Classification and Impact Scoring

Not all changes are equal. Classification helps prioritize review and automate workflows:

```python
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

class ChangeImpact(Enum):
    BREAKING = "breaking"  # Requires immediate action
    MAJOR = "major"        # Significant but manageable
    MINOR = "minor"        # Low impact
    COSMETIC = "cosmetic"  # No functional impact

@dataclass
class ClassifiedChange:
    change_type: str
    content_before: Optional[str]
    content_after: Optional[str]
    impact: ChangeImpact
    reasoning: str
    action_required: Optional[str]

def classify_changes(
    v1: str,
    v2: str,
    domain_context: str = ""
) -> List[ClassifiedChange]:
    """Classify changes with impact assessment"""
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    prompt = f"""Analyze changes between these document versions.
Domain context: {domain_context}

VERSION 1:
{v1}

VERSION 2:
{v2}

For each distinct change, provide classification in this JSON format:
{{
  "changes": [
    {{
      "change_type": "modification|addition|deletion",
      "content_before": "original text or null",
      "content_after": "new text or null",
      "impact": "breaking|major|minor|cosmetic",
      "reasoning": "why this impact level",
      "action_required": "what users must do, or null"
    }}
  ]
}}

Impact level definitions:
- BREAKING: Existing functionality stops working without changes
- MAJOR: Significant feature change, plan migration but not urgent
- MINOR: Small improvement or clarification, awareness sufficient
- COSMETIC: Formatting, typos, no functional impact"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse JSON response
    import json
    response_text = message.content[0].text
    
    # Extract JSON from markdown code blocks if present
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    data = json.loads(response_text.strip())
    
    return [
        ClassifiedChange(
            change_type=c["change_type"],
            content_before=c.get("content_before"),
            content_after=c.get("content_after"),
            impact=ChangeImpact(c["impact"]),
            reasoning=c["reasoning"],
            action_required=c.get("action_required")
        )
        for c in data["changes"]
    ]

# Example usage with API documentation
api_v1 = """
Authentication: Send API key in X-API-Key header
Rate limit: 100 requests/minute
Endpoints:
- GET /users - Returns user list
- POST /users - Creates user (requires admin role)
"""

api_v2 = """
Authentication: Send Bearer token in Authorization header
Rate limit: 100 requests/minute
Endpoints:
- GET /users - Returns paginated user list (max 50 per page)
- POST /users - Creates user (requires admin or manager role)
- DELETE /users/:id - Deletes user (requires admin role)
"""

changes = classify_changes(api_v1, api_v2, "REST API documentation")

for change in changes:
    print(f"\n{change.impact.value.upper()}: {change.change_type}")
    print(f"Reasoning: {change.reasoning}")
    if change.action_required:
        print(f"Action: {change.action_required}")
```

**Practical implication:** Automated classification enables routing—breaking changes trigger incident response workflows, cosmetic changes auto-approve. In production systems managing 100+ documents, this reduces manual review workload by 70-80%.

### 3. Multi-Version Timeline Analysis

Comparing two versions is useful; analyzing trends across many versions reveals patterns:

```python
from datetime import datetime
from typing import List, Dict, Tuple
import anthropic
import os

@dataclass
class VersionSnapshot:
    version_id: str
    timestamp: datetime
    content: str
    author: Optional[str] = None

def analyze_version_timeline(
    versions: List[VersionSnapshot],
    focus_areas: List[str] = None
) -> Dict[str, Any]:
    """Analyze evolution patterns across multiple versions"""
    
    if len(versions) < 2:
        raise ValueError("Need at least 2 versions for timeline analysis")
    
    # Sort by timestamp
    versions = sorted(versions, key=lambda v: v.timestamp)
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Build timeline representation
    timeline = "\n\n".join([
        f"VERSION {v.version_id} ({v.timestamp.isoformat()}):\n{v.content}"
        for v in versions
    ])
    
    focus_section = ""
    if focus_areas:
        focus_section = f"\nPay special attention to changes in: {', '.join(focus_areas)}"
    
    prompt = f"""Analyze the evolution of this document across {len(versions)} versions.{focus_section}

{timeline}

Provide analysis in this structure:
1. EVOLUTION SUMMARY: Overall trajectory of changes
2. CHANGE VELOCITY: Rate and frequency of modifications (increasing/decreasing/stable)
3. STABILITY ANALYSIS: Which sections are stable vs. volatile
4. BREAKING CHANGE HISTORY: Timeline of breaking changes
5. PATTERNS: Recurring themes or systematic changes
6. RECOMMENDATIONS: Suggestions for version management"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "analysis": message.content[0].text,
        "versions_analyzed": len(versions),
        "time_span": (versions[-1].timestamp - versions[0].timestamp).days,
        "token_usage": {
            "input": message.usage.input_tokens,
            "output": message.usage.output_tokens
        }
    }

# Example: Track API rate limit changes over time
versions = [
    VersionSnapshot(
        "v1.0", 
        datetime(2024, 1, 1),
        "Rate limit: 1000 req/hour"
    ),
    VersionSnapshot(
        "v1.1",
        datetime(2024, 2, 15),
        "Rate limit: 500 req/hour"
    ),
    VersionSnapshot(
        "v1.2",
        datetime(2024, 3, 1),
        "Rate limit: 500 req/hour\nNote: Will increase to 750 in v2.0"
    ),
    VersionSnapshot(
        "v2.0",
        datetime(2024, 4, 1),
        "Rate limit: 750 req/hour\nBurst limit: 100 req/minute"
    ),
]

timeline_analysis = analyze_version_timeline(
    versions