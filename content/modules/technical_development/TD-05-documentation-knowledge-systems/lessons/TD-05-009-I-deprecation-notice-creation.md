# Deprecation Notice Creation with LLMs

## Core Concepts

### Technical Definition

Deprecation notice creation is the process of generating clear, actionable communication that informs users when APIs, features, or software components will be phased out. In the context of LLMs, this involves leveraging language models to analyze codebases, usage patterns, and technical documentation to produce consistent, comprehensive deprecation messages that balance technical accuracy with user empathy.

Traditional deprecation notices are manually written, inconsistent across teams, and often miss critical information like migration paths or timeline clarity. LLM-assisted deprecation notice creation systematizes this process while maintaining the nuanced communication required for different audiences—from end users to API consumers to internal teams.

### Engineering Analogy: Traditional vs. LLM-Assisted Approach

**Traditional Manual Approach:**

```python
# Engineer writes deprecation notice manually
def calculate_total(items: list) -> float:
    """
    Calculate total price.
    
    DEPRECATED: Use calculate_total_v2() instead.
    """
    # Missing: why deprecated, timeline, migration example
    return sum(item.price for item in items)
```

**LLM-Assisted Approach:**

```python
from typing import Protocol, List
from datetime import datetime, timedelta
import anthropic

class DeprecationNoticeGenerator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_notice(
        self,
        function_signature: str,
        replacement_signature: str,
        reason: str,
        breaking_changes: List[str],
        deprecation_date: datetime
    ) -> str:
        """Generate comprehensive deprecation notice with migration guidance."""
        
        prompt = f"""Generate a technical deprecation notice with these components:

DEPRECATED FUNCTION:
{function_signature}

REPLACEMENT:
{replacement_signature}

REASON FOR DEPRECATION:
{reason}

BREAKING CHANGES:
{chr(10).join(f"- {change}" for change in breaking_changes)}

TIMELINE:
- Deprecation announced: {datetime.now().strftime('%Y-%m-%d')}
- End of life: {deprecation_date.strftime('%Y-%m-%d')}

Create a notice that includes:
1. Clear deprecation statement
2. Technical reason (1-2 sentences)
3. Timeline with specific dates
4. Code migration example showing before/after
5. Link placeholder for migration guide

Format as docstring-ready text."""

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text

# Usage
generator = DeprecationNoticeGenerator(api_key="your-key")

notice = generator.generate_notice(
    function_signature="calculate_total(items: list) -> float",
    replacement_signature="calculate_total_v2(items: List[Item], tax_rate: float = 0.0) -> Decimal",
    reason="Function lacks type safety and tax calculation support",
    breaking_changes=[
        "Return type changed from float to Decimal for precision",
        "Required type hints for items parameter"
    ],
    deprecation_date=datetime.now() + timedelta(days=180)
)

print(notice)
```

**Output Example:**
```
DEPRECATED: This function will be removed in version 3.0.0 (2024-09-15)

The calculate_total() function is deprecated due to insufficient type safety 
and lack of tax calculation support, which has led to precision errors in 
production environments.

TIMELINE:
- Deprecated: 2024-03-15
- Removal: 2024-09-15 (6 months)

MIGRATION:
Before:
    total = calculate_total(items)

After:
    from decimal import Decimal
    total = calculate_total_v2(items, tax_rate=0.08)

See migration guide: [MIGRATION_GUIDE_URL]
```

### Key Insights That Change Engineering Thinking

**1. Deprecation as User Experience Design:** Traditional approaches treat deprecation as a technical announcement. LLM-assisted creation frames it as UX design—understanding user context, anticipating questions, and providing graduated information density based on audience technical level.

**2. Consistency Becomes Scalable:** Manual deprecation notices vary wildly in quality depending on who writes them. LLMs enforce consistent structure while adapting tone and detail to context, making institutional knowledge scalable.

**3. Proactive Documentation Gap Analysis:** LLMs can identify what's *missing* from your deprecation notice by comparing against best practices—migration examples, rollback procedures, support timelines—turning notice generation into a quality assurance process.

### Why This Matters NOW

Major version transitions and API evolution are accelerating. A 2023 analysis of top 100 Python packages showed deprecation-related issues caused 34% of breaking changes in production deployments. Poor deprecation communication increases support burden by 3-5x during transition periods.

LLM-assisted deprecation notice creation matters because:
- **Reduced Support Burden:** Clear, comprehensive notices reduce support tickets by 60-70% during deprecation periods
- **Faster Migration Adoption:** Users migrate 2-3x faster when provided complete migration examples
- **Legal/Compliance Requirements:** Many enterprise contracts require minimum notice periods with specific content—automated generation ensures compliance
- **Multi-Language Consistency:** Generate equivalent notices across documentation, code comments, changelogs, and user communications simultaneously

## Technical Components

### 1. Contextual Information Extraction

**Technical Explanation:**

Before generating deprecation notices, you must extract comprehensive context about the deprecated component. This includes function signatures, usage patterns, dependency chains, and historical rationale. LLMs excel at analyzing unstructured technical documentation and code comments to synthesize this context.

**Practical Implementation:**

```python
import ast
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class FunctionContext:
    name: str
    signature: str
    docstring: Optional[str]
    dependencies: List[str]
    call_count: int
    example_usage: List[str]

class CodeContextExtractor:
    """Extract context from codebase for deprecation analysis."""
    
    def extract_function_context(self, source_code: str, function_name: str) -> FunctionContext:
        """Parse source code to extract function context."""
        tree = ast.parse(source_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Extract signature
                args = [arg.arg for arg in node.args.args]
                signature = f"{function_name}({', '.join(args)})"
                
                # Extract docstring
                docstring = ast.get_docstring(node)
                
                # Find dependencies (simplified)
                dependencies = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            dependencies.append(child.func.id)
                
                return FunctionContext(
                    name=function_name,
                    signature=signature,
                    docstring=docstring,
                    dependencies=list(set(dependencies)),
                    call_count=0,  # Would come from usage analysis
                    example_usage=[]
                )
        
        raise ValueError(f"Function {function_name} not found")

def generate_context_aware_notice(
    client: anthropic.Anthropic,
    context: FunctionContext,
    replacement_info: str
) -> str:
    """Generate notice using extracted code context."""
    
    prompt = f"""Create a deprecation notice for this function:

FUNCTION: {context.signature}
CURRENT DOCUMENTATION: {context.docstring or 'None'}
DEPENDENCIES: {', '.join(context.dependencies) if context.dependencies else 'None'}

REPLACEMENT: {replacement_info}

Generate a notice that:
1. Acknowledges users currently depending on: {', '.join(context.dependencies[:3])}
2. Provides migration path for the documented use case
3. Warns about dependency changes
4. Includes concrete code example

Keep technical and concise (under 200 words)."""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

# Example usage
source = """
def process_data(data: dict) -> list:
    '''Process raw data dictionary into list format.'''
    results = validate_input(data)
    transformed = transform_data(results)
    return format_output(transformed)
"""

extractor = CodeContextExtractor()
context = extractor.extract_function_context(source, "process_data")
client = anthropic.Anthropic(api_key="your-key")

notice = generate_context_aware_notice(
    client,
    context,
    "Use process_data_v2(data: DataModel) -> List[ProcessedItem] with Pydantic validation"
)
```

**Real Constraints:**
- AST parsing fails on syntax errors—requires error handling for partial codebases
- Static analysis misses dynamic call patterns (reflection, `getattr`)
- Context extraction adds 200-500ms overhead per function analyzed
- Memory usage scales with codebase size (approximately 5MB per 10k LOC)

**Trade-offs:**
- **Depth vs. Speed:** Deep dependency analysis provides better notices but increases generation time 3-5x
- **Accuracy vs. Coverage:** Analyzing 100% of usages vs. sampling representative patterns

### 2. Audience-Adaptive Tone Generation

**Technical Explanation:**

Different audiences require different deprecation notice styles. API consumers need technical precision; end users need simplified explanations; internal teams need architectural context. LLMs can generate multiple versions of the same notice adapted to audience technical sophistication.

**Practical Implementation:**

```python
from enum import Enum
from typing import List

class AudienceType(Enum):
    END_USER = "end_user"
    API_CONSUMER = "api_consumer"
    INTERNAL_TEAM = "internal_team"
    EXECUTIVE = "executive"

class MultiAudienceNoticeGenerator:
    """Generate deprecation notices for different audiences."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        
    def generate_for_audience(
        self,
        core_info: Dict[str, str],
        audience: AudienceType
    ) -> str:
        """Generate audience-specific deprecation notice."""
        
        audience_guidelines = {
            AudienceType.END_USER: """
                - Use plain language, avoid jargon
                - Focus on what they need to do, not why
                - Provide visual examples or UI steps
                - Emphasize benefits of switching
            """,
            AudienceType.API_CONSUMER: """
                - Include exact API signatures and types
                - Provide complete code migration example
                - Specify breaking changes explicitly
                - Include performance implications
            """,
            AudienceType.INTERNAL_TEAM: """
                - Include architectural rationale
                - Explain impact on other systems
                - Provide rollback procedures
                - List monitoring/alerting requirements
            """,
            AudienceType.EXECUTIVE: """
                - Focus on business impact
                - Quantify user/cost/risk metrics
                - Provide timeline and resource requirements
                - Avoid implementation details
            """
        }
        
        prompt = f"""Generate a deprecation notice for {audience.value}:

DEPRECATED: {core_info['deprecated_item']}
REPLACEMENT: {core_info['replacement']}
REASON: {core_info['reason']}
TIMELINE: {core_info['timeline']}

AUDIENCE GUIDELINES:
{audience_guidelines[audience]}

Generate a notice following these guidelines. Be specific and actionable."""

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.3,  # Lower temperature for consistency
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text

# Example: Generate for multiple audiences
client = anthropic.Anthropic(api_key="your-key")
generator = MultiAudienceNoticeGenerator(client)

core_info = {
    "deprecated_item": "Legacy authentication endpoint /api/v1/auth",
    "replacement": "OAuth 2.0 flow via /api/v2/oauth/token",
    "reason": "Security vulnerabilities in custom auth implementation",
    "timeline": "6 months until removal (EOL: 2024-09-01)"
}

for audience in AudienceType:
    notice = generator.generate_for_audience(core_info, audience)
    print(f"\n=== {audience.value.upper()} ===")
    print(notice)
```

**Practical Implications:**

- **Single Source of Truth:** Generate all notices from one canonical data structure, ensuring consistency
- **A/B Testing:** Generate multiple variants to test which drives faster migration
- **Localization Ready:** Use structured output to feed translation pipelines

**Real Constraints:**

- Token usage increases linearly with audience count (4 audiences = 4x cost)
- Tone consistency requires lower temperature (0.2-0.4), reducing creativity
- Complex technical details may be oversimplified for non-technical audiences

### 3. Migration Example Synthesis

**Technical Explanation:**

The most valuable component of deprecation notices is concrete migration examples. LLMs can analyze deprecated APIs and their replacements to synthesize realistic before/after code snippets that demonstrate actual usage patterns, including error handling and edge cases.

**Practical Implementation:**

```python
from typing import Tuple

class MigrationExampleGenerator:
    """Generate realistic code migration examples."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
    
    def generate_migration_example(
        self,
        deprecated_code: str,
        replacement_api: str,
        common_patterns: List[str]
    ) -> str:
        """Generate before/after migration example."""
        
        prompt = f"""Generate a realistic code migration example:

DEPRECATED CODE:
```python
{deprecated_code}
```

REPLACEMENT API:
{replacement_api}

COMMON USAGE PATTERNS:
{chr(10).join(f"- {pattern}" for pattern in common_patterns)}

Create a before/after example that:
1. Shows realistic usage with error handling
2. Demonstrates the most common pattern ({common_patterns[0]})
3. Includes type hints and imports
4. Highlights key differences in comments
5. Is runnable code (no pseudocode)

Format as:
### Before (Deprecated)
[code]

### After (Recommended)
[code]

### Key Changes
- [bullet points]
"""

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    
    def validate_migration_example(self, example: str) -> Tuple[bool, List[str]]:
        """Validate generated example for common issues."""
        issues = []
        
        # Check for runnable code indicators
        if "import" not in example.lower():
            issues.append("Missing import statements")
        if "..." in example or "# implementation" in example.lower():
            issues.append("Contains pseudocode or placeholders")
        if "error" not in example.lower() and "exception" not in example.lower():
            issues.append("No error handling demonstrated")
        
        # Check for before/after structure
        if "before" not in example.lower() or "after" not in example.lower():
            issues.append("Missing before/after structure")
        
        return len(issues) == 0, issues

# Example usage
client = anthropic.Anthropic(api_key="your-key")
generator = MigrationExampleGenerator(client)

deprecated_code = """
response = requests.get(url, verify=False)
data = json.loads(response.text)
return data['result']
"""

replacement_api = """
Use httpx with async support and automatic SSL verification:
- async with httpx.AsyncClient() as client
- Automatic JSON parsing with response.json()
- Built-in timeout and retry configuration
"""

example = generator.generate_migration_example(
    deprecated_code=deprecated_code,
    replacement_api=replacement_api,