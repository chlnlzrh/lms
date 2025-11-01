# Pull Request Description Templates: Engineering Communication Through Structured Prompts

## Core Concepts

A pull request description is a structured prompt that transforms unorganized code changes into actionable context for both human reviewers and AI systems. Traditional PR descriptions are freeform text that developers rush through, treating them as bureaucratic overhead. Modern PR templates are engineered interfaces that extract maximum value from minimal input by providing consistent structure that both humans and LLMs can parse, index, and act upon.

### Traditional vs. Structured Approach

```python
# Traditional: Unstructured PR description
"""
Fixed the bug with user login. Also updated some dependencies and 
refactored the auth module. Should work now.
"""

# Modern: Structured template as a prompt
"""
## Change Type
- [x] Bug Fix
- [ ] Feature
- [ ] Refactor

## Problem Statement
Users experiencing 403 errors on login when session cookies expire 
mid-request (affects ~2% of daily active users based on error logs).

## Solution Approach
Modified session validation middleware to refresh expired tokens 
before authentication check rather than after.

## Technical Changes
- `auth/middleware.py`: Added token refresh in `validate_session()`
- `auth/tokens.py`: Extracted `is_token_expired()` helper
- `tests/auth/test_middleware.py`: Added expiration edge cases

## Testing Evidence
- Unit tests: All 47 auth tests passing
- Manual verification: Tested expired session flow in staging
- Error rate: Dropped from 2.1% to 0.3% in staging over 24h

## Deployment Notes
- No migration required
- No configuration changes
- Backward compatible with existing sessions
"""
```

The structured version provides explicit slots for critical information, making it impossible to skip essential context. More importantly, it creates a consistent format that enables automated tooling, AI-powered review assistance, and reliable documentation generation.

### Why This Matters Now

LLMs have transformed PR descriptions from write-only documentation into interactive interfaces. When you structure a PR description properly, you create:

1. **Machine-readable context** that AI code review tools can parse without hallucinating missing information
2. **Consistent training data** for custom models that learn your team's code patterns
3. **Retrievable knowledge** that vector databases can embed and surface during future development
4. **Reproducible prompts** that generate accurate summaries, release notes, and impact analyses

The median time saved per PR review is 7-12 minutes when reviewers receive structured context versus unstructured descriptions. For a team of 10 engineers reviewing 5 PRs daily, that's 17-29 hours per week—more than half an engineer's time.

### Key Insights

**Insight 1: Templates are prompts for humans and machines alike.** Every field in your PR template is a prompt that elicits specific information. Asking "What changed?" yields vague answers. Asking "Which files contain business logic changes vs. test changes?" yields precise, actionable information.

**Insight 2: Structure enables automation.** Unstructured text requires LLMs to extract structure through inference (expensive, error-prone). Pre-structured input allows deterministic parsing (fast, reliable) with LLMs only handling semantic understanding.

**Insight 3: Constraints accelerate writing.** Developers spend less time on structured templates because decision fatigue is eliminated. The template tells you exactly what to write and where.

## Technical Components

### 1. Field Structure and Semantic Slots

Each field in a PR template serves as a semantic slot that captures a specific type of information. The key is designing slots that map to decision-making needs rather than documentation conventions.

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class ChangeType(Enum):
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    DOCS = "docs"
    PERF = "performance"
    SECURITY = "security"

class ImpactLevel(Enum):
    BREAKING = "breaking"  # API changes, migrations required
    MAJOR = "major"        # Behavior changes, testing required
    MINOR = "minor"        # Isolated changes, low risk
    TRIVIAL = "trivial"    # Docs, comments, formatting

@dataclass
class PRDescription:
    """Structured PR metadata that both humans and LLMs can parse."""
    
    change_type: ChangeType
    impact_level: ImpactLevel
    problem_statement: str  # What problem does this solve?
    solution_approach: str  # How does this solve it?
    changed_components: List[str]  # Which modules/files changed?
    test_evidence: str  # How do we know it works?
    deployment_notes: Optional[str] = None
    rollback_plan: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Generate template-formatted description."""
        template = f"""## Change Classification
**Type:** {self.change_type.value}
**Impact:** {self.impact_level.value}

## Problem Statement
{self.problem_statement}

## Solution Approach
{self.solution_approach}

## Changed Components
{chr(10).join(f'- `{c}`' for c in self.changed_components)}

## Testing Evidence
{self.test_evidence}
"""
        if self.deployment_notes:
            template += f"\n## Deployment Notes\n{self.deployment_notes}\n"
        if self.rollback_plan:
            template += f"\n## Rollback Plan\n{self.rollback_plan}\n"
        return template

# Usage
pr = PRDescription(
    change_type=ChangeType.BUGFIX,
    impact_level=ImpactLevel.MINOR,
    problem_statement="Session tokens not refreshing before expiration checks",
    solution_approach="Reordered middleware to refresh tokens before validation",
    changed_components=[
        "auth/middleware.py",
        "auth/tokens.py",
        "tests/auth/test_middleware.py"
    ],
    test_evidence="47 unit tests passing, staging error rate 0.3% (was 2.1%)",
    deployment_notes="No migration required, backward compatible"
)

print(pr.to_markdown())
```

**Practical implications:** Using enums for classification fields ensures consistency and enables filtering. A dataclass representation allows programmatic generation from git metadata, making it possible to pre-fill templates automatically.

**Trade-offs:** More structure means more upfront effort. The optimal complexity level captures just enough information to make review decisions without becoming burdensome. Fields that are always left empty should be removed.

### 2. Progressive Disclosure

Not all PRs need the same level of detail. A one-line documentation fix shouldn't require the same template as a security patch. Progressive disclosure means showing relevant fields based on change type.

```python
from typing import Protocol

class TemplateGenerator(Protocol):
    """Interface for generating context-appropriate templates."""
    
    def generate(self, change_type: ChangeType, 
                 impact_level: ImpactLevel) -> str:
        """Generate template with appropriate sections."""
        ...

class AdaptiveTemplateGenerator:
    """Generates templates that adapt to change complexity."""
    
    MINIMAL_FIELDS = ["change_type", "summary"]
    STANDARD_FIELDS = MINIMAL_FIELDS + [
        "problem_statement", 
        "solution_approach", 
        "test_evidence"
    ]
    COMPREHENSIVE_FIELDS = STANDARD_FIELDS + [
        "security_considerations",
        "performance_impact", 
        "deployment_notes",
        "rollback_plan",
        "monitoring_plan"
    ]
    
    def generate(self, change_type: ChangeType, 
                 impact_level: ImpactLevel) -> str:
        """Select fields based on change characteristics."""
        
        if change_type == ChangeType.DOCS or impact_level == ImpactLevel.TRIVIAL:
            fields = self.MINIMAL_FIELDS
        elif impact_level == ImpactLevel.BREAKING or change_type == ChangeType.SECURITY:
            fields = self.COMPREHENSIVE_FIELDS
        else:
            fields = self.STANDARD_FIELDS
        
        return self._build_template(fields)
    
    def _build_template(self, fields: List[str]) -> str:
        """Construct markdown template from field list."""
        sections = []
        
        field_templates = {
            "change_type": "## Change Type\n- [ ] Feature\n- [ ] Bug Fix\n- [ ] Refactor\n- [ ] Docs",
            "summary": "## Summary\n<!-- One-line description -->",
            "problem_statement": "## Problem Statement\n<!-- What problem are we solving? -->",
            "solution_approach": "## Solution Approach\n<!-- How does this change solve it? -->",
            "test_evidence": "## Testing Evidence\n<!-- How do we know this works? -->",
            "security_considerations": "## Security Considerations\n<!-- Authentication, authorization, data exposure impacts -->",
            "performance_impact": "## Performance Impact\n<!-- Benchmarks, profiling results, resource usage -->",
            "deployment_notes": "## Deployment Notes\n<!-- Migration steps, configuration changes, feature flags -->",
            "rollback_plan": "## Rollback Plan\n<!-- How to safely revert if issues arise -->",
            "monitoring_plan": "## Monitoring Plan\n<!-- Metrics to watch, alerts to add -->"
        }
        
        for field in fields:
            if field in field_templates:
                sections.append(field_templates[field])
        
        return "\n\n".join(sections)

# Usage example
generator = AdaptiveTemplateGenerator()

# Trivial documentation change
minimal_template = generator.generate(
    ChangeType.DOCS, 
    ImpactLevel.TRIVIAL
)
print("=== MINIMAL TEMPLATE ===")
print(minimal_template)

# Breaking security fix
comprehensive_template = generator.generate(
    ChangeType.SECURITY,
    ImpactLevel.BREAKING
)
print("\n=== COMPREHENSIVE TEMPLATE ===")
print(comprehensive_template)
```

**Practical implications:** Adaptive templates reduce cognitive load by only asking for relevant information. This increases template adoption because developers don't feel they're filling out forms unnecessarily.

**Constraints:** The logic for determining which template to use must be transparent. Developers should understand why they're seeing specific fields and be able to override when the heuristic is wrong.

### 3. Automated Field Population

The best template field is one the developer doesn't have to fill manually. Many PR description fields can be auto-populated from git metadata, code analysis, or CI results.

```python
import subprocess
import re
from pathlib import Path
from typing import Set, Dict, Any

class PRMetadataExtractor:
    """Extract PR description fields from git and codebase analysis."""
    
    def __init__(self, base_branch: str = "main"):
        self.base_branch = base_branch
    
    def extract_all(self) -> Dict[str, Any]:
        """Extract all auto-fillable metadata."""
        return {
            "changed_files": self.get_changed_files(),
            "changed_components": self.identify_components(),
            "test_files_changed": self.get_test_changes(),
            "commit_summary": self.summarize_commits(),
            "potential_impact": self.assess_impact()
        }
    
    def get_changed_files(self) -> List[str]:
        """List all files changed relative to base branch."""
        result = subprocess.run(
            ["git", "diff", "--name-only", self.base_branch],
            capture_output=True,
            text=True,
            check=True
        )
        return [f for f in result.stdout.strip().split("\n") if f]
    
    def identify_components(self) -> Set[str]:
        """Map changed files to logical components."""
        changed_files = self.get_changed_files()
        components = set()
        
        # Simple heuristic: top-level directory is component
        for filepath in changed_files:
            parts = Path(filepath).parts
            if len(parts) > 1 and not parts[0].startswith("."):
                components.add(parts[0])
        
        return components
    
    def get_test_changes(self) -> Dict[str, int]:
        """Count test vs. non-test file changes."""
        changed_files = self.get_changed_files()
        
        test_files = [f for f in changed_files if "test" in f]
        source_files = [f for f in changed_files if "test" not in f]
        
        return {
            "test_files": len(test_files),
            "source_files": len(source_files),
            "test_coverage_ratio": len(test_files) / len(source_files) if source_files else 0
        }
    
    def summarize_commits(self) -> str:
        """Generate summary from commit messages."""
        result = subprocess.run(
            ["git", "log", "--oneline", f"{self.base_branch}..HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        
        commits = result.stdout.strip().split("\n")
        if len(commits) == 1:
            return commits[0].split(" ", 1)[1] if " " in commits[0] else commits[0]
        
        return f"{len(commits)} commits: {commits[0].split(' ', 1)[1] if ' ' in commits[0] else commits[0]}..."
    
    def assess_impact(self) -> ImpactLevel:
        """Heuristic impact assessment based on changes."""
        test_changes = self.get_test_changes()
        changed_files = self.get_changed_files()
        
        # Check for migration files
        has_migrations = any("migration" in f for f in changed_files)
        
        # Check for API changes
        has_api_changes = any(re.search(r"api|endpoint|route", f) for f in changed_files)
        
        # Check for database schema changes
        has_schema_changes = any(re.search(r"model|schema|database", f) for f in changed_files)
        
        if has_migrations or has_schema_changes:
            return ImpactLevel.BREAKING
        elif has_api_changes:
            return ImpactLevel.MAJOR
        elif test_changes["source_files"] <= 2:
            return ImpactLevel.MINOR
        else:
            return ImpactLevel.TRIVIAL

# Usage
extractor = PRMetadataExtractor(base_branch="main")
metadata = extractor.extract_all()

print(f"Changed components: {metadata['changed_components']}")
print(f"Test coverage: {metadata['test_files_changed']['test_coverage_ratio']:.1%}")
print(f"Impact assessment: {metadata['potential_impact'].value}")
```

**Practical implications:** Auto-population reduces friction dramatically. Developers can review and edit pre-filled templates rather than starting from scratch, decreasing PR description time from 5-10 minutes to 1-2 minutes.

**Trade-offs:** Heuristics for auto-population will be wrong sometimes. The key is making auto-filled content obviously editable and not hiding the logic, so developers understand why the system made specific suggestions.

### 4. Validation and Quality Gates

A template is only valuable if developers actually fill it out. Automated validation ensures minimum quality standards are met before review begins.

```python
from typing import List, Tuple
import re

class PRDescriptionValidator:
    """Validate PR descriptions meet quality standards."""
    
    def __init__(self, min_problem_statement_words: int = 10,
                 min_test_evidence_words: int = 5):
        self.min_problem_statement_words = min_problem_statement_words
        self.min_test_evidence_words = min_test_evidence_words
    
    def validate(self, description: str) -> Tuple[bool, List[str]]:
        """
        Validate PR description quality.
        
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # Extract sections using regex
        sections = self._parse_sections(description)
        
        # Check for required sections
        required = ["Problem Statement", "Solution Approach", "Testing Evidence"]
        for section in required:
            if section not in sections:
                issues.append(f"Missing required section: {section}")
            elif not sections[section].strip():
                issues.append(f"Section is empty: {section}")
        
        # Validate problem statement depth
        if "Problem Statement" in sections:
            word_count = len(sections["Problem Statement"].split())
            if word_count < self.min_problem_statement_words:
                