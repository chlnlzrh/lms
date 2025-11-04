# Migration Guide Automation with LLMs

## Core Concepts

**Technical Definition:** Migration guide automation uses large language models to analyze source code, documentation, and API specifications to generate accurate, context-aware migration instructions when upgrading dependencies, frameworks, or language versions. Unlike static code analysis or find-and-replace tools, LLMs understand semantic relationships between code patterns and can generate explanations alongside transformations.

### Engineering Analogy: Traditional vs. LLM-Assisted Migration

**Traditional Approach:**
```python
import re
from typing import List, Tuple

def generate_migration_guide_traditional(
    old_code: str,
    old_api_patterns: List[Tuple[str, str]]
) -> str:
    """Traditional regex-based migration guide generation"""
    migrations = []
    
    for old_pattern, new_pattern in old_api_patterns:
        # Simple pattern matching - no semantic understanding
        if re.search(old_pattern, old_code):
            migrations.append(
                f"Replace '{old_pattern}' with '{new_pattern}'"
            )
    
    return "\n".join(migrations) if migrations else "No migrations needed"

# Example usage
old_code = """
user = User.find_by_name('John')
posts = Post.where(author: user).order('created_at DESC')
"""

patterns = [
    (r"find_by_(\w+)", r"find_by(\1: value)"),
    (r"\.where\((\w+):", r".where(\1 ==")
]

print(generate_migration_guide_traditional(old_code, patterns))
# Output: Generic pattern matches without context
```

**LLM-Assisted Approach:**
```python
from typing import Dict, List
import anthropic
import json

def generate_migration_guide_llm(
    old_code: str,
    old_version: str,
    new_version: str,
    framework: str,
    api_changes_doc: str
) -> Dict[str, any]:
    """LLM-based migration with semantic understanding"""
    
    client = anthropic.Anthropic()
    
    prompt = f"""Analyze this {framework} code from version {old_version} 
and generate a migration guide to version {new_version}.

OLD CODE:
```
{old_code}
```

API CHANGES DOCUMENTATION:
{api_changes_doc}

Generate a JSON response with:
1. "changes": List of required changes with line numbers
2. "new_code": Complete migrated code
3. "rationale": Explanation for each change
4. "risks": Potential issues or behavioral differences
5. "test_recommendations": What to test after migration
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)

# Example usage
old_rails_code = """
class UsersController < ApplicationController
  def index
    @users = User.find(:all, conditions: ['active = ?', true])
    render json: @users.to_json(only: [:id, :name])
  end
end
"""

api_changes = """
Rails 6 to 7 Changes:
- find(:all) deprecated, use .all or .where instead
- :conditions hash deprecated, use where() method
- to_json options changed, use as_json with explicit fields
"""

result = generate_migration_guide_llm(
    old_rails_code,
    "6.x",
    "7.x",
    "Ruby on Rails",
    api_changes
)

print(json.dumps(result, indent=2))
```

The LLM approach understands that `find(:all, conditions: ...)` should become `where(...)`, recognizes the Rails-specific patterns, and explains *why* each change is necessary. It catches cascading implications (like deprecation warnings that will become errors) that regex cannot detect.

### Key Engineering Insights

**1. Context Accumulation Matters More Than Pattern Matching**

Traditional migration tools operate on syntax trees; LLMs operate on semantic understanding. When you feed an LLM the old API documentation, new API documentation, and your code simultaneously, it builds a mental model of intent versus implementation. This means it can suggest migrations even for code patterns that aren't explicitly documented in the changelog.

**2. Migration Confidence Scoring Is Critical**

Not all LLM suggestions are equally reliable. High-quality migration automation includes confidence scoring:

```python
def score_migration_confidence(
    original_code: str,
    suggested_migration: str,
    api_docs: str,
    model_response: Dict
) -> float:
    """Calculate confidence score for migration suggestion"""
    
    confidence_factors = {
        "exact_api_match": 0.0,      # Found in official docs
        "pattern_similarity": 0.0,    # Similar to documented patterns
        "breaking_change": 0.0,       # Involves known breaking changes
        "complexity": 0.0,            # Complexity of transformation
        "test_coverage": 0.0          # Whether tests exist
    }
    
    # Check if migration target appears in docs
    if suggested_migration.split('(')[0] in api_docs:
        confidence_factors["exact_api_match"] = 0.4
    
    # Penalize complex transformations
    complexity = abs(len(suggested_migration) - len(original_code)) / len(original_code)
    confidence_factors["complexity"] = max(0, 0.2 - complexity * 0.1)
    
    # Reward breaking change awareness
    if "BREAKING" in model_response.get("rationale", ""):
        confidence_factors["breaking_change"] = 0.3
    
    return sum(confidence_factors.values())
```

**3. Iterative Validation Loops Catch Errors Early**

Generate migration → attempt compilation/linting → feed errors back to LLM → refine. This loop typically converges in 2-3 iterations and catches 90%+ of migration errors before human review.

### Why This Matters Now

**Dependency Upgrade Velocity Has Outpaced Tooling:** Modern frameworks release major versions every 12-18 months with breaking changes. Teams delay upgrades for 6+ months purely due to migration uncertainty. LLM-assisted migration reduces this delay by providing instant, context-aware guidance.

**Technical Debt Compounds Faster:** Each delayed upgrade makes the next migration harder (exponential complexity growth). Automated migration guides enable continuous small upgrades instead of painful big-bang migrations.

**Developer Time is the Bottleneck:** A senior engineer spending 3 days on a framework migration represents $5-10K in opportunity cost. If LLMs can reduce that to 4 hours of review and validation, the ROI is immediate and measurable.

## Technical Components

### 1. Code Context Extraction

**Technical Explanation:** Before generating migration guidance, you must extract the relevant context: direct dependencies, transitive dependencies, configuration files, and usage patterns. LLMs perform better with structured context than raw repository dumps.

```python
import ast
import os
from pathlib import Path
from typing import Dict, List, Set

class CodeContextExtractor:
    """Extract migration-relevant context from codebase"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.dependency_graph: Dict[str, Set[str]] = {}
        
    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file"""
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
            
            return imports
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return set()
    
    def find_usage_patterns(
        self,
        target_package: str,
        file_path: Path
    ) -> List[Dict[str, any]]:
        """Find how a package is used in specific file"""
        patterns = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Find function calls to target package
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        # e.g., requests.get(...)
                        if hasattr(node.func.value, 'id'):
                            if node.func.value.id == target_package:
                                patterns.append({
                                    "type": "function_call",
                                    "function": node.func.attr,
                                    "line": node.lineno,
                                    "args_count": len(node.args),
                                    "has_kwargs": len(node.keywords) > 0
                                })
        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
        
        return patterns
    
    def build_migration_context(
        self,
        target_package: str,
        max_files: int = 50
    ) -> Dict[str, any]:
        """Build comprehensive context for migration"""
        context = {
            "target_package": target_package,
            "affected_files": [],
            "usage_summary": {},
            "total_occurrences": 0
        }
        
        python_files = list(self.repo_path.rglob("*.py"))[:max_files]
        
        for file_path in python_files:
            imports = self.extract_imports(file_path)
            
            if target_package in imports:
                patterns = self.find_usage_patterns(target_package, file_path)
                
                if patterns:
                    context["affected_files"].append({
                        "path": str(file_path.relative_to(self.repo_path)),
                        "patterns": patterns
                    })
                    
                    # Summarize usage patterns
                    for pattern in patterns:
                        func = pattern["function"]
                        context["usage_summary"][func] = \
                            context["usage_summary"].get(func, 0) + 1
                        context["total_occurrences"] += 1
        
        return context

# Example usage
extractor = CodeContextExtractor("/path/to/repo")
context = extractor.build_migration_context("requests")

print(f"Found {context['total_occurrences']} uses across "
      f"{len(context['affected_files'])} files")
print(f"Most common: {sorted(context['usage_summary'].items(), "
      f"key=lambda x: x[1], reverse=True)[:3]}")
```

**Practical Implications:** Without proper context extraction, LLMs generate generic advice. With it, they provide file-specific, line-specific guidance. This code identifies that you use `requests.get()` 47 times but `requests.post()` only twice—allowing the LLM to prioritize migration guidance.

**Real Constraints:**
- Large repositories (10K+ files) require sampling strategies
- Binary files and generated code should be excluded
- Abstract syntax tree parsing fails on syntax errors (handle gracefully)

### 2. Changelog Semantic Parsing

**Technical Explanation:** Framework changelogs vary wildly in structure. Some use semantic versioning with detailed breaking change sections; others are unstructured blog posts. Parsing these into machine-readable formats improves LLM accuracy.

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import re

class ChangeType(Enum):
    BREAKING = "breaking"
    DEPRECATION = "deprecation"
    NEW_FEATURE = "new_feature"
    BUG_FIX = "bug_fix"
    SECURITY = "security"

@dataclass
class APIChange:
    change_type: ChangeType
    old_signature: Optional[str]
    new_signature: Optional[str]
    description: str
    version_introduced: str
    code_example: Optional[str] = None

class ChangelogParser:
    """Parse unstructured changelogs into structured format"""
    
    BREAKING_INDICATORS = [
        "BREAKING", "breaking change", "removed", "no longer",
        "incompatible", "must migrate"
    ]
    
    DEPRECATION_INDICATORS = [
        "deprecated", "will be removed", "legacy", "obsolete"
    ]
    
    def parse_markdown_changelog(
        self,
        changelog_text: str,
        version: str
    ) -> List[APIChange]:
        """Extract structured changes from markdown changelog"""
        changes = []
        lines = changelog_text.split('\n')
        
        current_section = None
        current_description = []
        
        for i, line in enumerate(lines):
            # Detect section headers
            if line.startswith('##'):
                if current_description:
                    # Process previous section
                    changes.extend(
                        self._extract_changes_from_section(
                            current_section,
                            '\n'.join(current_description),
                            version
                        )
                    )
                    current_description = []
                
                current_section = line.strip('#').strip()
            else:
                current_description.append(line)
        
        # Process final section
        if current_description:
            changes.extend(
                self._extract_changes_from_section(
                    current_section,
                    '\n'.join(current_description),
                    version
                )
            )
        
        return changes
    
    def _extract_changes_from_section(
        self,
        section_title: Optional[str],
        section_content: str,
        version: str
    ) -> List[APIChange]:
        """Extract individual changes from a section"""
        changes = []
        
        # Determine change type from section title and content
        change_type = self._determine_change_type(
            section_title or "",
            section_content
        )
        
        # Extract old/new signatures from code blocks
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', 
                                 section_content, 
                                 re.DOTALL)
        
        # Split into bullet points
        items = re.split(r'\n[-*]\s+', section_content)
        
        for item in items:
            if len(item.strip()) < 10:
                continue
            
            # Extract old -> new patterns
            signature_match = re.search(
                r'`([^`]+)`\s*(?:->|→|to|replaced by)\s*`([^`]+)`',
                item
            )
            
            old_sig = signature_match.group(1) if signature_match else None
            new_sig = signature_match.group(2) if signature_match else None
            
            # Find associated code example
            code_example = None
            if code_blocks:
                # Simple heuristic: use next code block
                code_example = code_blocks.pop(0) if code_blocks else None
            
            changes.append(APIChange(
                change_type=change_type,
                old_signature=old_sig,
                new_signature=new_sig,
                description=item.strip(),
                version_introduced=version,
                code_example=code_example
            ))
        
        return changes
    
    def _determine_change_type(
        self,
        title: str,
        content: str
    ) -> ChangeType:
        """Classify change type based on keywords"""
        text = (title + " " + content).lower()
        
        if any(indicator in text for indicator in self.BREAKING_INDICATORS):
            return ChangeType.BREAKING
        elif any(indicator in text for indicator in self.DEPRECATION_INDICATORS):
            return ChangeType.DEPRECATION
        elif "security" in text or "CVE" in text:
            return ChangeType.SECURITY
        elif "fix" in text or "bug" in text:
            return ChangeType.BUG_FIX
        else:
            return ChangeType.NEW_FEATURE

# Example usage
changelog = """
## Breaking Changes in v3.0

- `fetch()` method signature