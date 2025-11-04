# Code Review Augmentation with LLMs

## Core Concepts

Code review augmentation uses language models to automate repetitive aspects of code review while preserving human judgment for architectural decisions and business logic validation. Unlike traditional static analysis tools that rely on pattern matching and abstract syntax trees, LLM-based reviewers understand code semantically—they can evaluate naming conventions contextually, identify logic errors that follow syntactic rules, and suggest refactorings based on broader code patterns.

### Traditional vs. LLM-Augmented Code Review

**Traditional automated review:**
```python
# Traditional linter catches this
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price  # Linter: Missing null check
    return total
```

**LLM-augmented review catches deeper issues:**
```python
# LLM can identify this subtle logic error
def calculate_discount(price, user_type):
    if user_type == "premium":
        return price * 0.9
    elif user_type == "regular":
        return price * 0.95
    # LLM notices: What happens with "guest" users?
    # LLM suggests: Missing default case and error handling
```

The key difference: static analyzers find syntactic violations; LLMs identify semantic problems, incomplete logic, unclear intent, and context-inappropriate patterns.

### Why This Matters Now

Modern codebases change rapidly. A senior engineer spending 2 hours daily on code reviews represents significant opportunity cost. LLMs can handle the mechanical aspects—style consistency, common error patterns, documentation completeness—freeing reviewers to focus on architectural implications and domain logic correctness.

More importantly, LLM reviewers provide instant feedback. Developers get initial review comments within seconds of opening a pull request, fixing issues before human reviewers even see the code. This reduces review cycles from days to hours.

The technology has crossed a critical threshold: false positive rates for common issues have dropped below 15%, making automated suggestions trustworthy enough for production use with human oversight.

## Technical Components

### 1. Contextual Code Understanding

LLMs process code as sequences of tokens, building representations that capture semantic relationships beyond syntax. When reviewing a function, the model considers variable naming patterns, control flow logic, error handling approaches, and how the code fits within the broader repository context.

**Technical Mechanism:**

```python
from typing import List, Dict, Optional
import anthropic

def analyze_code_context(
    code_snippet: str,
    file_path: str,
    repository_context: Optional[str] = None
) -> Dict[str, any]:
    """
    Analyze code with full contextual understanding.
    
    Args:
        code_snippet: The code to review
        file_path: File path for context about purpose
        repository_context: Relevant surrounding code/docs
    
    Returns:
        Structured analysis with issues and suggestions
    """
    client = anthropic.Anthropic()
    
    prompt = f"""Analyze this code for correctness and clarity:

File: {file_path}

Code:
```
{code_snippet}
```

{f"Repository Context:\n{repository_context}" if repository_context else ""}

Identify:
1. Logic errors (incorrect conditions, missing cases)
2. Clarity issues (unclear names, confusing flow)
3. Missing error handling
4. Potential bugs (edge cases, race conditions)

Format as JSON:
{{
  "severity": "high|medium|low",
  "issues": [
    {{
      "line": <number>,
      "type": "logic|clarity|error-handling|bug",
      "description": "<specific issue>",
      "suggestion": "<concrete fix>"
    }}
  ]
}}"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    import json
    return json.loads(response.content[0].text)
```

**Practical Implications:**

Context matters enormously. A function named `process()` might be acceptable in a single-purpose utility script but problematic in a large service. LLMs evaluate appropriateness based on file path, surrounding code, and common patterns.

**Trade-offs:**

- **Token Limits:** Context windows are finite (200K tokens ≈ 150K words). You can't send entire repositories. Strategic context selection is critical.
- **Latency:** Full context analysis takes 2-5 seconds. Batch similar checks or use streaming for incremental results.
- **Cost:** Processing 10K tokens costs ~$0.03. Reviewing 100 PRs daily with 2K tokens each = $6/day per repository.

### 2. Multi-Level Issue Detection

Effective code review operates at multiple abstraction levels: syntax (caught by parsers), semantics (logic correctness), pragmatics (real-world usage), and architecture (system-level implications).

**Implementation Example:**

```python
from enum import Enum
from dataclasses import dataclass
from typing import List

class IssueLevel(Enum):
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    PRAGMATIC = "pragmatic"
    ARCHITECTURAL = "architectural"

@dataclass
class ReviewIssue:
    level: IssueLevel
    severity: str
    line: int
    message: str
    suggestion: str
    auto_fixable: bool

def multi_level_review(code: str, context: str) -> List[ReviewIssue]:
    """
    Perform layered code analysis at multiple abstraction levels.
    """
    client = anthropic.Anthropic()
    
    analysis_prompt = f"""Review this code at multiple levels:

Code:
```python
{code}
```

Context: {context}

Analyze at these levels:

1. SEMANTIC: Logic errors, incorrect algorithms, edge case handling
2. PRAGMATIC: Real-world usability, error messages, debugging support
3. ARCHITECTURAL: Design patterns, coupling, maintainability

For each issue found:
- Line number
- Level (semantic/pragmatic/architectural)
- Severity (high/medium/low)
- Specific problem
- Concrete fix
- Whether auto-fixable

Return as JSON array."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=3000,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    import json
    raw_issues = json.loads(response.content[0].text)
    
    return [
        ReviewIssue(
            level=IssueLevel(issue['level']),
            severity=issue['severity'],
            line=issue['line'],
            message=issue['problem'],
            suggestion=issue['fix'],
            auto_fixable=issue['auto_fixable']
        )
        for issue in raw_issues
    ]
```

**Real Constraints:**

LLMs excel at semantic and pragmatic analysis but may miss architectural issues requiring deep domain knowledge. A pattern that seems reasonable in isolation might violate team-specific architectural principles. Always combine LLM insights with human architectural review.

### 3. Differential Analysis

The most valuable reviews focus on changes, not entire files. Differential analysis examines what changed, why, and whether the change introduces new issues or fixes existing ones.

**Implementation:**

```python
import difflib
from typing import Tuple, List

def extract_changed_sections(
    original: str,
    modified: str
) -> List[Tuple[int, str, str]]:
    """
    Extract changed code sections with line numbers.
    
    Returns list of (line_num, old_code, new_code) tuples.
    """
    original_lines = original.splitlines()
    modified_lines = modified.splitlines()
    
    differ = difflib.unified_diff(
        original_lines,
        modified_lines,
        lineterm='',
        n=3  # Context lines
    )
    
    changes = []
    current_line = 0
    
    for line in differ:
        if line.startswith('@@'):
            # Parse line number from diff header
            parts = line.split()
            current_line = int(parts[2].split(',')[0].replace('+', ''))
        elif line.startswith('+') and not line.startswith('+++'):
            changes.append((current_line, '', line[1:]))
            current_line += 1
        elif line.startswith('-') and not line.startswith('---'):
            changes.append((current_line, line[1:], ''))
        else:
            current_line += 1
    
    return changes

def review_changes(
    original_code: str,
    modified_code: str,
    change_description: str
) -> List[ReviewIssue]:
    """
    Review only the changed portions of code.
    """
    changes = extract_changed_sections(original_code, modified_code)
    
    if not changes:
        return []
    
    client = anthropic.Anthropic()
    
    changes_summary = "\n".join([
        f"Line {line}: '{old}' -> '{new}'"
        for line, old, new in changes[:20]  # Limit context
    ])
    
    prompt = f"""Review these code changes:

Change Description: {change_description}

Changes:
{changes_summary}

Full Modified Code:
```python
{modified_code}
```

Focus on:
1. Does the change accomplish the stated goal?
2. Are there new bugs introduced?
3. Are edge cases handled?
4. Is error handling appropriate?
5. Are there better approaches?

Return JSON array of issues."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    import json
    issues_data = json.loads(response.content[0].text)
    
    return [
        ReviewIssue(
            level=IssueLevel.SEMANTIC,
            severity=issue['severity'],
            line=issue['line'],
            message=issue['message'],
            suggestion=issue['suggestion'],
            auto_fixable=False
        )
        for issue in issues_data
    ]
```

**Practical Implications:**

Differential analysis reduces review scope by 80-95%, dramatically lowering costs and latency. Instead of analyzing a 500-line file, you analyze 20 changed lines plus surrounding context.

**Trade-offs:**

Context boundaries matter. A change might be correct locally but break assumptions elsewhere in the file. Include 5-10 lines of context before/after changes to catch these issues.

### 4. Automated Suggestion Generation

Beyond identifying problems, LLMs can generate concrete fixes. These range from simple corrections (typo fixes, consistent formatting) to complex refactorings (extracting functions, improving error handling).

**Implementation:**

```python
def generate_fix_suggestion(
    code: str,
    issue: ReviewIssue,
    context_lines: int = 10
) -> str:
    """
    Generate specific code fix for an identified issue.
    
    Args:
        code: Full code content
        issue: The identified problem
        context_lines: Lines of context around issue
    
    Returns:
        Suggested fixed code
    """
    lines = code.splitlines()
    start = max(0, issue.line - context_lines)
    end = min(len(lines), issue.line + context_lines)
    context = "\n".join(lines[start:end])
    
    client = anthropic.Anthropic()
    
    prompt = f"""Fix this code issue:

Problem at line {issue.line}:
{issue.message}

Code context:
```python
{context}
```

Generate ONLY the fixed code section (lines {start}-{end}).
Preserve formatting and style.
Add comments explaining the fix.

Return only the code, no explanations."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract code from response
    fixed_code = response.content[0].text
    if "```python" in fixed_code:
        fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
    
    return fixed_code
```

**Real Constraints:**

Automated fixes require validation. Never auto-apply suggestions without human review or comprehensive test coverage. Treat suggestions as starting points, not definitive solutions.

### 5. Review Consistency and Learning

LLMs can learn from past review feedback, maintaining consistency across team members and evolving style guidelines.

**Pattern Storage:**

```python
from datetime import datetime
from typing import Dict, List
import json

class ReviewPatternDatabase:
    """
    Store and retrieve common review patterns and decisions.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.patterns: Dict[str, List[Dict]] = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, List[Dict]]:
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"approved": [], "rejected": [], "style_preferences": []}
    
    def add_review_decision(
        self,
        code_pattern: str,
        decision: str,
        reasoning: str,
        reviewer: str
    ):
        """
        Record a review decision for future consistency.
        """
        entry = {
            "pattern": code_pattern,
            "reasoning": reasoning,
            "reviewer": reviewer,
            "timestamp": datetime.now().isoformat()
        }
        
        self.patterns[decision].append(entry)
        self._save_patterns()
    
    def _save_patterns(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def get_relevant_patterns(
        self,
        code_snippet: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve similar past review decisions.
        """
        client = anthropic.Anthropic()
        
        all_patterns = (
            self.patterns["approved"] + 
            self.patterns["rejected"] + 
            self.patterns["style_preferences"]
        )
        
        if not all_patterns:
            return []
        
        patterns_text = "\n\n".join([
            f"Pattern: {p['pattern']}\nDecision: {p.get('decision', 'N/A')}\n"
            f"Reasoning: {p['reasoning']}"
            for p in all_patterns[:20]  # Limit context
        ])
        
        prompt = f"""Given this code:
```python
{code_snippet}
```

And these past review patterns:
{patterns_text}

Return the {top_k} most relevant patterns as JSON array with indices."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            relevant_indices = json.loads(response.content[0].text)
            return [all_patterns[i] for i in relevant_indices[:top_k]]
        except (json.JSONDecodeError, IndexError):
            return all_patterns[:top_k]

def review_with_consistency(
    code: str,
    pattern_db: ReviewPatternDatabase
) -> List[ReviewIssue]:
    """
    Review code using established team patterns and preferences.
    """
    relevant_patterns = pattern_db.get_relevant_patterns(code)
    
    patterns_context = "\n".join([
        f"- {p['reasoning']}" for p in relevant_patterns
    ])
    
    client = anthropic.Anthropic()
    
    prompt = f"""Review this code following our team's established patterns:

Team Patterns and Preferences:
{patterns_context}

Code to Review:
```python
{code}
```

Ensure consistency with past decisions.
Return JSON array of issues."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    issues_data = json.loads(response.content[0].text)
    
    return [
        ReviewIssue(
            level=IssueLevel.PRAGMATIC,
            severity=issue['severity'],