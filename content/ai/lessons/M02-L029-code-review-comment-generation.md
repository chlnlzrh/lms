# Code Review Comment Generation with LLMs

## Core Concepts

Code review comment generation is the automated process of using language models to analyze code changes and produce contextual, actionable feedback—similar to what an experienced engineer would provide during manual review. Unlike static analysis tools that pattern-match against predefined rules, LLM-based review leverages learned patterns from millions of code examples to understand intent, identify subtle issues, and suggest improvements in natural language.

### Traditional vs. Modern Approach

```python
# Traditional: Rule-based static analysis
class TraditionalReviewer:
    def review(self, code: str) -> list[str]:
        issues = []
        
        # Fixed pattern matching
        if "TODO" in code:
            issues.append("Contains TODO comment")
        if len(code.split('\n')) > 50:
            issues.append("Function too long")
        if "except:" in code:
            issues.append("Bare except clause")
            
        return issues

# Output: Generic, inflexible warnings
# ["Function too long", "Bare except clause"]
```

```python
# Modern: LLM-based contextual analysis
from typing import TypedDict
import anthropic

class ReviewComment(TypedDict):
    line: int
    severity: str
    comment: str
    suggestion: str

class LLMReviewer:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def review(self, code: str, context: str = "") -> list[ReviewComment]:
        prompt = f"""Analyze this code change and provide specific review comments.
        
Context: {context}

Code:
{code}

For each issue, provide:
1. Line number
2. Severity (critical/important/suggestion)
3. Clear explanation of the issue
4. Concrete code suggestion if applicable

Format as JSON array."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse and return structured comments
        return self._parse_comments(response.content[0].text)

# Output: Context-aware, actionable feedback
# [{"line": 23, "severity": "important", 
#   "comment": "This exception handler catches all exceptions including KeyboardInterrupt...",
#   "suggestion": "except ValueError: # Only catch expected errors"}]
```

The difference is profound: traditional tools tell you *what* violates rules, while LLM-based reviewers explain *why* code might be problematic in its specific context and *how* to improve it.

### Key Insights That Change Engineering Thinking

**Context matters more than rules.** A bare `except:` clause might be acceptable in a script that needs to log all failures and continue, but dangerous in a library. LLMs can understand this distinction when given appropriate context.

**Review quality scales with prompt engineering, not model size alone.** A well-prompted smaller model often outperforms a poorly-prompted larger one. The architectural pattern—how you structure the review request—matters more than raw capability.

**Generated comments need validation layers.** LLMs can hallucinate issues or miss critical problems. Production systems require confidence scoring, hallucination detection, and fallback mechanisms.

### Why This Matters Now

Engineering teams face a review bottleneck: senior engineers spend 20-30% of their time reviewing code, yet 40-60% of review comments address issues that could be automatically detected with sufficient context awareness. LLM-based review doesn't replace human judgment—it handles the first pass, allowing humans to focus on architecture, design decisions, and business logic rather than catching missing error handling or inconsistent naming.

## Technical Components

### 1. Prompt Architecture for Code Review

The prompt is your review policy encoded as instructions. Poor prompts produce generic, unhelpful comments; well-structured prompts produce targeted, actionable feedback.

**Technical Explanation:**

A review prompt must balance specificity (what to look for) with flexibility (allowing the model to identify unexpected issues). The structure typically includes:

- **Role definition**: Establishes the model's expertise level and perspective
- **Context injection**: Provides codebase conventions, recent changes, or domain knowledge
- **Review criteria**: Explicit list of what matters for this review
- - **Output format**: Structured format for parsing and presentation

**Practical Implementation:**

```python
from dataclasses import dataclass
from enum import Enum

class ReviewFocus(Enum):
    SECURITY = "security vulnerabilities and data handling"
    PERFORMANCE = "performance implications and resource usage"
    MAINTAINABILITY = "code clarity and long-term maintainability"
    CORRECTNESS = "logical errors and edge cases"
    ALL = "all aspects of code quality"

@dataclass
class ReviewConfig:
    focus: ReviewFocus
    language: str
    conventions: dict[str, str]
    max_comments: int = 10

def build_review_prompt(
    code_diff: str,
    config: ReviewConfig,
    file_context: str = ""
) -> str:
    conventions_str = "\n".join(
        f"- {k}: {v}" for k, v in config.conventions.items()
    )
    
    prompt = f"""You are an experienced {config.language} engineer conducting a code review.

Review Focus: {config.focus.value}

Project Conventions:
{conventions_str}

File Context:
{file_context}

Code Changes:
```
{code_diff}
```

Provide up to {config.max_comments} specific, actionable review comments. For each:
1. Quote the relevant code line
2. Explain the issue clearly
3. Suggest a concrete improvement
4. Rate severity: CRITICAL, IMPORTANT, or MINOR

Focus on issues that matter. Don't comment on style if it follows conventions."""

    return prompt

# Usage example
config = ReviewConfig(
    focus=ReviewFocus.SECURITY,
    language="Python",
    conventions={
        "error_handling": "Use specific exception types",
        "secrets": "Never hardcode credentials",
        "input_validation": "Validate all external inputs"
    }
)

code_diff = """
+ def process_user_data(user_id):
+     api_key = "sk-12345"  # API key for external service
+     response = requests.get(f"https://api.example.com/users/{user_id}")
+     return response.json()
"""

prompt = build_review_prompt(code_diff, config)
```

**Real Constraints:**

- **Token limits**: A full file context + diff + instructions can easily exceed 8K tokens. You'll need chunking strategies for large changes.
- **Prompt sensitivity**: Small wording changes can significantly affect output quality. "Find bugs" produces different results than "Identify potential runtime errors."
- **Cost scaling**: Each review costs tokens. A 500-line PR might cost $0.10-$0.50 depending on model and context size.

### 2. Diff Processing and Context Extraction

Raw git diffs are noisy—they include metadata, unchanged context lines, and formatting that confuses models. Effective review requires clean, focused code presentation.

**Technical Explanation:**

Git diffs show added (+), removed (-), and context lines. For review, you need:
- Only changed lines for focused analysis
- Enough surrounding context to understand intent
- File-level metadata (language, purpose, dependencies)

**Practical Implementation:**

```python
import re
from typing import NamedTuple

class CodeChange(NamedTuple):
    file_path: str
    added_lines: list[tuple[int, str]]  # (line_num, content)
    removed_lines: list[tuple[int, str]]
    context_before: str
    context_after: str

def parse_unified_diff(diff_text: str) -> list[CodeChange]:
    """Parse unified diff format into structured changes."""
    changes = []
    current_file = None
    added_lines = []
    removed_lines = []
    line_num = 0
    
    for line in diff_text.split('\n'):
        # File header
        if line.startswith('+++'):
            file_path = line.split('\t')[0][6:]  # Remove '+++ b/'
            current_file = file_path
            continue
            
        # Hunk header: @@ -start,count +start,count @@
        if line.startswith('@@'):
            match = re.match(r'@@ -\d+,?\d* \+(\d+),?\d* @@', line)
            if match:
                line_num = int(match.group(1))
            continue
        
        # Changed lines
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append((line_num, line[1:]))
            line_num += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed_lines.append((line_num, line[1:]))
        else:
            line_num += 1
    
    if current_file:
        changes.append(CodeChange(
            file_path=current_file,
            added_lines=added_lines,
            removed_lines=removed_lines,
            context_before="",
            context_after=""
        ))
    
    return changes

def extract_reviewable_chunks(
    change: CodeChange,
    max_chunk_lines: int = 50
) -> list[str]:
    """Split large changes into reviewable chunks."""
    chunks = []
    current_chunk = []
    
    for line_num, content in change.added_lines:
        current_chunk.append(f"{line_num}: {content}")
        
        if len(current_chunk) >= max_chunk_lines:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

# Example usage
diff = """
--- a/auth.py
+++ b/auth.py
@@ -15,3 +15,8 @@ def login(username, password):
     return user
+
+def validate_token(token):
+    # Validate JWT token
+    decoded = jwt.decode(token, verify=False)
+    return decoded
"""

changes = parse_unified_diff(diff)
for change in changes:
    chunks = extract_reviewable_chunks(change)
    print(f"File: {change.file_path}, Chunks: {len(chunks)}")
```

**Real Constraints:**

- **Context window trade-offs**: More context improves review quality but increases cost and latency. Optimal context is typically 10-20 lines before/after changes.
- **Binary and generated files**: Diffs may include non-reviewable content (images, compiled assets, lock files). Filter these out before sending to the model.
- **Renamed files**: Git shows renames as full deletions + additions. Detect renames to avoid reviewing unchanged code.

### 3. Comment Filtering and Ranking

LLMs generate varying quality comments—some insightful, others obvious or incorrect. Production systems need automated quality filtering.

**Technical Explanation:**

Not all generated comments are useful. Filtering criteria include:
- **Confidence scoring**: Does the model seem certain or hedging?
- **Actionability**: Does the comment suggest a concrete improvement?
- **Severity assessment**: Is this critical, important, or minor?
- **Duplication detection**: Are multiple comments addressing the same issue?

**Practical Implementation:**

```python
from typing import Optional
import re

@dataclass
class ReviewCommentRaw:
    line: int
    text: str
    severity: str
    suggestion: Optional[str] = None

@dataclass
class ScoredComment:
    comment: ReviewCommentRaw
    confidence: float
    actionability: float
    final_score: float

class CommentFilter:
    # Patterns indicating low confidence
    LOW_CONFIDENCE_PHRASES = [
        "might want to consider",
        "possibly",
        "perhaps",
        "could potentially",
        "may want to"
    ]
    
    # Patterns indicating high actionability
    HIGH_ACTIONABILITY_PATTERNS = [
        r"change .* to .*",
        r"replace .* with .*",
        r"add .*",
        r"remove .*",
        r"use .* instead"
    ]
    
    def score_confidence(self, comment: ReviewCommentRaw) -> float:
        """Score 0-1 based on language certainty."""
        text_lower = comment.text.lower()
        
        # Start with base confidence
        score = 0.7
        
        # Reduce for hedging language
        for phrase in self.LOW_CONFIDENCE_PHRASES:
            if phrase in text_lower:
                score -= 0.15
        
        # Increase for specific references
        if re.search(r'line \d+', text_lower):
            score += 0.1
        if comment.suggestion is not None:
            score += 0.2
            
        return max(0.0, min(1.0, score))
    
    def score_actionability(self, comment: ReviewCommentRaw) -> float:
        """Score 0-1 based on how actionable the comment is."""
        if comment.suggestion:
            return 1.0
        
        score = 0.3  # Base score for explanation-only comments
        
        # Check for actionable patterns
        for pattern in self.HIGH_ACTIONABILITY_PATTERNS:
            if re.search(pattern, comment.text, re.IGNORECASE):
                score += 0.3
                break
        
        # Check for security/correctness keywords
        critical_keywords = ['vulnerability', 'bug', 'error', 'crash', 'leak']
        if any(kw in comment.text.lower() for kw in critical_keywords):
            score += 0.2
        
        return min(1.0, score)
    
    def filter_and_rank(
        self,
        comments: list[ReviewCommentRaw],
        min_score: float = 0.5
    ) -> list[ScoredComment]:
        """Filter and rank comments by quality."""
        scored = []
        
        for comment in comments:
            confidence = self.score_confidence(comment)
            actionability = self.score_actionability(comment)
            
            # Weighted final score
            final_score = (confidence * 0.4) + (actionability * 0.6)
            
            if final_score >= min_score:
                scored.append(ScoredComment(
                    comment=comment,
                    confidence=confidence,
                    actionability=actionability,
                    final_score=final_score
                ))
        
        # Sort by score descending
        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored

# Example usage
raw_comments = [
    ReviewCommentRaw(
        line=23,
        text="This might want to consider using a try-except block",
        severity="MINOR"
    ),
    ReviewCommentRaw(
        line=45,
        text="SQL injection vulnerability on line 45. User input is directly interpolated.",
        severity="CRITICAL",
        suggestion="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
    ),
    ReviewCommentRaw(
        line=67,
        text="Consider refactoring this function",
        severity="MINOR"
    )
]

filter = CommentFilter()
scored = filter.filter_and_rank(raw_comments, min_score=0.5)

for sc in scored:
    print(f"Score: {sc.final_score:.2f} - {sc.comment.text[:60]}...")
```

**Real Constraints:**

- **False positives vs. false negatives**: Aggressive filtering reduces noise but may discard valid edge-case insights. Tune thresholds based on team tolerance.
- **Model-specific patterns**: Different models hedge differently. GPT models often use "consider", while Claude uses "might want to". Adjust patterns accordingly.
- **Severity calibration**: Model-assigned severity may not match team standards. Consider re-scoring based on issue type rather than trusting model output.

### 4. Integration Patterns

Code review systems must integrate with version control, notification systems, and developer workflows without creating friction.

**Technical Explanation:**

Effective integration requires:
- **Webhook handling**: Triggered on PR creation/update
- **Async processing**: Review may take 10-60 seconds; don't block PR workflow
- **Comment threading**: Link generated comments to specific code lines in the PR UI
- **Human oversight**: Flag generated comments as automated; allow developers to dismiss

**Practical Implementation:**

```python
import asyncio
import hashlib
from datetime import datetime