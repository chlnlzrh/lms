# Change Log & Release Notes Generation with LLMs

## Core Concepts

Change logs and release notes are structured technical documentation that communicate software changes to users and stakeholders. Traditionally, engineers manually curate git commit messages, ticket references, and code diffs into coherent narratives—a time-consuming process prone to inconsistency and omission.

LLMs transform this from a documentation chore into an automated synthesis task. By processing raw development artifacts (commits, pull requests, issue trackers), LLMs can extract semantic meaning, categorize changes, and generate audience-appropriate narratives in seconds rather than hours.

### Traditional vs. Modern Approach

**Traditional approach:**

```python
# Manual release note compilation
def generate_release_notes_manual(version: str, commits: list[str]) -> str:
    """Engineer manually reviews commits and writes notes"""
    notes = f"## Version {version}\n\n"
    
    # Engineer must:
    # 1. Read through all commits
    # 2. Group by type (features, fixes, breaking changes)
    # 3. Translate technical jargon to user language
    # 4. Cross-reference with issue tracker
    # 5. Format consistently
    
    notes += "### Features\n"
    notes += "- [Manually written feature description]\n"
    notes += "- [Another manual entry]\n\n"
    notes += "### Bug Fixes\n"
    notes += "- [Manually written fix description]\n"
    
    return notes

# Result: 30-60 minutes per release, inconsistent quality
```

**LLM-powered approach:**

```python
from typing import List, Dict
import anthropic
import os

def generate_release_notes_llm(
    version: str,
    commits: List[Dict[str, str]],
    audience: str = "technical"
) -> str:
    """Automated release note generation from commit history"""
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Prepare commit data
    commit_summary = "\n".join([
        f"- {c['hash'][:8]}: {c['message']} (by {c['author']})"
        for c in commits
    ])
    
    prompt = f"""Analyze these git commits and generate release notes for version {version}.

Commits:
{commit_summary}

Requirements:
- Group changes into: Features, Bug Fixes, Performance, Breaking Changes, Documentation
- Use {audience} language (technical users understand API details, general users need plain language)
- Highlight breaking changes prominently
- Include relevant commit hashes for reference
- Format as markdown

Generate concise, accurate release notes."""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

# Result: 2-5 seconds per release, consistent structure
```

### Key Engineering Insights

**1. Semantic vs. Syntactic Processing**: Traditional tooling can parse commit formats (conventional commits, regex patterns) but cannot understand semantic relationships. An LLM recognizes that commits touching authentication code, error handling, and documentation all relate to a single security feature enhancement—even without explicit linking.

**2. Audience Translation is Context-Dependent**: The same code change requires different descriptions for different audiences. A database index optimization is "improved query performance for large datasets" for product teams but "added B-tree index on user_events.timestamp, 40% query speedup on date-range filters" for technical teams. LLMs excel at this contextual rewriting.

**3. Change Logs are Information Compression**: Raw development artifacts contain 10-100x more information than useful for release notes. The engineering challenge isn't generating text—it's deciding what to include, what to omit, and how to structure remaining information. LLMs provide learned heuristics about what matters for software changes.

### Why This Matters Now

Modern development produces overwhelming change velocity: multiple deploys daily, distributed teams, microservices with independent versioning. Manual curation doesn't scale. Meanwhile, compliance requirements (SOC2, ISO 27001), user expectations, and internal accountability demand comprehensive change documentation. LLMs bridge this gap between volume and quality requirements without proportionally increasing documentation team size.

## Technical Components

### 1. Input Data Structure & Preparation

LLMs require structured context about what changed, not just raw text. Quality input determines output quality more than prompt engineering.

**Technical Implementation:**

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import subprocess
import json

@dataclass
class Commit:
    hash: str
    message: str
    author: str
    timestamp: datetime
    files_changed: List[str]
    additions: int
    deletions: int
    
@dataclass
class PullRequest:
    number: int
    title: str
    description: str
    commits: List[Commit]
    labels: List[str]

def extract_git_commits(from_ref: str, to_ref: str) -> List[Commit]:
    """Extract structured commit data from git history"""
    
    git_log = subprocess.run(
        [
            "git", "log",
            f"{from_ref}..{to_ref}",
            "--pretty=format:%H|%s|%an|%at",
            "--numstat"
        ],
        capture_output=True,
        text=True,
        check=True
    )
    
    commits = []
    current_commit = None
    
    for line in git_log.stdout.split('\n'):
        if '|' in line:  # Commit metadata line
            parts = line.split('|')
            current_commit = Commit(
                hash=parts[0],
                message=parts[1],
                author=parts[2],
                timestamp=datetime.fromtimestamp(int(parts[3])),
                files_changed=[],
                additions=0,
                deletions=0
            )
            commits.append(current_commit)
        elif line.strip() and current_commit:  # File change line
            parts = line.split('\t')
            if len(parts) == 3:
                adds, dels, filename = parts
                current_commit.files_changed.append(filename)
                current_commit.additions += int(adds) if adds.isdigit() else 0
                current_commit.deletions += int(dels) if dels.isdigit() else 0
    
    return commits

def prepare_context_for_llm(
    commits: List[Commit],
    pull_requests: Optional[List[PullRequest]] = None,
    max_commits: int = 100
) -> str:
    """Structure commit data for optimal LLM processing"""
    
    # Filter out merge commits and version bumps (noise)
    significant_commits = [
        c for c in commits 
        if not c.message.startswith("Merge") 
        and not c.message.startswith("Bump version")
    ][:max_commits]
    
    context_parts = []
    
    # Group commits by conventional commit type
    grouped = {}
    for commit in significant_commits:
        prefix = commit.message.split(':')[0] if ':' in commit.message else 'other'
        grouped.setdefault(prefix, []).append(commit)
    
    for commit_type, commits_list in grouped.items():
        context_parts.append(f"\n## {commit_type.upper()} Changes:")
        for c in commits_list:
            # Include file context for better understanding
            file_context = f" (files: {', '.join(c.files_changed[:3])})" if c.files_changed else ""
            context_parts.append(
                f"- {c.hash[:8]}: {c.message}{file_context}"
            )
    
    # Add PR context if available
    if pull_requests:
        context_parts.append("\n## Pull Request Context:")
        for pr in pull_requests:
            labels_str = f"[{', '.join(pr.labels)}]" if pr.labels else ""
            context_parts.append(f"- PR #{pr.number} {labels_str}: {pr.title}")
            if pr.description:
                context_parts.append(f"  {pr.description[:200]}")
    
    return "\n".join(context_parts)
```

**Practical Implications:**

- **File context matters**: Including which files changed helps LLMs infer impact scope (API changes vs. internal refactoring)
- **Filtering noise**: Merge commits and automated version bumps dilute signal; remove them
- **Token budget management**: 100+ commits can exceed context windows; prioritize meaningful changes
- **Structured grouping**: Pre-grouping by conventional commit types improves LLM categorization accuracy by ~30%

**Trade-offs:**

- More context → better understanding but higher cost and slower processing
- Pre-filtering risks omitting relevant information
- Conventional commit format dependency (not all teams use it consistently)

### 2. Prompt Engineering for Structured Output

Release notes require consistent structure across releases. Prompt design must balance flexibility (handling varied changes) with rigidity (maintaining format).

**Technical Implementation:**

```python
from typing import Literal
from enum import Enum

class AudienceType(Enum):
    TECHNICAL = "technical"
    PRODUCT = "product"
    EXECUTIVE = "executive"
    END_USER = "end_user"

def build_release_notes_prompt(
    version: str,
    context: str,
    audience: AudienceType,
    include_breaking_changes: bool = True,
    tone: Literal["formal", "casual"] = "formal"
) -> str:
    """Generate audience-specific release notes prompt"""
    
    audience_guidance = {
        AudienceType.TECHNICAL: """
- Include API changes, function signatures, configuration updates
- Reference specific commit hashes for traceability
- Mention performance metrics (latency, throughput) when relevant
- Detail migration steps for breaking changes""",
        
        AudienceType.PRODUCT: """
- Focus on user-facing features and capabilities
- Explain business value and use cases
- Avoid implementation details unless relevant to usage
- Highlight UX improvements and new workflows""",
        
        AudienceType.EXECUTIVE: """
- Emphasize strategic impact and business outcomes
- Group changes by initiative or business objective
- Quantify improvements (cost reduction, efficiency gains)
- Minimize technical jargon""",
        
        AudienceType.END_USER: """
- Use plain language, no technical terminology
- Focus only on visible changes to user experience
- Include screenshots or visual descriptions when applicable
- Provide simple "what this means for you" explanations"""
    }
    
    breaking_section = """
### Breaking Changes Section
If any breaking changes exist, create a prominent section:
- Explain what breaks and why
- Provide migration path with code examples
- Estimate migration effort (time/complexity)
""" if include_breaking_changes else ""
    
    prompt = f"""You are generating release notes for version {version} of a software system.

## Change Context
{context}

## Output Requirements

Structure the release notes with these sections (omit if no relevant changes):

1. **Overview**: 2-3 sentence summary of the release theme
2. **Features**: New capabilities added
3. **Enhancements**: Improvements to existing functionality
4. **Bug Fixes**: Issues resolved
5. **Performance**: Speed/efficiency improvements
6. **Security**: Security-related changes
7. **Documentation**: Documentation updates
{breaking_section}
8. **Deprecations**: Features marked for removal

## Audience: {audience.value}
{audience_guidance[audience]}

## Tone: {tone}

## Constraints
- Maximum 800 words total
- Each change item: 1-2 sentences maximum
- Use bullet points, not prose paragraphs
- Be specific: cite metrics, features, APIs
- Avoid vague terms like "improved," "enhanced" without specifics

Generate the release notes now:"""
    
    return prompt

# Usage example
def generate_audience_specific_notes(
    version: str,
    commits: List[Commit],
    audiences: List[AudienceType]
) -> Dict[AudienceType, str]:
    """Generate release notes for multiple audiences"""
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    context = prepare_context_for_llm(commits)
    
    notes = {}
    for audience in audiences:
        prompt = build_release_notes_prompt(version, context, audience)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        notes[audience] = message.content[0].text
    
    return notes
```

**Practical Implications:**

- **Audience parameter dramatically changes output**: Same commits generate 4+ distinct documents for different stakeholders
- **Explicit constraints prevent bloat**: Without word limits, LLMs tend toward verbose descriptions
- **Structured sections improve scanability**: Engineers skip to "Breaking Changes", executives read "Overview"

**Trade-offs:**

- Rigid structure may force artificial categorization of ambiguous changes
- Multi-audience generation increases API costs linearly (4 audiences = 4 calls)
- Tone guidance is subjective; requires iteration to match organizational voice

### 3. Change Classification & Semantic Analysis

Not all commits are equal. LLMs can assess impact severity, categorize by domain, and identify related changes that should be grouped.

**Technical Implementation:**

```python
from typing import Dict, List
import anthropic
import json

@dataclass
class ClassifiedChange:
    category: str  # feature, bugfix, performance, security, etc.
    severity: str  # major, minor, patch
    domains: List[str]  # auth, database, api, ui, etc.
    breaking: bool
    user_visible: bool
    description: str
    related_commits: List[str]

def classify_changes(commits: List[Commit]) -> List[ClassifiedChange]:
    """Use LLM to classify and group related changes"""
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Batch commits for efficient classification
    commit_batch = [
        {"hash": c.hash[:8], "message": c.message, "files": c.files_changed[:5]}
        for c in commits[:50]  # Process in batches to stay under token limits
    ]
    
    prompt = f"""Analyze these git commits and classify each one. Return a JSON array.

Commits:
{json.dumps(commit_batch, indent=2)}

For each commit, determine:
1. **category**: feature, bugfix, performance, security, refactor, docs, test, chore
2. **severity**: major (new capability/breaking change), minor (enhancement), patch (fix/tweak)
3. **domains**: Which system areas are affected (e.g., ["authentication", "database", "api"])
4. **breaking**: true if this breaks backward compatibility
5. **user_visible**: true if end users will notice this change
6. **description**: One-sentence summary in user-friendly language
7. **related_to**: List of other commit hashes this logically groups with

Return ONLY a JSON array with this structure:
[
  {{
    "hash": "abc12345",
    "category": "feature",
    "severity": "minor",
    "domains": ["api", "authentication"],
    "breaking": false,
    "user_visible": true,
    "description": "Added OAuth2 token refresh endpoint",
    "related_to": []
  }}
]"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse JSON response
    response_text = message.content[0].text
    # Extract JSON from markdown code blocks if present
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    
    classifications = json.loads(response_text.strip())
    
    # Convert to ClassifiedChange objects
    changes = []
    for item in classifications:
        changes.append(ClassifiedChange(
            category=item['category'],
            severity=item['severity'],
            domains=item['domains'],
            breaking=item['breaking'],
            user_visible=item['user_visible'],
            description=item['description'],
            related_commits=item.get('related_to', [])
        ))
    
    return changes

def group_by_