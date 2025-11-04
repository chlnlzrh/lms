# GitHub Team Collaboration for AI Development

## Core Concepts

### Technical Definition

GitHub Teams is a repository access control and collaboration framework that maps organizational structure to code ownership and review workflows. It provides hierarchical permission management, automated notification routing, and code review assignment through team-based policies rather than individual-level access grants.

For AI/LLM development specifically, teams enable critical collaboration patterns: dataset version control, prompt library sharing, model training pipeline coordination, and distributed evaluation workflows where multiple engineers need synchronized access to rapidly evolving codebases.

### Engineering Analogy: Individual vs. Team-Based Access Control

**Traditional Individual Access Approach:**

```python
# Manual permission management - doesn't scale
class RepositoryAccess:
    def __init__(self):
        self.permissions = {
            "user_alice": ["repo_model_training", "repo_datasets"],
            "user_bob": ["repo_model_training", "repo_evaluation"],
            "user_carol": ["repo_datasets", "repo_evaluation"],
            "user_david": ["repo_model_training"],
            # Adding 10 new ML engineers means 10+ lines of changes
            # Removing someone requires finding all their permissions
            # No clear ownership - who's responsible for what?
        }
    
    def grant_access(self, user: str, repo: str):
        if user in self.permissions:
            self.permissions[user].append(repo)
        else:
            self.permissions[user] = [repo]
    
    def check_access(self, user: str, repo: str) -> bool:
        return repo in self.permissions.get(user, [])

# Problem: When Alice leaves, you must manually audit every repo
# Problem: No clear signal that Bob and David work on the same component
# Problem: Code review assignments require manual tracking
```

**Modern Team-Based Approach:**

```python
# Team-based access - maps to organizational reality
class TeamBasedAccess:
    def __init__(self):
        self.teams = {
            "ml-training-team": {
                "members": ["alice", "david", "eve"],
                "repositories": ["model_training", "experiment_tracking"],
                "role": "maintain",  # Can merge PRs, manage issues
                "code_review_assignment": "round_robin"
            },
            "data-team": {
                "members": ["alice", "carol", "frank"],
                "repositories": ["datasets", "data_pipelines"],
                "role": "write",  # Can push but not merge without review
                "code_review_assignment": "load_balance"
            },
            "evaluation-team": {
                "members": ["bob", "carol", "grace"],
                "repositories": ["evaluation", "benchmarks"],
                "role": "maintain",
                "code_review_assignment": "code_owners"
            }
        }
    
    def check_access(self, user: str, repo: str) -> tuple[bool, str]:
        for team_name, team_config in self.teams.items():
            if user in team_config["members"]:
                if repo in team_config["repositories"]:
                    return True, team_config["role"]
        return False, None
    
    def get_reviewers(self, repo: str, author: str) -> list[str]:
        """Automatically assign reviewers from appropriate team"""
        for team_config in self.teams.values():
            if repo in team_config["repositories"]:
                # Exclude author, return team members for review
                return [m for m in team_config["members"] if m != author]
        return []

# Advantages:
# - Alice's removal: Update one team membership list
# - Clear ownership: "data-team" owns dataset repos
# - Automatic review routing: PR to 'datasets' auto-notifies data-team
# - Organizational clarity: Teams reflect actual working groups
```

### Key Insights for AI Engineers

**1. Permission Inheritance Prevents Configuration Drift**

In AI projects, you'll work across 5-15 repositories: model training code, dataset repos, evaluation harnesses, deployment configs, prompt libraries. Individual permissions create N×M complexity (N users × M repos). Teams create N+M complexity (N users in teams + M repos assigned to teams).

**2. Code Ownership Becomes Explicit**

With AI systems, accountability matters: whose prompt engineering changes degraded model performance? Which team owns the data preprocessing pipeline? Team-based CODEOWNERS files make this machine-readable, enabling automatic review assignment and clear escalation paths.

**3. Notification Routing Reduces Context Switching**

LLM development generates high PR volume (prompt iterations, evaluation tweaks, dataset updates). Team mentions (@data-team) route notifications to the right engineers, while individual mentions create "notification bankruptcy" where engineers ignore 90% of alerts.

### Why This Matters NOW

AI development is inherently collaborative. You're not building isolated services—you're orchestrating model training (ML team), data pipelines (data team), evaluation frameworks (research team), and production deployment (platform team). 

Without structured team collaboration:
- Data scientists block on waiting for dataset access approvals
- Prompt changes ship without review from engineers who understand production constraints  
- Security issues in training data go unnoticed because no team has clear ownership
- Knowledge silos form: only one engineer understands the evaluation pipeline

GitHub Teams solves this by making collaboration structure explicit and automated.

## Technical Components

### 1. Team Hierarchies and Permission Inheritance

**Technical Explanation:**

Teams support parent-child relationships where child teams inherit parent permissions plus their own. This maps to organizational hierarchies: an "AI Engineering" parent team might have "Model Training" and "Data Engineering" children. Children inherit base repository access, then receive specialized permissions.

**Practical Implications:**

```python
# Modeling team structure in code for understanding
class TeamHierarchy:
    def __init__(self):
        self.structure = {
            "ai-engineering": {  # Parent team
                "members": ["tech_lead_alice", "senior_bob"],
                "base_repos": ["shared_utils", "documentation"],
                "permission": "read",
                "children": ["model-training", "data-engineering"]
            },
            "model-training": {  # Child team
                "members": ["ml_engineer_carol", "ml_engineer_david"],
                "base_repos": ["training_pipeline", "model_registry"],
                "permission": "maintain",
                "parent": "ai-engineering"
            },
            "data-engineering": {  # Child team
                "members": ["data_engineer_eve", "data_engineer_frank"],
                "base_repos": ["datasets", "etl_pipelines"],
                "permission": "maintain",
                "parent": "ai-engineering"
            }
        }
    
    def effective_permissions(self, user: str, repo: str) -> str:
        """Calculate effective permissions including inheritance"""
        permissions = []
        
        # Find all teams user belongs to
        for team_name, config in self.structure.items():
            if user in config.get("members", []):
                # Check direct permissions
                if repo in config.get("base_repos", []):
                    permissions.append(config["permission"])
                
                # Check parent permissions
                if "parent" in config:
                    parent = self.structure[config["parent"]]
                    if repo in parent.get("base_repos", []):
                        permissions.append(parent["permission"])
        
        # Permission hierarchy: maintain > write > read
        if "maintain" in permissions:
            return "maintain"
        elif "write" in permissions:
            return "write"
        elif "read" in permissions:
            return "read"
        return "none"

# Example usage
hierarchy = TeamHierarchy()
print(hierarchy.effective_permissions("ml_engineer_carol", "shared_utils"))
# Output: "read" (inherited from parent ai-engineering team)
print(hierarchy.effective_permissions("ml_engineer_carol", "training_pipeline"))
# Output: "maintain" (direct permission from model-training team)
```

**Real Constraints:**

- Maximum nesting depth is typically 1 parent + child (no grandchildren)
- Permission inheritance is additive (can't remove parent permissions in child)
- Changes to parent team propagate to all children, creating potential blast radius

**Concrete Example:**

In a multimodal LLM project:
- Parent "multimodal-ai" team: All members read access to shared prompt library
- Child "vision-team": Maintain access to image processing repos
- Child "text-team": Maintain access to text embedding repos
- Both children inherit prompt library access without duplicate configuration

### 2. CODEOWNERS and Automatic Review Assignment

**Technical Explanation:**

CODEOWNERS is a file (`.github/CODEOWNERS`) that maps file paths to teams/individuals using gitignore-style patterns. When a PR modifies files matching a pattern, GitHub automatically requests reviews from specified owners. This creates enforceable code ownership policies.

**Practical Implications:**

```python
# Example CODEOWNERS file content (as string for demonstration)
CODEOWNERS_CONTENT = """
# Global owners - get notified of all changes
* @ai-engineering/tech-leads

# Model training pipeline
/training/** @ai-engineering/model-training
/training/config/*.yaml @ai-engineering/model-training @ai-engineering/ml-ops

# Dataset management - requires data team + security review
/datasets/** @ai-engineering/data-engineering
/datasets/pii/** @ai-engineering/data-engineering @security-team

# Evaluation and benchmarks
/evaluation/** @ai-engineering/evaluation-team
/evaluation/metrics/*.py @ai-engineering/evaluation-team @ai-engineering/research

# Production deployment configs - require platform team
/deployment/** @platform-team
/deployment/kubernetes/** @platform-team @ai-engineering/ml-ops

# Prompt engineering templates
/prompts/** @ai-engineering/prompt-engineering
/prompts/production/*.json @ai-engineering/prompt-engineering @ai-engineering/safety-team
"""

import re
from typing import List, Set

class CodeOwnersParser:
    def __init__(self, codeowners_content: str):
        self.rules = []
        for line in codeowners_content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    pattern = parts[0]
                    owners = parts[1:]
                    self.rules.append((pattern, owners))
    
    def get_owners(self, file_path: str) -> Set[str]:
        """Get all owners for a given file path (most specific wins)"""
        owners = set()
        for pattern, pattern_owners in self.rules:
            if self._matches_pattern(pattern, file_path):
                owners.update(pattern_owners)
        return owners
    
    def _matches_pattern(self, pattern: str, path: str) -> bool:
        """Simplified pattern matching (real implementation uses gitignore)"""
        if pattern == "*":
            return True
        if pattern.endswith("**"):
            return path.startswith(pattern[:-2])
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])
        return path.startswith(pattern)

# Usage example
parser = CodeOwnersParser(CODEOWNERS_CONTENT)

# Example: PR modifying prompt templates
changed_files = [
    "/prompts/production/summarization.json",
    "/prompts/development/test_prompt.json"
]

for file in changed_files:
    owners = parser.get_owners(file)
    print(f"{file} requires review from: {owners}")

# Output:
# /prompts/production/summarization.json requires review from:
#   {'@ai-engineering/tech-leads', '@ai-engineering/prompt-engineering', 
#    '@ai-engineering/safety-team'}
# /prompts/development/test_prompt.json requires review from:
#   {'@ai-engineering/tech-leads', '@ai-engineering/prompt-engineering'}
```

**Real Constraints:**

- CODEOWNERS file must be in root, `.github/`, or `docs/` directory
- Order matters: last matching pattern takes precedence for conflicts
- Protected branches can enforce CODEOWNERS approval (blocks merge without owner review)
- Large files with many owners can create review bottlenecks

**Concrete Example:**

When a data scientist opens a PR adding PII-containing training data (`/datasets/pii/customer_conversations.jsonl`), the CODEOWNERS rule automatically:
1. Requests review from @data-engineering team (data quality check)
2. Requests review from @security-team (PII compliance verification)
3. Blocks merge until both teams approve (if branch protection enabled)

### 3. Team Mentions and Notification Routing

**Technical Explanation:**

Team mentions (`@organization/team-name`) in issues, PRs, and discussions trigger notifications to all team members. Combined with notification filtering rules, this creates topic-based routing where engineers subscribe to relevant team channels rather than monitoring individual repositories.

**Practical Implications:**

```python
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class NotificationEvent:
    event_type: str  # "mention", "review_request", "issue_assignment"
    source: str
    teams_mentioned: List[str]
    repository: str

class NotificationRouter:
    def __init__(self):
        self.team_subscriptions = {
            "alice": ["@ai-engineering/model-training", "@ai-engineering/tech-leads"],
            "bob": ["@ai-engineering/evaluation-team"],
            "carol": ["@ai-engineering/model-training", "@ai-engineering/data-engineering"],
        }
        
        self.notification_preferences = {
            "alice": {
                "mention": "real_time",
                "review_request": "real_time", 
                "issue_assignment": "daily_digest"
            },
            "bob": {
                "mention": "real_time",
                "review_request": "hourly_digest",
                "issue_assignment": "weekly_digest"
            }
        }
    
    def route_notification(self, event: NotificationEvent) -> Dict[str, str]:
        """Determine who gets notified and how"""
        notifications = {}
        
        for user, subscribed_teams in self.team_subscriptions.items():
            # Check if user's team was mentioned
            if any(team in event.teams_mentioned for team in subscribed_teams):
                delivery_method = self.notification_preferences[user].get(
                    event.event_type, "daily_digest"
                )
                notifications[user] = delivery_method
        
        return notifications

# Example: PR review request scenario
router = NotificationRouter()

# Scenario 1: Quick model fix needs immediate review
urgent_pr = NotificationEvent(
    event_type="review_request",
    source="PR #456: Fix training loss calculation",
    teams_mentioned=["@ai-engineering/model-training"],
    repository="training_pipeline"
)

print("Urgent PR notifications:")
print(router.route_notification(urgent_pr))
# Output: {'alice': 'real_time', 'carol': 'real_time'}
# Both model-training team members get immediate notifications

# Scenario 2: Issue about evaluation metrics
evaluation_issue = NotificationEvent(
    event_type="mention",
    source="Issue #789: Evaluation metric clarification",
    teams_mentioned=["@ai-engineering/evaluation-team"],
    repository="evaluation"
)

print("\nEvaluation issue notifications:")
print(router.route_notification(evaluation_issue))
# Output: {'bob': 'real_time'}
# Only evaluation team member gets notified
```

**Real Constraints:**

- Team mentions notify ALL members (can't selectively notify subset)
- Notification fatigue: teams >15 members often ignore mentions
- Can't mention teams from other organizations (even in public repos)
- Rate limits: Excessive mentions can trigger spam detection

**Concrete Example:**

In a prompt engineering workflow:
- Developer opens PR with experimental prompt: mentions `@ai-engineering/prompt-engineering` for review
- Prompt fails safety check: Developer comments mentioning `@ai-engineering/safety-team` 
- Safety team member reviews, approves with modifications
- Final merge notification goes to both teams automatically

Result: Right experts notified at right time without manual CC-ing or Slack messages.

### 4. Team Discussions and Knowledge Sharing

**Technical Explanation:**

GitHub Discussions with team-based categories provides structured, searchable knowledge sharing tied to repositories. Unlike chat (ephemeral, unsearchable) or wikis (disconnected from code), discussions live alongside code with built-in markdown, code blocks, and cross-referencing to issues/PRs.

**Practical Implications:**

```python
from typing import List, Optional
from datetime import datetime

class Discussion:
    def __init__(self, title: str, category: str, team: str, body: str):