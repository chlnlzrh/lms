# LLM-Assisted User Story Mapping

## Core Concepts

User story mapping is a product planning technique that arranges user stories spatially to visualize the user journey and prioritize features. Traditionally, this involves sticky notes on walls, whiteboards, and hours of facilitated workshops. LLM-assisted user story mapping transforms this from a synchronous, location-dependent activity into an iterative, collaborative process where AI handles the mechanical work of generation, organization, and refinement while humans focus on strategic decisions.

### Traditional vs. LLM-Assisted Approach

**Traditional Approach:**
```python
# Traditional user story mapping workflow (pseudocode representation)
def traditional_story_mapping():
    # Step 1: Gather stakeholders in a room (2-4 hours)
    stakeholders = schedule_workshop(min_attendees=5)
    
    # Step 2: Brainstorm activities (30-60 minutes)
    activities = []
    for stakeholder in stakeholders:
        activities.extend(stakeholder.suggest_activities())
    
    # Step 3: Manually organize on wall (60-90 minutes)
    organized_map = manually_arrange(activities)
    
    # Step 4: Generate stories for each activity (90-120 minutes)
    stories = []
    for activity in organized_map:
        stories.extend(team_brainstorm_stories(activity))
    
    # Step 5: Photograph board, transcribe to digital (30-60 minutes)
    digital_map = transcribe_physical_board(photograph)
    
    return digital_map  # Total: 5-8 hours, requires everyone present

# Result: Single iteration, difficult to revise, knowledge locked in one session
```

**LLM-Assisted Approach:**
```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import anthropic
import json

@dataclass
class UserStory:
    """Structured representation of a user story"""
    as_a: str
    i_want: str
    so_that: str
    acceptance_criteria: List[str]
    priority: str
    estimated_effort: str

@dataclass
class Activity:
    """High-level user activity in the journey"""
    name: str
    description: str
    user_stories: List[UserStory]
    sequence: int

def llm_assisted_story_mapping(
    product_description: str,
    target_users: List[str],
    constraints: Optional[Dict[str, str]] = None
) -> List[Activity]:
    """
    Generate user story map iteratively with LLM assistance
    Time: 30-60 minutes, asynchronous-friendly
    """
    client = anthropic.Anthropic()
    
    # Step 1: Generate backbone activities (5 minutes)
    activities_prompt = f"""Given this product: {product_description}
Target users: {', '.join(target_users)}
{f"Constraints: {constraints}" if constraints else ""}

Generate a user journey backbone with 5-8 high-level activities.
Return as JSON array with: name, description, sequence"""

    activities_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": activities_prompt}]
    )
    
    activities_data = json.loads(activities_response.content[0].text)
    
    # Step 2: Generate stories for each activity (parallel, 10-15 minutes total)
    activities = []
    for activity_data in activities_data:
        stories_prompt = f"""For the activity "{activity_data['name']}" 
in the context of: {product_description}

Generate 3-5 user stories with acceptance criteria.
Format as JSON array with: as_a, i_want, so_that, 
acceptance_criteria (array), priority (high/medium/low), 
estimated_effort (small/medium/large)"""

        stories_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            messages=[{"role": "user", "content": stories_prompt}]
        )
        
        stories_data = json.loads(stories_response.content[0].text)
        stories = [UserStory(**story) for story in stories_data]
        
        activities.append(Activity(
            name=activity_data['name'],
            description=activity_data['description'],
            sequence=activity_data['sequence'],
            user_stories=stories
        ))
    
    return activities  # Total: 30-60 minutes, can be paused/resumed

# Result: Multiple iterations possible, easy to refine, 
# knowledge captured in structured format
```

### Key Engineering Insights

**1. Structure Amplifies LLM Effectiveness:** Unlike free-form brainstorming, user story mapping has well-defined structure (activities → stories → acceptance criteria). This structure provides clear context windows for LLMs, making outputs more consistent and actionable.

**2. Human-LLM Task Decomposition:** The optimal pattern isn't "LLM does everything" or "LLM does nothing"—it's strategic decomposition. LLMs excel at generating variations, filling gaps, and formatting. Humans excel at strategic prioritization, domain-specific validation, and stakeholder alignment.

**3. Iterative Refinement Over Single-Shot Perfection:** Traditional workshops aim for "complete" output in one session. LLM-assisted approaches enable rapid iteration cycles: generate → validate → refine → expand. This matches modern development practices better than big-batch planning.

### Why This Matters Now

The convergence of three factors makes this timing critical:

1. **Remote work normalization:** Co-located workshop assumptions are broken. Teams need asynchronous-friendly planning methods.

2. **LLM context window expansion:** Modern models (100K+ tokens) can hold entire product contexts, enabling coherent story generation across complex domains.

3. **Structured output reliability:** Recent advances in JSON mode and guided generation mean LLMs produce consistently parseable outputs, eliminating the "interesting but unusable" problem of earlier generations.

Engineers who master LLM-assisted planning techniques can reduce planning overhead by 60-70% while improving documentation quality and enabling true asynchronous collaboration.

## Technical Components

### 1. Prompt Engineering for Structured Story Generation

The quality of your story map depends critically on prompt design. LLMs need explicit structure, context, and constraints to generate useful stories.

**Technical Explanation:**

Effective prompts for story mapping require three layers:
- **Context layer:** Product domain, user types, constraints
- **Structure layer:** Expected output format, relationships, hierarchies
- **Quality layer:** Criteria for good stories, common pitfalls to avoid

**Practical Implementation:**

```python
from typing import List, Dict
from pydantic import BaseModel, Field
import anthropic

class StoryPromptTemplate:
    """Reusable prompt templates for different story mapping phases"""
    
    @staticmethod
    def activity_backbone(
        product_desc: str,
        user_types: List[str],
        business_goals: List[str]
    ) -> str:
        """Generate prompt for high-level activity identification"""
        return f"""You are a product analyst creating a user story map.

PRODUCT CONTEXT:
{product_desc}

TARGET USERS:
{chr(10).join(f"- {user}" for user in user_types)}

BUSINESS GOALS:
{chr(10).join(f"- {goal}" for goal in business_goals)}

TASK:
Identify 5-8 high-level activities that represent the user's journey.
Activities should:
- Represent what users DO, not what they WANT (use verbs)
- Flow in chronological order
- Cover the complete user journey from discovery to completion
- Be technology-agnostic (focus on user needs, not implementation)

OUTPUT FORMAT (JSON):
{{
  "activities": [
    {{
      "sequence": 1,
      "name": "Discover options",
      "description": "User becomes aware of and evaluates available solutions",
      "user_goal": "Find a solution that meets my needs",
      "success_metric": "Time to find relevant options"
    }}
  ]
}}"""

    @staticmethod
    def user_stories_for_activity(
        activity_name: str,
        activity_desc: str,
        product_context: str,
        mvp_mode: bool = False
    ) -> str:
        """Generate prompt for stories within an activity"""
        scope_guidance = """Focus on MVP stories only:
- Core functionality required for activity completion
- No nice-to-have features
- Prioritize stories marked as 'high' priority
""" if mvp_mode else "Include full range of stories from must-have to nice-to-have."
        
        return f"""Generate user stories for this activity:

ACTIVITY: {activity_name}
DESCRIPTION: {activity_desc}

PRODUCT CONTEXT:
{product_context}

SCOPE:
{scope_guidance}

Generate 4-6 user stories following this structure:
- "As a [user type], I want [action] so that [benefit]"
- Each story should be independently deliverable
- Include 3-5 concrete acceptance criteria per story
- Avoid technical implementation details

STORY CRITERIA:
- Independent: Can be developed without depending on other stories
- Negotiable: Details can be discussed and refined
- Valuable: Delivers clear user value
- Estimable: Team can estimate effort
- Small: Can be completed in one sprint
- Testable: Clear acceptance criteria

OUTPUT FORMAT (JSON):
{{
  "stories": [
    {{
      "id": "unique-id",
      "as_a": "user type",
      "i_want": "action/capability",
      "so_that": "benefit/value",
      "acceptance_criteria": [
        "Given [context], when [action], then [outcome]",
        "..."
      ],
      "priority": "high|medium|low",
      "estimated_effort": "small|medium|large",
      "dependencies": ["story-id-1"]
    }}
  ]
}}"""

# Usage example
def generate_initial_backbone(product_info: Dict) -> List[Dict]:
    """Generate activity backbone with proper context"""
    client = anthropic.Anthropic()
    
    prompt = StoryPromptTemplate.activity_backbone(
        product_desc=product_info['description'],
        user_types=product_info['user_types'],
        business_goals=product_info['business_goals']
    )
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7  # Allow creativity in activity identification
    )
    
    return json.loads(response.content[0].text)['activities']
```

**Trade-offs:**

- **Detailed prompts vs. flexibility:** Highly structured prompts produce consistent output but may miss creative alternatives. Balance with temperature settings and multiple generation passes.
- **Context size vs. focus:** Including full product specs improves relevance but dilutes focus. Use hierarchical generation: context → activities → stories.

### 2. Validation and Quality Control Loops

LLM outputs require validation. Raw generated stories often contain logical gaps, duplicate functionality, or missing acceptance criteria.

**Technical Explanation:**

Implement automated validation layers that check structural completeness, logical consistency, and business rule compliance before human review.

**Implementation:**

```python
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ValidationSeverity(Enum):
    ERROR = "error"      # Must fix before use
    WARNING = "warning"  # Should review
    INFO = "info"        # Nice to improve

@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    story_id: str
    field: str
    message: str
    suggestion: Optional[str] = None

class StoryValidator:
    """Automated validation for generated user stories"""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def validate_story_structure(self, story: UserStory) -> List[ValidationIssue]:
        """Check structural completeness and quality"""
        issues = []
        
        # Check mandatory fields
        if not story.as_a or len(story.as_a) < 3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                story_id=getattr(story, 'id', 'unknown'),
                field='as_a',
                message='User type missing or too short',
                suggestion='Specify a concrete user role (e.g., "first-time buyer", "system administrator")'
            ))
        
        # Check for implementation details leaking into story
        impl_keywords = ['database', 'API', 'cache', 'server', 'JSON', 'REST']
        combined_text = f"{story.i_want} {story.so_that}".lower()
        
        if any(keyword in combined_text for keyword in impl_keywords):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                story_id=getattr(story, 'id', 'unknown'),
                field='i_want',
                message='Story contains implementation details',
                suggestion='Focus on user needs, not technical implementation'
            ))
        
        # Check acceptance criteria quality
        if not story.acceptance_criteria or len(story.acceptance_criteria) < 2:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                story_id=getattr(story, 'id', 'unknown'),
                field='acceptance_criteria',
                message='Insufficient acceptance criteria',
                suggestion='Include at least 2-3 testable criteria in Given/When/Then format'
            ))
        
        # Check for vague acceptance criteria
        vague_terms = ['properly', 'correctly', 'quickly', 'easily', 'user-friendly']
        for i, criterion in enumerate(story.acceptance_criteria):
            if any(term in criterion.lower() for term in vague_terms):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    story_id=getattr(story, 'id', 'unknown'),
                    field=f'acceptance_criteria[{i}]',
                    message=f'Vague criterion: "{criterion}"',
                    suggestion='Use measurable, specific criteria'
                ))
        
        return issues
    
    def validate_story_map_coherence(
        self, 
        activities: List[Activity]
    ) -> List[ValidationIssue]:
        """Check logical flow and coverage across the entire map"""
        issues = []
        all_story_ids = set()
        
        # Check for duplicate story IDs
        for activity in activities:
            for story in activity.user_stories:
                story_id = getattr(story, 'id', None)
                if story_id:
                    if story_id in all_story_ids:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            story_id=story_id,
                            field='id',
                            message='Duplicate story ID found'
                        ))
                    all_story_ids.add(story_id)
        
        # Check for orphaned dependencies
        for activity in activities:
            for story in activity.user_stories:
                if hasattr(story, 'dependencies'):
                    for dep_id in story.dependencies:
                        if dep_id not in all_story_ids:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                story_id=getattr(story, 'id', 'unknown'),
                                field='dependencies',
                                message=f'Referenced story "{dep_id}" does not exist'
                            ))
        
        # Check activity sequence
        sequences = [a.sequence for a in activities]
        if len(sequences) != len(set(sequences)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                story_id='map',
                field='activity_sequences',
                message='Duplicate or missing sequence numbers in activities'
            ))
        
        return issues
    
    def format_validation_report(self, issues: List[ValidationIssue]) -> str:
        """Generate human-readable validation report"""
        if not issues:
            return "✓ All validation checks passed"
        
        report = ["Validation Issues Found:\n"]
        
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]
        infos = [i for i in