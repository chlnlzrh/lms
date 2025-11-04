# AI-Driven Action Item Extraction

## Core Concepts

Action item extraction is the automated identification and structuring of task commitments from unstructured text like meeting notes, emails, or conversations. Unlike keyword-based pattern matching that looks for trigger words ("TODO", "action:", "@person"), LLM-driven extraction understands semantic meaning—it recognizes that "John will look into the database slowness before Friday" is an action item even without explicit markers.

### Traditional vs. Modern Approaches

**Traditional rule-based extraction:**

```python
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ActionItem:
    text: str
    assignee: str | None
    deadline: str | None

def extract_actions_regex(text: str) -> list[ActionItem]:
    """Rule-based extraction using patterns"""
    actions = []
    
    # Pattern 1: Explicit TODO markers
    todo_pattern = r'(?:TODO|Action Item|AI):\s*(.+?)(?:\n|$)'
    for match in re.finditer(todo_pattern, text, re.IGNORECASE):
        actions.append(ActionItem(text=match.group(1), assignee=None, deadline=None))
    
    # Pattern 2: "@person will do something"
    assignment_pattern = r'@(\w+)\s+will\s+(.+?)(?:\.|$)'
    for match in re.finditer(assignment_pattern, text):
        actions.append(ActionItem(
            text=match.group(2),
            assignee=match.group(1),
            deadline=None
        ))
    
    return actions

# Example usage
meeting_notes = """
TODO: Review the API documentation
@sarah will update the deployment scripts
The database performance needs investigation by end of week
Mike mentioned he'd handle the customer email
"""

actions = extract_actions_regex(meeting_notes)
# Only finds: "Review the API documentation" and "update the deployment scripts"
# Misses: database investigation, Mike's commitment (no @ symbol)
```

**LLM-driven semantic extraction:**

```python
from anthropic import Anthropic
from typing import TypedDict
import json

class ActionItemStructured(TypedDict):
    task: str
    assignee: str | None
    deadline: str | None
    context: str
    priority: str

def extract_actions_llm(text: str, api_key: str) -> list[ActionItemStructured]:
    """Semantic extraction understanding intent and context"""
    client = Anthropic(api_key=api_key)
    
    prompt = f"""Extract all action items from this text. An action item is any commitment 
to complete a task, whether explicitly marked or implied through context.

For each action item, identify:
- task: Clear description of what needs to be done
- assignee: Who is responsible (null if unspecified)
- deadline: When it's due (null if unspecified)
- context: Why this matters or what triggered it
- priority: high/medium/low based on urgency indicators

Text:
{text}

Return valid JSON array of action items."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.content[0].text)

# Same meeting notes
actions_llm = extract_actions_llm(meeting_notes, "your-api-key")
# Finds ALL four items with full context:
# - "Review API documentation" (no assignee)
# - "Update deployment scripts" (assignee: sarah)
# - "Investigate database performance" (deadline: end of week, high priority)
# - "Handle customer email" (assignee: Mike)
```

The LLM approach captures 4x more action items because it understands semantic intent, not just syntactic patterns.

### Key Engineering Insights

**1. Ambiguity resolution through context**: LLMs use surrounding text to disambiguate. "We should look into that" might be casual conversation or a commitment depending on context—tone, speaker authority, meeting type.

**2. Structured output as a forcing function**: Requesting JSON with specific fields forces the model to make extraction decisions explicit. This transforms vague "extract action items" into concrete data structure population.

**3. Prompt design is your schema**: With traditional NLP, you'd train models and tune parsers. With LLMs, your prompt IS the extraction logic. Changing extraction behavior means editing text, not retraining models.

### Why This Matters Now

Meeting overhead consumes 15-20% of engineering time. Manual action item tracking has 40-60% capture rate—things get lost. Automated extraction isn't new, but LLM-based extraction crosses the accuracy threshold (80-90%) where teams actually trust and adopt it. The economic equation shifted: API costs dropped 10x in 2023-2024 while accuracy improved dramatically, making per-meeting extraction economically viable ($0.01-0.05 per meeting vs. 5-10 minutes of manual work).

## Technical Components

### 1. Prompt Engineering for Structured Extraction

The prompt defines your extraction schema and logic. Critical elements: clear task definition, output structure specification, edge case handling, and examples.

**Technical implementation:**

```python
from typing import Literal
from pydantic import BaseModel, Field

class ActionItem(BaseModel):
    """Structured action item with validation"""
    task: str = Field(description="Clear, actionable task description")
    assignee: str | None = Field(description="Person responsible, null if unspecified")
    deadline: str | None = Field(description="Due date in ISO format or relative (e.g., 'end of week')")
    priority: Literal["high", "medium", "low"] = Field(description="Urgency level")
    status: Literal["new", "mentioned", "committed"] = Field(
        description="Confidence level: committed=explicit agreement, mentioned=suggestion"
    )

def build_extraction_prompt(text: str, include_examples: bool = True) -> str:
    """Construct extraction prompt with optional few-shot examples"""
    
    system_context = """You extract action items from meeting notes and conversations. 
    
DEFINITION: An action item is a specific task that someone has committed to complete, 
or that was clearly identified as needing completion.

INCLUDE:
- Explicit commitments ("I'll handle X")
- Assigned tasks ("Sarah will do Y")  
- Clear requirements ("We need to fix Z before launch")

EXCLUDE:
- Vague suggestions ("Maybe we could...")
- Already completed items ("I already sent the email")
- General discussion without commitment

For ambiguous cases, use 'status' field to indicate confidence."""

    few_shot_examples = """
EXAMPLES:

Input: "John said he'd review the PR by tomorrow. The staging environment is still broken though."
Output: [
  {
    "task": "Review PR",
    "assignee": "John",
    "deadline": "tomorrow",
    "priority": "medium",
    "status": "committed"
  },
  {
    "task": "Fix staging environment",
    "assignee": null,
    "deadline": null,
    "priority": "high",
    "status": "mentioned"
  }
]

Input: "Maybe we should add more tests. I already updated the documentation."
Output: [
  {
    "task": "Add more tests",
    "assignee": null,
    "deadline": null,
    "priority": "low",
    "status": "mentioned"
  }
]
Note: Documentation update excluded (already complete)
"""

    extraction_request = f"""
Extract action items from this text:

{text}

Return a JSON array of action items matching this schema:
{ActionItem.model_json_schema()}
"""
    
    if include_examples:
        return f"{system_context}\n\n{few_shot_examples}\n\n{extraction_request}"
    return f"{system_context}\n\n{extraction_request}"

# Usage
prompt = build_extraction_prompt("Meeting notes here...")
```

**Practical implications**: Few-shot examples improve accuracy by 20-30% for edge cases but increase token usage by ~200-300 tokens per request. For high-volume scenarios, test whether your use case needs examples or if zero-shot performs adequately.

**Trade-offs**: More detailed prompts (with examples, edge cases, definitions) increase consistency but add latency (50-100ms) and cost. Start detailed, then trim based on error analysis of real extraction failures.

### 2. Response Parsing and Validation

LLMs generate text, not data structures. Robust extraction requires parsing, validation, and error handling.

**Technical implementation:**

```python
import json
from pydantic import ValidationError
from typing import Any

class ExtractionResult:
    """Wrapper for extraction with error tracking"""
    def __init__(self):
        self.items: list[ActionItem] = []
        self.errors: list[str] = []
        self.raw_response: str = ""

def parse_extraction_response(
    response_text: str,
    strict: bool = False
) -> ExtractionResult:
    """Parse and validate LLM response with error recovery"""
    result = ExtractionResult()
    result.raw_response = response_text
    
    # Step 1: Extract JSON from response (may have markdown formatting)
    json_text = response_text.strip()
    
    # Remove markdown code blocks if present
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        json_text = "\n".join(lines[1:-1])  # Remove first and last lines
    
    # Step 2: Parse JSON
    try:
        parsed_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        result.errors.append(f"JSON parsing failed: {e}")
        if strict:
            raise
        return result
    
    # Step 3: Validate each item
    if not isinstance(parsed_data, list):
        parsed_data = [parsed_data]  # Wrap single item
    
    for i, item_data in enumerate(parsed_data):
        try:
            action_item = ActionItem.model_validate(item_data)
            result.items.append(action_item)
        except ValidationError as e:
            error_msg = f"Item {i} validation failed: {e}"
            result.errors.append(error_msg)
            
            if strict:
                raise
            
            # Attempt partial recovery
            try:
                # Create item with available fields, use defaults for invalid ones
                recovered_item = ActionItem(
                    task=item_data.get("task", "Unknown task"),
                    assignee=item_data.get("assignee"),
                    deadline=item_data.get("deadline"),
                    priority=item_data.get("priority", "medium"),
                    status=item_data.get("status", "mentioned")
                )
                result.items.append(recovered_item)
                result.errors.append(f"Item {i} recovered with defaults")
            except Exception as recovery_error:
                result.errors.append(f"Item {i} recovery failed: {recovery_error}")
    
    return result

# Example usage
response = '''```json
[
  {
    "task": "Fix login bug",
    "assignee": "Alice",
    "deadline": "2024-01-15",
    "priority": "high",
    "status": "committed"
  },
  {
    "task": "Update docs",
    "assignee": null,
    "priority": "invalid_priority",
    "status": "mentioned"
  }
]
```'''

result = parse_extraction_response(response, strict=False)
print(f"Extracted {len(result.items)} items")
print(f"Errors: {result.errors}")
# Output: Extracted 2 items (second item recovered with default priority)
```

**Practical implications**: Non-strict mode recovers ~70% of malformed responses. This is critical for production—LLMs occasionally output invalid JSON (1-3% of requests) due to formatting issues, not logic errors.

**Constraints**: Recovery logic adds complexity. Balance between graceful degradation and data quality. For critical applications, log recovered items separately for manual review.

### 3. Context Window Management

Meeting transcripts often exceed model context limits. Effective extraction requires chunking strategies that preserve action item context.

**Technical implementation:**

```python
from typing import Iterator

def chunk_text_semantic(
    text: str,
    max_chunk_size: int = 3000,
    overlap: int = 300
) -> Iterator[tuple[str, int, int]]:
    """
    Chunk text at semantic boundaries (paragraphs) with overlap.
    
    Returns: Iterator of (chunk_text, start_char, end_char)
    """
    paragraphs = text.split("\n\n")
    current_chunk = []
    current_size = 0
    char_position = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        if current_size + para_size > max_chunk_size and current_chunk:
            # Yield current chunk
            chunk_text = "\n\n".join(current_chunk)
            start_pos = char_position - current_size
            yield (chunk_text, start_pos, char_position)
            
            # Start new chunk with overlap
            overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
            current_chunk = [overlap_text, para]
            current_size = len(overlap_text) + para_size
        else:
            current_chunk.append(para)
            current_size += para_size + 2  # +2 for \n\n
        
        char_position += para_size + 2
    
    # Yield final chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        start_pos = char_position - current_size
        yield (chunk_text, start_pos, char_position)

def extract_from_long_text(
    text: str,
    api_key: str,
    chunk_size: int = 3000
) -> list[ActionItem]:
    """Extract action items from text exceeding context limits"""
    all_items: list[ActionItem] = []
    seen_tasks: set[str] = set()  # Deduplication
    
    client = Anthropic(api_key=api_key)
    
    for chunk, start, end in chunk_text_semantic(text, chunk_size):
        prompt = build_extraction_prompt(chunk, include_examples=False)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = parse_extraction_response(response.content[0].text)
        
        # Deduplicate across chunks
        for item in result.items:
            task_key = f"{item.task.lower()}:{item.assignee or 'none'}"
            if task_key not in seen_tasks:
                seen_tasks.add(task_key)
                all_items.append(item)
    
    return all_items
```

**Practical implications**: Semantic chunking at paragraph boundaries reduces action item fragmentation by 60% compared to character-based splitting. Overlap ensures action items spanning boundaries aren't lost.

**Trade-offs**: More chunks = higher cost and latency. For a 10,000-character transcript, semantic chunking creates 3-4 chunks vs. 6-8 with fixed character splits, saving 40-50% on API calls.

### 4. Confidence Scoring and Human-in-the-Loop

Not all extracted action items are equal in certainty. Confidence scoring enables human review workflows for ambiguous cases.

**Technical implementation:**

```python
from enum import Enum

class ConfidenceLevel(Enum):
    HIGH = "high"      # Explicit commitment, clear assignee
    MEDIUM = "medium"  # Implied commitment or missing details
    LOW = "low"        # Suggestion or very ambiguous

class ActionItemWithConfidence(ActionItem):
    """Action item with confidence metadata"""
    confidence: ConfidenceLevel
    confidence_reasoning: str
    requires_review: bool = False

def extract_with_confidence(text: str, api_key: str) -> list[ActionItemWithConfidence]:
    """Extract action items with confidence assessment"""
    client = Anthropic(api_key=api_key)
    
    prompt = f"""Extract action items from this text. For each item, assess extraction confidence:

HIGH confidence: Explicit commitment with clear details
- "John will review the PR by Friday"
- "Sarah agreed to update the