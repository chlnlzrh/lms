# Sprint Ceremony Support with LLMs

## Core Concepts

Sprint ceremonies—standups, planning, retrospectives, reviews—consume 15-30% of engineering time in agile teams. These meetings generate repetitive cognitive overhead: synthesizing updates, extracting action items, identifying blockers, and maintaining context across iterations. LLMs can automate the mechanical aspects of ceremony support while preserving the human elements that drive team alignment.

### Engineering Analogy

**Traditional Approach:**
```python
class SprintCeremony:
    def __init__(self):
        self.notes = []
        self.action_items = []
        
    def record_meeting(self, transcript: str) -> None:
        """Scrum master manually takes notes during meeting"""
        self.notes.append(transcript)
        # Action items extracted manually after meeting
        # Takes 15-30 minutes of post-processing
        
    def get_summary(self) -> str:
        """Manual summarization, often incomplete"""
        return "\n".join(self.notes)
```

**LLM-Assisted Approach:**
```python
from typing import List, Dict
from dataclasses import dataclass
import json

@dataclass
class ActionItem:
    task: str
    assignee: str
    priority: str
    context: str

class LLMCeremonySupport:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.context_window = []
        
    def process_meeting(self, transcript: str, 
                       ceremony_type: str) -> Dict:
        """Real-time processing with structured output"""
        prompt = self._build_ceremony_prompt(transcript, ceremony_type)
        
        result = self.llm.generate(
            prompt,
            response_format="json",
            temperature=0.2  # Low temperature for consistency
        )
        
        return self._parse_structured_output(result)
    
    def _build_ceremony_prompt(self, transcript: str, 
                               ceremony_type: str) -> str:
        return f"""Analyze this {ceremony_type} transcript.
        
Extract:
1. Key decisions made
2. Action items with assignees
3. Blockers identified
4. Risks flagged
5. Sprint velocity indicators (if planning/review)

Transcript:
{transcript}

Return JSON with structure:
{{
  "decisions": [...],
  "action_items": [{{"task": "", "assignee": "", "priority": ""}}],
  "blockers": [...],
  "risks": [...],
  "metrics": {{}}
}}"""

    def _parse_structured_output(self, response: str) -> Dict:
        """Parse and validate LLM output"""
        try:
            data = json.loads(response)
            # Validation logic here
            return data
        except json.JSONDecodeError:
            # Fallback parsing or retry logic
            return self._retry_with_stricter_format(response)
```

The difference: **Manual ceremony support scales linearly with team size** (larger teams = more time in meetings). **LLM-assisted support scales logarithmically**—the incremental cost per additional team member approaches zero.

### Key Insights

1. **LLMs excel at pattern matching across ceremony types**: A standup blocker often becomes a retrospective discussion point. LLMs can track these patterns across meetings automatically, something humans struggle with due to cognitive load.

2. **Structured output matters more than natural language**: Engineering teams need action items in ticket systems, not prose summaries. Designing prompts for JSON/YAML output with specific schemas provides direct integration with tooling.

3. **Context preservation is the killer feature**: LLMs can maintain sprint-over-sprint context that humans forget. "This is the third sprint where frontend testing was mentioned as a blocker" is trivial for an LLM, difficult for humans.

4. **The 80/20 rule applies aggressively**: 80% of ceremony value comes from decisions, action items, and blockers. LLMs can extract these with 90%+ accuracy, while the remaining 20% (team dynamics, morale, subtle conflicts) requires human judgment.

### Why This Matters Now

Modern transformer models (2023+) crossed the reliability threshold for production ceremony support:

- **Token context windows** (32K-128K+) can hold entire sprint's worth of meeting transcripts
- **Function calling** enables direct integration with project management APIs
- **Structured output modes** guarantee parseable responses (no more regex parsing of prose)
- **Sub-$0.01 per meeting** cost makes this economically viable even for small teams

The bottleneck shifted from "Can LLMs do this?" to "How do we integrate this into existing workflows without disruption?"

## Technical Components

### 1. Meeting Transcript Processing

**Technical Explanation:**

Real-time meeting transcripts contain disfluencies, crosstalk, and context-dependent references. LLMs need preprocessing to handle speech artifacts while preserving semantic content.

```python
import re
from typing import Tuple

class TranscriptProcessor:
    def __init__(self):
        self.speaker_patterns = re.compile(r'^([A-Z][a-z]+):\s*')
        self.filler_words = {'um', 'uh', 'like', 'you know', 'sort of'}
        
    def clean_transcript(self, raw_transcript: str) -> str:
        """Remove speech artifacts while preserving meaning"""
        lines = raw_transcript.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preserve speaker labels
            speaker_match = self.speaker_patterns.match(line)
            if speaker_match:
                speaker = speaker_match.group(1)
                content = line[len(speaker)+1:].strip()
                
                # Remove excessive filler words
                words = content.split()
                filtered = [w for w in words 
                           if w.lower() not in self.filler_words]
                
                # Rejoin with proper spacing
                cleaned_content = ' '.join(filtered)
                cleaned_lines.append(f"{speaker}: {cleaned_content}")
            else:
                cleaned_lines.append(line)
                
        return '\n'.join(cleaned_lines)
    
    def segment_by_topic(self, transcript: str) -> List[Tuple[str, str]]:
        """Split transcript into topical segments"""
        # LLMs process focused chunks better than full transcripts
        prompt = f"""Segment this meeting into distinct topics.
        
For each segment, provide:
- Topic label (2-5 words)
- Start/end markers from transcript

Transcript:
{transcript}

Format: JSON array of {{"topic": "", "content": ""}}"""
        
        # This would call your LLM
        # Shown here as illustration of the pattern
        return self._call_llm_for_segmentation(prompt)
```

**Practical Implications:**

- **Chunk size matters**: Full 60-minute transcripts (10K+ tokens) cost more and process slower. Segment into 5-10 minute topical chunks for 3-5x faster processing.
- **Speaker identification is critical**: LLMs need to know who said what for accurate action item assignment. Anonymous transcripts reduce accuracy by ~30%.

**Real Constraints:**

- Speech-to-text APIs (Whisper, cloud services) have 2-10 second latency. Real-time processing requires buffering strategies.
- Multi-speaker overlap creates attribution errors. Audio quality directly impacts LLM input quality—garbage in, garbage out.

### 2. Structured Output Schemas

**Technical Explanation:**

Ceremony outputs must feed into existing tools (Jira, Linear, GitHub Issues). This requires rigid schema adherence, not flexible prose.

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ActionItemSchema(BaseModel):
    """Strict schema for action items"""
    task: str = Field(..., min_length=10, max_length=200)
    assignee: str = Field(..., pattern=r'^[A-Z][a-z]+\s[A-Z][a-z]+$')
    priority: Priority
    due_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    context: str = Field(..., max_length=500)
    related_blocker: Optional[str] = None
    
    @validator('task')
    def task_must_be_actionable(cls, v):
        """Ensure task starts with action verb"""
        action_verbs = {'create', 'update', 'fix', 'implement', 
                        'review', 'test', 'deploy', 'investigate'}
        first_word = v.split()[0].lower()
        if first_word not in action_verbs:
            raise ValueError(f'Task must start with action verb, got: {first_word}')
        return v

class StandupOutput(BaseModel):
    """Complete standup ceremony output"""
    date: str
    attendees: List[str]
    action_items: List[ActionItemSchema]
    blockers: List[Dict[str, str]]
    notable_progress: List[str]
    
def extract_standup_output(transcript: str, 
                          llm_client) -> StandupOutput:
    """Extract structured standup data with validation"""
    
    schema_json = ActionItemSchema.schema_json(indent=2)
    
    prompt = f"""Extract standup information from this transcript.

Required output schema:
{schema_json}

Rules:
- Every action item needs assignee (real name from transcript)
- Priority based on language: "urgent"/"asap"/"critical" = high
- Context should reference why the task matters
- If blocker mentioned, link it to related action item

Transcript:
{transcript}

Return valid JSON matching the schema."""

    response = llm_client.generate(
        prompt,
        temperature=0.1,  # Very low for schema adherence
        max_tokens=2000
    )
    
    try:
        # Pydantic validates against schema
        return StandupOutput.parse_raw(response)
    except ValidationError as e:
        # Log validation errors for prompt refinement
        print(f"Schema validation failed: {e}")
        # Implement retry logic with explicit error feedback
        return retry_with_validation_hints(transcript, e, llm_client)
```

**Practical Implications:**

- **Schema-first design**: Define output schemas before writing prompts. Prompts should reference the schema explicitly.
- **Validation catches hallucinations**: If LLM generates `assignee: "the frontend team"` instead of a person's name, Pydantic validation fails immediately.

**Trade-offs:**

- Stricter schemas (more validators) = more LLM retries = higher latency and cost
- Looser schemas = faster processing but more post-processing required
- Sweet spot: Validate critical fields (assignees, dates) strictly, allow flexibility in descriptions

### 3. Cross-Ceremony Context Tracking

**Technical Explanation:**

Sprint ceremonies don't exist in isolation. A planning commitment becomes a review deliverable becomes a retrospective discussion point. LLMs can maintain this context graph automatically.

```python
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set

class CeremonyContextTracker:
    def __init__(self, llm_client):
        self.llm = llm_client
        # Graph: ceremony_id -> related items
        self.context_graph = defaultdict(list)
        # Temporal index: topic -> [ceremony_ids] where discussed
        self.topic_timeline = defaultdict(list)
        
    def add_ceremony(self, ceremony_id: str, 
                     ceremony_type: str,
                     ceremony_date: datetime,
                     extracted_data: Dict) -> None:
        """Add ceremony to context graph"""
        
        # Extract topics/themes from this ceremony
        topics = self._extract_topics(extracted_data)
        
        for topic in topics:
            self.topic_timeline[topic].append({
                'ceremony_id': ceremony_id,
                'ceremony_type': ceremony_type,
                'date': ceremony_date,
                'data': extracted_data
            })
            
    def get_related_context(self, current_ceremony_data: Dict,
                           lookback_sprints: int = 3) -> str:
        """Generate context summary for current ceremony"""
        
        current_topics = self._extract_topics(current_ceremony_data)
        
        # Find previous ceremonies discussing same topics
        related_ceremonies = []
        for topic in current_topics:
            if topic in self.topic_timeline:
                # Get last N sprint ceremonies
                related_ceremonies.extend(
                    self.topic_timeline[topic][-lookback_sprints:]
                )
        
        # Deduplicate and sort by date
        unique_ceremonies = {c['ceremony_id']: c 
                            for c in related_ceremonies}
        sorted_ceremonies = sorted(unique_ceremonies.values(),
                                  key=lambda x: x['date'])
        
        # Build context summary
        context_parts = []
        for ceremony in sorted_ceremonies:
            summary = self._summarize_ceremony(ceremony)
            context_parts.append(summary)
            
        return "\n\n".join(context_parts)
    
    def _extract_topics(self, ceremony_data: Dict) -> Set[str]:
        """Extract key topics from ceremony data"""
        
        # Combine action items, blockers, decisions into text
        text_content = []
        if 'action_items' in ceremony_data:
            text_content.extend([item['task'] 
                               for item in ceremony_data['action_items']])
        if 'blockers' in ceremony_data:
            text_content.extend(ceremony_data['blockers'])
        if 'decisions' in ceremony_data:
            text_content.extend(ceremony_data['decisions'])
            
        combined_text = ' '.join(text_content)
        
        # Use LLM to extract topics (alternative: keyword extraction)
        prompt = f"""Extract 3-5 key topics from this ceremony content.
        
Topics should be:
- Technical areas (e.g., "authentication", "database performance")
- Features (e.g., "user onboarding", "payment integration")
- Process concerns (e.g., "testing coverage", "deployment pipeline")

Content:
{combined_text}

Return as JSON array of strings."""

        response = self.llm.generate(prompt, temperature=0.3)
        topics = json.loads(response)
        return set(topics)
    
    def _summarize_ceremony(self, ceremony: Dict) -> str:
        """Create brief summary of past ceremony"""
        ceremony_type = ceremony['ceremony_type']
        date = ceremony['date'].strftime('%Y-%m-%d')
        
        # Extract most relevant points
        data = ceremony['data']
        action_count = len(data.get('action_items', []))
        blocker_count = len(data.get('blockers', []))
        
        return f"""[{ceremony_type} on {date}]
- {action_count} action items created
- {blocker_count} blockers identified
- Key decisions: {', '.join(data.get('decisions', [])[:2])}"""

# Usage in retrospective with historical context
def generate_retrospective_insights(current_retro_data: Dict,
                                   context_tracker: CeremonyContextTracker,
                                   llm_client) -> Dict:
    """Generate retrospective insights with sprint history"""
    
    # Get relevant context from past ceremonies
    historical_context = context_tracker.get_related_context(
        current_retro_data,
        lookback_sprints=3
    )
    
    prompt = f"""Analyze this retrospective with historical context.

Historical Context (last 3 sprints):
{historical_context}

Current Retrospective Data:
{json.dumps(current_retro_data, indent=2)}

Provide insights on:
1. Recurring issues (mentioned in 2+ past ceremonies)
2. Progress on previously identified problems
3. New concerns not seen before
4. Suggested action items based on patterns

Return as JSON."""

    response = llm_client.generate(prompt, temperature=0.4)
    return json.loads(response)
```

**Practical Implications:**

- **Memory grows linearly with sprints**: After 10 sprints, you have 40-50 ceremonies. Context retrieval must be selective, not dumping all history into every prompt.
- **Topic extraction is the critical step**: Poor topic extraction = irrelevant context = wasted tokens and confused LLM outputs.