# Meeting & Communication Summarization with LLMs

## Core Concepts

Meeting and communication summarization is the process of using language models to extract, condense, and structure information from conversational data—transforming hours of unstructured dialogue into actionable insights. Unlike document summarization, meeting content contains interruptions, tangents, cross-talk, and implicit context that make it technically challenging to process.

### Traditional vs. Modern Approaches

```python
# Traditional approach: Rule-based extraction
def traditional_meeting_summary(transcript: str) -> dict:
    """Rule-based approach with rigid patterns"""
    summary = {
        "action_items": [],
        "decisions": [],
        "topics": []
    }
    
    # Brittle pattern matching
    lines = transcript.split('\n')
    for line in lines:
        if "action item" in line.lower() or "todo" in line.lower():
            summary["action_items"].append(line)
        if "we decided" in line.lower() or "agreement" in line.lower():
            summary["decisions"].append(line)
    
    # Misses: context, implicit decisions, paraphrased actions
    # False positives: "we haven't decided", "not an action item"
    return summary

# Modern LLM approach: Semantic understanding
def llm_meeting_summary(transcript: str, api_client) -> dict:
    """Context-aware extraction with semantic understanding"""
    prompt = """Analyze this meeting transcript and extract:
1. Action items (who, what, when)
2. Decisions made (what was decided and why)
3. Key discussion topics with outcomes

Transcript:
{transcript}

Format as JSON with arrays for each category."""
    
    response = api_client.complete(prompt.format(transcript=transcript))
    
    # Captures: implicit agreements, context-dependent actions, 
    # speaker intent, unspoken conclusions
    return parse_json(response)
```

The traditional approach fails because meetings are inherently ambiguous. When someone says "let's revisit this next week," is that a decision to defer, an action item to schedule, or just a verbal placeholder? LLMs use contextual understanding—who said it, what preceded it, the overall meeting flow—to make intelligent judgments.

### Key Engineering Insights

**1. Meetings are compression problems with lossy trade-offs**  
You're reducing 60 minutes (15,000+ words) to 200-500 words. What you preserve determines utility. A 90% compression optimized for action items will lose strategic context. A summary optimized for absent stakeholders will differ from one for participants who need reminders.

**2. Speaker attribution matters more than content accuracy**  
"The engineering team agreed to ship by Friday" vs "Marketing suggested Friday but engineering pushed back" are both summaries of the same discussion, but operationally different. Misattributing decisions or consensus creates downstream failures.

**3. Temporal structure changes meaning**  
"We'll use PostgreSQL" → 10 minutes later → "Actually, MongoDB makes more sense" requires understanding that the second statement supersedes the first. LLMs must track decision evolution, not just extract statements.

### Why This Matters Now

The average engineer spends 8-12 hours weekly in meetings. If you attend a 1-hour meeting with 5 people, that's 5 person-hours invested. Without good summarization, 30-40% of decisions are misremembered or forgotten within 48 hours. LLM summarization isn't about convenience—it's about recovering value from one of your organization's largest time investments.

Additionally, asynchronous work patterns mean people increasingly need to "attend" meetings they missed. High-quality summarization is becoming infrastructure, not a nice-to-have.

## Technical Components

### 1. Transcript Preprocessing & Normalization

Raw meeting transcripts from tools like Zoom, Teams, or Otter.ai contain noise that degrades LLM performance: filler words, false starts, crosstalk, timestamps, and technical artifacts.

**Technical Explanation:**  
Preprocessing normalizes the input to maximize information density within token limits. You're essentially doing feature engineering for the LLM—removing noise, adding structure, and preserving critical metadata.

```python
from typing import List, Dict
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TranscriptSegment:
    speaker: str
    text: str
    timestamp: datetime
    confidence: float = 1.0

def preprocess_transcript(
    raw_segments: List[Dict[str, any]],
    remove_fillers: bool = True,
    merge_same_speaker: bool = True,
    min_segment_length: int = 10
) -> List[TranscriptSegment]:
    """
    Normalize transcript for LLM processing
    
    Args:
        raw_segments: List of {speaker, text, timestamp, confidence}
        remove_fillers: Strip um, uh, like, you know
        merge_same_speaker: Combine consecutive segments from same speaker
        min_segment_length: Drop segments shorter than N characters
    """
    segments = []
    
    for seg in raw_segments:
        text = seg['text'].strip()
        
        # Remove filler words
        if remove_fillers:
            fillers = r'\b(um|uh|like|you know|i mean|sort of|kind of)\b'
            text = re.sub(fillers, '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Skip very short segments (likely crosstalk artifacts)
        if len(text) < min_segment_length:
            continue
        
        segments.append(TranscriptSegment(
            speaker=seg['speaker'],
            text=text,
            timestamp=datetime.fromisoformat(seg['timestamp']),
            confidence=seg.get('confidence', 1.0)
        ))
    
    # Merge consecutive segments from same speaker
    if merge_same_speaker and segments:
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.speaker == merged[-1].speaker:
                # Only merge if within 5 seconds
                if (seg.timestamp - merged[-1].timestamp).seconds < 5:
                    merged[-1].text += " " + seg.text
                    continue
            merged.append(seg)
        segments = merged
    
    return segments

def format_for_llm(segments: List[TranscriptSegment]) -> str:
    """Convert to LLM-friendly format"""
    formatted_lines = []
    
    for seg in segments:
        # Include timestamp for temporal reasoning
        time_marker = seg.timestamp.strftime("%H:%M")
        formatted_lines.append(f"[{time_marker}] {seg.speaker}: {seg.text}")
    
    return "\n".join(formatted_lines)
```

**Practical Implications:**  
Without preprocessing, you'll waste 20-30% of your context window on noise. A 60-minute meeting might generate 12,000 raw tokens but only need 8,000 after cleanup. This directly impacts cost and whether you can fit the meeting into a single API call.

**Constraints:**  
Aggressive preprocessing can remove context ("um, actually no" → "actually no" loses emphasis). Balance noise removal with semantic preservation. For critical meetings, keep fillers; for routine syncs, strip aggressively.

### 2. Hierarchical Summarization Strategy

Long meetings exceed token limits. A 90-minute discussion might generate 18,000 tokens, but models like GPT-4 or Claude have limits (8k-128k depending on version). You need chunking strategies.

**Technical Explanation:**  
Hierarchical summarization processes the transcript in overlapping windows, summarizes each chunk, then summarizes the summaries. It's a map-reduce pattern for narrative content.

```python
from typing import List, Optional
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Accurately count tokens for a given model"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_transcript(
    transcript: str,
    max_chunk_tokens: int = 3000,
    overlap_tokens: int = 200
) -> List[str]:
    """
    Split transcript into overlapping chunks
    
    Overlap ensures context continuity across boundaries
    """
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(transcript)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + max_chunk_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        
        # Move start back by overlap amount for next chunk
        start = end - overlap_tokens
    
    return chunks

def hierarchical_summarize(
    transcript: str,
    api_client,
    max_tokens_per_chunk: int = 3000,
    model: str = "gpt-4"
) -> dict:
    """
    Multi-level summarization for long meetings
    
    Level 1: Chunk summaries (detailed)
    Level 2: Combined summary (strategic)
    """
    total_tokens = count_tokens(transcript)
    
    # If fits in one call, use direct summarization
    if total_tokens <= max_tokens_per_chunk:
        return direct_summarize(transcript, api_client)
    
    # Level 1: Summarize each chunk
    chunks = chunk_transcript(transcript, max_tokens_per_chunk)
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        prompt = f"""Summarize this portion of a meeting (part {i+1}/{len(chunks)}):

{chunk}

Extract:
- Key points discussed
- Any decisions or action items
- Important questions raised

Keep context for combining with other sections."""
        
        summary = api_client.complete(prompt, max_tokens=500)
        chunk_summaries.append(summary)
    
    # Level 2: Combine chunk summaries
    combined_prompt = f"""Here are summaries from different parts of a meeting:

{chr(10).join(f"Section {i+1}: {s}" for i, s in enumerate(chunk_summaries))}

Create a unified summary with:
1. Action Items (who, what, when)
2. Decisions Made
3. Key Discussion Topics
4. Open Questions

Format as JSON."""
    
    final_summary = api_client.complete(combined_prompt, max_tokens=1000)
    
    return {
        "method": "hierarchical",
        "chunks_processed": len(chunks),
        "summary": final_summary,
        "chunk_summaries": chunk_summaries  # Keep for debugging
    }
```

**Practical Implications:**  
Hierarchical summarization costs 2-3x more in API calls but handles unlimited meeting lengths. A 2-hour meeting might require 4 chunk summaries + 1 final summary = 5 API calls vs. impossible with direct summarization.

**Trade-offs:**  
Each summarization layer loses fidelity. Details mentioned once in a 90-minute meeting might not survive two rounds of compression. For critical details, use hybrid approaches: hierarchical for overview, targeted extraction for specific types (action items only).

### 3. Structured Extraction with JSON Schema

Free-form summaries are hard to integrate into workflows. Structured extraction produces machine-readable outputs you can route to task systems, calendars, or databases.

**Technical Explanation:**  
By defining a JSON schema and using function calling or structured output modes, you enforce consistent output format regardless of meeting content.

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ActionItem(BaseModel):
    task: str = Field(description="What needs to be done")
    assignee: Optional[str] = Field(description="Who is responsible")
    deadline: Optional[str] = Field(description="When it's due (ISO date or relative)")
    priority: Priority = Field(default=Priority.MEDIUM)
    context: Optional[str] = Field(description="Why this is needed")

class Decision(BaseModel):
    decision: str = Field(description="What was decided")
    rationale: Optional[str] = Field(description="Why this decision was made")
    alternatives_considered: List[str] = Field(default_factory=list)
    dissenters: List[str] = Field(default_factory=list)

class MeetingSummary(BaseModel):
    action_items: List[ActionItem]
    decisions: List[Decision]
    key_topics: List[str] = Field(description="Main discussion themes")
    unresolved_questions: List[str] = Field(default_factory=list)
    next_meeting_suggestions: Optional[str] = None

def extract_structured_summary(
    transcript: str,
    api_client,
    schema: type[BaseModel] = MeetingSummary
) -> MeetingSummary:
    """
    Extract meeting summary with enforced structure
    
    Uses function calling or JSON mode depending on API
    """
    prompt = f"""Analyze this meeting transcript and extract structured information:

{transcript}

Extract all action items with assignees and deadlines.
Identify explicit decisions and their rationale.
Note any unresolved questions or topics needing follow-up.

Be precise with attribution - only assign tasks/decisions if clearly stated."""

    # Example using OpenAI function calling
    response = api_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        functions=[{
            "name": "record_meeting_summary",
            "description": "Record structured meeting summary",
            "parameters": schema.model_json_schema()
        }],
        function_call={"name": "record_meeting_summary"}
    )
    
    # Parse function call arguments into Pydantic model
    function_args = response.choices[0].message.function_call.arguments
    return schema.model_validate_json(function_args)

# Usage example
def process_and_route_meeting(transcript: str, api_client, task_system, calendar):
    """Extract structured data and route to appropriate systems"""
    summary = extract_structured_summary(transcript, api_client)
    
    # Create tasks in project management system
    for item in summary.action_items:
        if item.assignee:
            task_system.create_task(
                title=item.task,
                assignee=item.assignee,
                due_date=item.deadline,
                priority=item.priority.value,
                description=item.context
            )
    
    # Log decisions
    for decision in summary.decisions:
        decision_log.record(
            decision=decision.decision,
            rationale=decision.rationale,
            timestamp=datetime.now()
        )
    
    return summary
```

**Practical Implications:**  
Structured extraction enables automation. Instead of someone manually creating Jira tickets from a summary document, the system does it automatically. This recovers the 15-30 minutes typically spent on post-meeting administrative work.

**Constraints:**  
Schema enforcement can cause LLMs to hallucinate data to fill required fields. Make most fields optional and validate outputs. If an action item has no clear assignee, forcing the model to provide one will produce incorrect attributions.

### 4. Multi-Perspective Summarization

Different stakeholders need different summaries from the same meeting. Engineers want technical decisions and blockers; executives want strategic outcomes and risks.

**Technical Explanation:**  
Perspective-based summarization uses role-specific prompts to extract relevant information for different audiences from a single transcript.

```python
from typing import Dict, List
from enum import Enum

class Perspective(str, Enum):
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    PARTICIPANT = "participant"
    ABSENT_STAKEHOLDER = "absent_stakeholder"

PERSPECTIVE_PROMPTS = {
    Perspective.EXECUTIVE: """Summarize this meeting for an executive who needs strategic oversight:
- Key decisions and their business impact
- Risks or blockers escalated
- Resource needs or timeline changes
- Strategic alignment issues

Keep technical details minimal. Focus on "what" and "why", not "how".""",

    Perspective.TECHNICAL: """Summarize this meeting for engineers who need implementation details:
- Technical decisions (architecture, tools, approaches)
- Specific blockers or dependencies
- Action items with technical context
- Open technical questions

Include relevant technical details and rationale.""",

    Perspective.PARTICIPANT: """Summarize for someone who attended but needs to recall key points:
- Quick refresher on main topics
- Personal action items
- Follow-up questions for them
- Important deadlines

Concise format, assumes they have context."""