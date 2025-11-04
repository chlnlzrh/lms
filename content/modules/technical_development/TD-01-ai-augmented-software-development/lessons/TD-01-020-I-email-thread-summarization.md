# Email Thread Summarization with LLMs

## Core Concepts

Email thread summarization is the process of condensing multi-message email conversations into concise, actionable summaries while preserving critical information like decisions, action items, and context. For engineers, this represents a shift from rule-based text extraction to context-aware semantic understanding.

### Traditional vs. Modern Approach

```python
# Traditional approach: Rule-based extraction
import re
from datetime import datetime
from typing import List, Dict

def traditional_summarize(emails: List[Dict[str, str]]) -> str:
    """Rule-based email summarization - brittle and shallow"""
    summary = []
    
    # Extract participants
    participants = set()
    for email in emails:
        participants.add(email['from'])
    
    summary.append(f"Participants: {', '.join(participants)}")
    
    # Look for action items using keywords
    action_items = []
    keywords = ['todo', 'action item', 'please', 'need to', 'should']
    for email in emails:
        for keyword in keywords:
            if keyword in email['body'].lower():
                # Extract sentence containing keyword
                sentences = email['body'].split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        action_items.append(sentence.strip())
    
    if action_items:
        summary.append(f"Action Items: {'; '.join(action_items[:3])}")
    
    return '\n'.join(summary)

# Modern approach: LLM-based semantic understanding
from anthropic import Anthropic

def llm_summarize(emails: List[Dict[str, str]], api_key: str) -> str:
    """LLM-based summarization - context-aware and flexible"""
    client = Anthropic(api_key=api_key)
    
    # Structure the thread for the LLM
    thread_text = "\n\n---\n\n".join([
        f"From: {email['from']}\n"
        f"Date: {email['date']}\n"
        f"Subject: {email['subject']}\n\n"
        f"{email['body']}"
        for email in emails
    ])
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Summarize this email thread. Include:
1. Main topic and outcome
2. Key decisions made
3. Action items with owners
4. Unresolved questions

Email thread:
{thread_text}"""
        }]
    )
    
    return response.content[0].text

# Example usage comparison
emails = [
    {
        "from": "alice@example.com",
        "date": "2024-01-15",
        "subject": "Q1 Infrastructure Budget",
        "body": "We need to finalize the Q1 infrastructure budget. I'm proposing $50K for the migration project."
    },
    {
        "from": "bob@example.com",
        "date": "2024-01-15",
        "subject": "Re: Q1 Infrastructure Budget",
        "body": "That seems high. Can we break this down? What's driving the cost?"
    },
    {
        "from": "alice@example.com",
        "date": "2024-01-16",
        "subject": "Re: Q1 Infrastructure Budget",
        "body": "Main costs: $30K for compute, $15K for storage, $5K contingency. Bob, can you review the storage estimate by Friday?"
    }
]

# Traditional output: Misses nuance, extracts irrelevant fragments
print(traditional_summarize(emails))
# Output: "Participants: alice@example.com, bob@example.com
# Action Items: Can we break this down"

# LLM output: Understands context, identifies true action items
# "Topic: Q1 infrastructure budget for migration project
# Decision: $50K budget proposed, breakdown provided ($30K compute, $15K storage, $5K contingency)
# Action Items: Bob to review storage estimate by Friday
# Unresolved: Bob's concerns about costs not fully addressed"
```

The traditional approach uses pattern matching and keyword detection—it identifies that "can you review" is a request but misses context about what needs review. The LLM approach understands semantic meaning: it knows Bob is the action owner, Friday is the deadline, and the storage estimate is the deliverable.

### Key Engineering Insights

1. **Context Compression vs. Information Loss**: LLMs don't just extract—they compress context semantically. A 50-email thread with 10,000 tokens can be reduced to 200 tokens while preserving decision logic that keyword extraction would miss.

2. **Implicit Information Extraction**: LLMs infer information not explicitly stated. If someone writes "I'm concerned about latency," an LLM can identify this as a blocker even without the word "blocker."

3. **Temporal Reasoning**: Email threads have temporal logic—earlier concerns may be resolved later, or decisions may be reversed. LLMs track this naturally; rule-based systems require complex state machines.

### Why This Matters Now

Email remains the primary async communication tool in enterprise environments. The average knowledge worker receives 120+ emails daily, with 30% being multi-message threads. Manual processing costs 28 minutes per day per employee. At scale, automated summarization can recover 15-20% of knowledge worker time while improving information flow across teams.

## Technical Components

### 1. Thread Reconstruction and Ordering

Email threads arrive out of order, with missing messages, and unclear reply chains. Proper reconstruction is critical for accurate summarization.

**Technical Explanation**: Email clients use `In-Reply-To` and `References` headers to build thread trees, but these are often broken or missing. You must reconstruct threads using heuristics: subject line similarity, temporal proximity, participant overlap, and content references.

**Practical Implications**: Poor thread reconstruction leads to lost context. If you summarize messages 1, 2, 5 but skip 3, 4, the LLM may infer incorrect conclusions.

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import re

@dataclass
class Email:
    id: str
    subject: str
    from_addr: str
    to_addrs: List[str]
    date: datetime
    body: str
    in_reply_to: Optional[str] = None
    references: List[str] = None
    
def normalize_subject(subject: str) -> str:
    """Remove Re:, Fwd:, etc. for comparison"""
    return re.sub(r'^(Re:|Fwd:|RE:|FWD:)\s*', '', subject, flags=re.IGNORECASE).strip()

def reconstruct_thread(emails: List[Email]) -> List[Email]:
    """Reconstruct email thread from potentially unordered messages"""
    # Index by message ID
    by_id = {email.id: email for email in emails}
    
    # Build thread using headers
    thread = []
    root = None
    
    # Find root (message with no in_reply_to)
    for email in emails:
        if not email.in_reply_to:
            root = email
            break
    
    if not root:
        # Fallback: oldest message with matching subject
        root = min(emails, key=lambda e: e.date)
    
    thread.append(root)
    remaining = [e for e in emails if e.id != root.id]
    
    # Build thread by following reply chains
    while remaining:
        added = False
        for email in remaining[:]:
            # Check if this replies to something in thread
            if email.in_reply_to and email.in_reply_to in by_id:
                thread.append(email)
                remaining.remove(email)
                added = True
            # Fallback: same subject, later date
            elif normalize_subject(email.subject) == normalize_subject(root.subject):
                thread.append(email)
                remaining.remove(email)
                added = True
        
        if not added and remaining:
            # Can't establish connection; add by date
            next_email = min(remaining, key=lambda e: e.date)
            thread.append(next_email)
            remaining.remove(next_email)
    
    return sorted(thread, key=lambda e: e.date)

# Example: Out-of-order emails
emails = [
    Email("msg3", "Re: Deploy Schedule", "carol@ex.com", ["alice@ex.com"], 
          datetime(2024, 1, 15, 12, 0), "Sounds good", in_reply_to="msg2"),
    Email("msg1", "Deploy Schedule", "alice@ex.com", ["bob@ex.com"], 
          datetime(2024, 1, 15, 9, 0), "Proposing Monday 3pm"),
    Email("msg2", "Re: Deploy Schedule", "bob@ex.com", ["alice@ex.com"], 
          datetime(2024, 1, 15, 10, 0), "Works for me", in_reply_to="msg1"),
]

ordered = reconstruct_thread(emails)
print([e.id for e in ordered])  # ['msg1', 'msg2', 'msg3']
```

**Trade-offs**: Header-based reconstruction is reliable but may fail with broken email clients. Subject-based fallback works 80% of the time but can merge unrelated threads with similar subjects. Always include a confidence score in production systems.

### 2. Structured Output Formatting

LLMs produce unstructured text by default. For programmatic use, you need structured outputs: JSON with specific fields.

**Technical Explanation**: Use explicit output schemas in prompts or use function calling/structured output APIs. Schemas enforce consistency and enable downstream processing (filtering, sorting, database insertion).

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic
import json

class ActionItem(BaseModel):
    task: str = Field(description="What needs to be done")
    owner: Optional[str] = Field(description="Who is responsible")
    deadline: Optional[str] = Field(description="When it's due")
    status: str = Field(description="open, completed, or blocked")

class Decision(BaseModel):
    decision: str = Field(description="What was decided")
    made_by: Optional[str] = Field(description="Who made the decision")
    timestamp: Optional[str] = Field(description="When it was made")

class ThreadSummary(BaseModel):
    main_topic: str
    outcome: str
    decisions: List[Decision]
    action_items: List[ActionItem]
    unresolved_questions: List[str]
    participants: List[str]

def summarize_with_structure(
    emails: List[Email], 
    api_key: str
) -> ThreadSummary:
    """Generate structured summary using schema"""
    client = Anthropic(api_key=api_key)
    
    thread_text = "\n\n---\n\n".join([
        f"From: {e.from_addr}\n"
        f"Date: {e.date.isoformat()}\n"
        f"To: {', '.join(e.to_addrs)}\n"
        f"Subject: {e.subject}\n\n{e.body}"
        for e in emails
    ])
    
    schema = ThreadSummary.model_json_schema()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Analyze this email thread and extract information according to this schema:

{json.dumps(schema, indent=2)}

Return only valid JSON matching this schema.

Email thread:
{thread_text}"""
        }]
    )
    
    # Parse and validate response
    response_text = response.content[0].text
    
    # Extract JSON if wrapped in markdown
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    
    summary_dict = json.loads(response_text)
    return ThreadSummary(**summary_dict)

# Usage
summary = summarize_with_structure(ordered, api_key="your-key")
print(f"Topic: {summary.main_topic}")
print(f"Action items: {len(summary.action_items)}")
for item in summary.action_items:
    print(f"  - {item.task} (Owner: {item.owner}, Due: {item.deadline})")
```

**Practical Implications**: Structured outputs enable automated workflows: create tickets from action items, update dashboards with decisions, trigger alerts for unresolved questions.

**Constraints**: Schema complexity affects accuracy. Keep schemas simple (≤10 fields) and provide clear descriptions. LLMs may hallucinate values to fit schema—always validate critical fields.

### 3. Context Window Management

Long email threads exceed LLM context windows. You must intelligently select or compress content.

**Technical Explanation**: Context windows range from 8K to 200K tokens. A 50-email thread with attachments can be 100K+ tokens. Strategies: progressive summarization (summarize in chunks), importance-based sampling (keep key messages), or dynamic truncation (remove boilerplate).

```python
from typing import List
import tiktoken

def count_tokens(text: str, model: str = "claude-3-5-sonnet-20241022") -> int:
    """Estimate token count (approximation for Claude)"""
    # Using GPT tokenizer as approximation
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def compress_thread(
    emails: List[Email], 
    max_tokens: int = 100000
) -> str:
    """Compress thread to fit context window"""
    # Strategy 1: Remove quoted text and signatures
    def strip_boilerplate(body: str) -> str:
        lines = body.split('\n')
        cleaned = []
        for line in lines:
            # Skip quoted lines
            if line.startswith('>'):
                continue
            # Skip signatures
            if line.startswith('--') or 'Sent from my' in line:
                break
            cleaned.append(line)
        return '\n'.join(cleaned).strip()
    
    # Strategy 2: Prioritize recent and first messages
    important = []
    if len(emails) > 0:
        important.append(emails[0])  # First message
    if len(emails) > 2:
        important.extend(emails[-2:])  # Last two messages
    if len(emails) > 5:
        important.extend(emails[len(emails)//2:len(emails)//2+1])  # Middle
    
    important = sorted(set(important), key=lambda e: e.date)
    
    # Build compressed thread
    compressed = []
    current_tokens = 0
    
    for email in important:
        body = strip_boilerplate(email.body)
        email_text = f"From: {email.from_addr}\nDate: {email.date}\n\n{body}"
        email_tokens = count_tokens(email_text)
        
        if current_tokens + email_tokens > max_tokens:
            # Truncate this email
            available = max_tokens - current_tokens
            truncated = email_text[:available * 4]  # Rough char to token ratio
            compressed.append(truncated + "\n[truncated]")
            break
        
        compressed.append(email_text)
        current_tokens += email_tokens
    
    return "\n\n---\n\n".join(compressed)

# Usage example
long_thread = [
    Email(f"msg{i}", "Re: Long Discussion", f"user{i}@ex.com", [], 
          datetime(2024, 1, 15, 9+i, 0), 
          f"{'Message content ' * 100}")  # Simulate long messages
    for i in range(50)
]

compressed = compress_thread(long_thread, max_tokens=10000)
print(f"Compressed from {count_tokens(str(long_thread))} to {count_tokens(compressed)} tokens")
```

**Trade-offs**: Compression loses information. Progressive summarization preserves more context but requires multiple LLM calls (higher latency and cost). Importance sampling is fast but may miss critical middle messages. Choose based on thread length distribution in your data.

### 4. Multi-Lingual and Encoding Handling

Email threads often contain multiple languages