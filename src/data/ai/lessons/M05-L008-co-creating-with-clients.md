# Co-Creating with Clients: Engineering Interactive AI Systems

## Core Concepts

Co-creating with AI clients means building systems where humans and language models iterate together through conversation to solve problems. Unlike traditional APIs that return deterministic results from fixed inputs, or simple chatbots that answer isolated questions, co-creation involves progressive refinement where each exchange builds on previous context to achieve increasingly precise outcomes.

### Traditional vs. Co-Creative Approach

```python
# Traditional API approach: One-shot execution
def generate_report_traditional(data: dict) -> str:
    """Generate report from structured data - no iteration possible"""
    template = load_template("quarterly_report.txt")
    return template.format(**data)

# Result: Fixed output, no refinement, rigid structure
report = generate_report_traditional({"revenue": 1000000, "growth": 15})
# If output is wrong, you modify code and re-run entire pipeline
```

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str

class CoCreativeSession:
    """Co-creative approach: Iterative refinement through conversation"""
    
    def __init__(self, system_prompt: str):
        self.messages: List[Message] = [
            Message(role="system", content=system_prompt)
        ]
        self.artifacts: Dict[str, any] = {}
    
    def iterate(self, user_input: str, llm_client) -> str:
        """Each iteration builds on complete conversation history"""
        self.messages.append(Message(role="user", content=user_input))
        
        response = llm_client.generate(
            messages=[{"role": m.role, "content": m.content} 
                     for m in self.messages]
        )
        
        self.messages.append(Message(role="assistant", content=response))
        return response
    
    def extract_artifact(self, key: str, content: str):
        """Store intermediate work products for reference"""
        self.artifacts[key] = content

# Result: Progressive refinement toward precise solution
session = CoCreativeSession(
    "You're helping create a quarterly report. Ask clarifying questions."
)

# Iteration 1: AI asks questions
response1 = session.iterate("Create a Q4 report", llm)
# "What metrics should I focus on? Revenue, users, or operational efficiency?"

# Iteration 2: User provides direction
response2 = session.iterate("Focus on revenue trends and growth drivers", llm)
# AI generates initial draft with specific focus

# Iteration 3: Refinement based on draft
response3 = session.iterate("Add comparison to industry benchmarks", llm)
# AI enhances draft with context-aware additions
```

The fundamental difference: traditional approaches require complete, correct specifications upfront. Co-creation acknowledges that requirements emerge through exploration, and builds systems that facilitate this discovery process.

### Key Engineering Insights

**Statefulness is Central**: Unlike REST APIs designed to be stateless, co-creative systems require careful state management. The conversation history IS the state, and every design decision—from context window management to artifact storage—revolves around maintaining coherent state across interactions.

**Ambiguity as a Feature**: Traditional software engineering minimizes ambiguity through explicit schemas and validation. Co-creative systems leverage ambiguity strategically—allowing users to express vague intents that get progressively clarified. This requires engineering systems that can detect ambiguity, ask clarifying questions, and track resolution.

**Non-Linear Workflows**: Users don't follow predetermined paths. They jump between refinement, exploration, and backtracking. Your system architecture must support branching, versioning, and non-sequential modification of previous work without losing coherence.

### Why This Matters Now

LLMs have crossed a capability threshold where they can maintain coherent context over extended interactions and perform complex reasoning. This creates an inflection point: for the first time, we can build systems where specification and implementation happen simultaneously through conversation rather than sequential phases.

For engineers, this means:
- **New architecture patterns**: Session management, context pruning, and artifact persistence become first-class concerns
- **Different quality metrics**: Success isn't just accuracy—it's convergence speed, user effort required, and graceful handling of ambiguity
- **Changed user expectations**: Users now expect iterative refinement capabilities in AI systems, not one-shot responses

## Technical Components

### 1. Conversation State Management

The conversation history determines all subsequent model behavior. Poor state management leads to context loss, repetitive responses, or degraded quality as conversations extend.

**Technical Explanation**: Each LLM call requires the complete conversation history (within context limits). State management involves: message array maintenance, token counting, strategic pruning when approaching limits, and persistence between sessions.

**Practical Implications**:

```python
from typing import List, Optional
import json

class ConversationStateManager:
    """Manages conversation history with automatic pruning"""
    
    def __init__(self, max_tokens: int = 4096, model_token_limit: int = 8192):
        self.messages: List[Dict[str, str]] = []
        self.max_tokens = max_tokens
        self.model_token_limit = model_token_limit
        self.system_prompt: Optional[str] = None
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation: 1 token ≈ 4 characters"""
        return len(text) // 4
    
    def add_message(self, role: str, content: str):
        """Add message with automatic pruning if needed"""
        new_message = {"role": role, "content": content}
        self.messages.append(new_message)
        
        total_tokens = sum(self.estimate_tokens(m["content"]) 
                          for m in self.messages)
        
        if total_tokens > self.max_tokens:
            self._prune_messages()
    
    def _prune_messages(self):
        """Remove oldest non-system messages, keep recent context"""
        # Always preserve system message and last N messages
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        non_system = [m for m in self.messages if m["role"] != "system"]
        
        # Keep system + most recent messages that fit in budget
        preserved = system_msgs + non_system[-10:]  # Keep last 10 exchanges
        
        # Add summary of pruned content
        if len(non_system) > 10:
            summary = {
                "role": "system",
                "content": f"[Previous context pruned. Conversation started with "
                          f"discussion about: {non_system[0]['content'][:100]}...]"
            }
            self.messages = system_msgs + [summary] + non_system[-10:]
        else:
            self.messages = preserved
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Return messages formatted for LLM API"""
        return self.messages.copy()
    
    def save_state(self, filepath: str):
        """Persist conversation for resumption"""
        with open(filepath, 'w') as f:
            json.dump({
                "messages": self.messages,
                "max_tokens": self.max_tokens
            }, f, indent=2)
    
    @classmethod
    def load_state(cls, filepath: str) -> 'ConversationStateManager':
        """Resume conversation from saved state"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        manager = cls(max_tokens=data["max_tokens"])
        manager.messages = data["messages"]
        return manager
```

**Real Constraints**: 
- Token estimation is approximate; actual tokenization varies by model
- Aggressive pruning loses context; conservative pruning hits token limits
- Summary injection can confuse models if not carefully worded

**Concrete Example**: In a code review assistant, preserving the original code snippet in context is critical even after 20 messages of discussion. Tag critical messages as "pinned" and exclude from pruning.

### 2. Artifact Extraction and Version Control

As conversations progress, they produce intermediate artifacts (code, documents, configurations). These artifacts need structured storage separate from conversation flow for retrieval, versioning, and modification.

**Technical Explanation**: Parse model outputs to identify structured content (code blocks, JSON, formatted sections), extract into typed storage, maintain versions as artifacts evolve through iterations.

```python
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class Artifact:
    """Represents a versioned work product"""
    id: str
    type: str  # "code", "document", "config"
    content: str
    version: int
    created_at: datetime
    parent_version: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

class ArtifactManager:
    """Extracts and versions artifacts from conversation"""
    
    def __init__(self):
        self.artifacts: Dict[str, List[Artifact]] = {}
    
    def extract_from_response(self, response: str, 
                             artifact_id: str) -> List[Artifact]:
        """Parse response for structured content"""
        extracted = []
        
        # Extract code blocks
        code_pattern = r'```(\w+)?\n(.*?)```'
        for match in re.finditer(code_pattern, response, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            
            artifact = self._create_or_update_artifact(
                artifact_id=f"{artifact_id}_code",
                artifact_type="code",
                content=code,
                metadata={"language": language}
            )
            extracted.append(artifact)
        
        # Extract JSON configurations
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, response):
            try:
                json.loads(match.group(0))  # Validate JSON
                artifact = self._create_or_update_artifact(
                    artifact_id=f"{artifact_id}_config",
                    artifact_type="config",
                    content=match.group(0)
                )
                extracted.append(artifact)
            except json.JSONDecodeError:
                continue
        
        return extracted
    
    def _create_or_update_artifact(self, artifact_id: str, 
                                   artifact_type: str,
                                   content: str,
                                   metadata: Dict = None) -> Artifact:
        """Create new version or new artifact"""
        if artifact_id not in self.artifacts:
            self.artifacts[artifact_id] = []
            version = 1
            parent = None
        else:
            parent = self.artifacts[artifact_id][-1].version
            version = parent + 1
        
        artifact = Artifact(
            id=artifact_id,
            type=artifact_type,
            content=content,
            version=version,
            created_at=datetime.now(),
            parent_version=parent,
            metadata=metadata or {}
        )
        
        self.artifacts[artifact_id].append(artifact)
        return artifact
    
    def get_latest(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve most recent version"""
        if artifact_id not in self.artifacts:
            return None
        return self.artifacts[artifact_id][-1]
    
    def get_version(self, artifact_id: str, version: int) -> Optional[Artifact]:
        """Retrieve specific version"""
        if artifact_id not in self.artifacts:
            return None
        
        for artifact in self.artifacts[artifact_id]:
            if artifact.version == version:
                return artifact
        return None
    
    def diff_versions(self, artifact_id: str, 
                     v1: int, v2: int) -> Optional[str]:
        """Generate diff between versions"""
        artifact_v1 = self.get_version(artifact_id, v1)
        artifact_v2 = self.get_version(artifact_id, v2)
        
        if not artifact_v1 or not artifact_v2:
            return None
        
        # Simple line-by-line diff
        lines_v1 = artifact_v1.content.split('\n')
        lines_v2 = artifact_v2.content.split('\n')
        
        diff_lines = []
        for i, (line1, line2) in enumerate(zip(lines_v1, lines_v2)):
            if line1 != line2:
                diff_lines.append(f"Line {i+1}:")
                diff_lines.append(f"- {line1}")
                diff_lines.append(f"+ {line2}")
        
        return '\n'.join(diff_lines)
```

**Real Constraints**:
- Regex-based extraction is fragile; models may format inconsistently
- Version history grows unbounded; implement retention policies
- Diffing large artifacts is computationally expensive

### 3. Guided Iteration Protocols

Co-creation requires protocols that guide the model through multi-step processes: asking clarifying questions, proposing alternatives, validating constraints, and tracking completion.

**Technical Explanation**: Embed structured instructions in system prompts that define interaction patterns. Use explicit state tracking and templated prompts to enforce protocols across conversation turns.

```python
from enum import Enum
from typing import Callable, Optional

class IterationPhase(Enum):
    """Phases in guided iteration protocol"""
    REQUIREMENTS_GATHERING = "gathering"
    PROPOSAL = "proposal"
    REFINEMENT = "refinement"
    VALIDATION = "validation"
    COMPLETE = "complete"

class GuidedIterationProtocol:
    """Enforces structured co-creation workflow"""
    
    def __init__(self, goal: str):
        self.goal = goal
        self.phase = IterationPhase.REQUIREMENTS_GATHERING
        self.requirements: Dict[str, any] = {}
        self.proposals: List[str] = []
        self.validations: List[bool] = []
    
    def get_system_prompt(self) -> str:
        """Generate phase-specific system prompt"""
        base = f"You are helping achieve this goal: {self.goal}\n\n"
        
        if self.phase == IterationPhase.REQUIREMENTS_GATHERING:
            return base + """Current phase: REQUIREMENTS GATHERING
Your task:
1. Ask 2-3 specific questions to clarify requirements
2. Focus on constraints, priorities, and success criteria
3. Do NOT propose solutions yet
4. End with: "Once you answer these, I'll propose solutions."
"""
        
        elif self.phase == IterationPhase.PROPOSAL:
            reqs = '\n'.join(f"- {k}: {v}" for k, v in self.requirements.items())
            return base + f"""Current phase: SOLUTION PROPOSAL
Requirements gathered:
{reqs}

Your task:
1. Propose 2-3 distinct approaches
2. For each, explain tradeoffs and when it's appropriate
3. Recommend one with justification
4. Ask which approach to develop further
"""
        
        elif self.phase == IterationPhase.REFINEMENT:
            return base + f"""Current phase: REFINEMENT
Selected approach: {self.proposals[-1]}

Your task:
1. Implement the selected approach with complete details
2. Include code, configuration, or concrete specifications
3. Explain any assumptions made
4. Ask for specific feedback on the implementation
"""
        
        elif self.phase == IterationPhase.VALIDATION:
            return base + """Current phase: VALIDATION
Your task:
1. Review implementation against original requirements
2. Identify any gaps or issues
3. Propose specific fixes if needed
4. Confirm readiness or continue refinement
"""
        
        return base + "Goal achieved. Summarize deliverables."
    
    def process_user_input(self, user_input: str) -> Optional[str]:
        """Update protocol state based on user input"""
        
        if self.phase == IterationPhase.REQUIREMENTS_GATHERING:
            # Parse requirements from user response
            # (In production, use structured extraction)
            self.requirements["user_input"] = user_input
            self.phase = IterationPhase.PROPOSAL
            return "Requirements captured. Moving to proposal phase."
        
        elif self.phase == IterationPhase.PROPOSAL:
            # User selects approach
            self.proposals.append(user_input)
            self.phase = IterationPhase.REFINEMENT
            return f"Developing approach: {user_input}"
        
        elif