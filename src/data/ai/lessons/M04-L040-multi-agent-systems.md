# Multi-Agent Systems: Coordinating Multiple LLMs for Complex Problem Solving

## Core Concepts

Multi-agent systems in the LLM context involve multiple AI components—each with specialized roles, knowledge, or capabilities—working together to solve problems that are difficult or inefficient for a single model to handle alone. Unlike monolithic prompt chains, multi-agent architectures distribute cognition across specialized agents that communicate, delegate, and coordinate to achieve complex objectives.

### Traditional vs. Multi-Agent Approaches

**Traditional Single-Agent Approach:**

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

def analyze_complex_document(document: str) -> dict:
    """Single model handles everything - analysis, summarization, extraction"""
    prompt = f"""
    Analyze this document and provide:
    1. A summary
    2. Key entities (people, organizations, dates)
    3. Sentiment analysis
    4. Action items
    5. Risk assessment
    
    Document: {document}
    """
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Single model juggles all tasks - quality varies across tasks
    return {"analysis": response.content[0].text}
```

**Multi-Agent Approach:**

```python
from typing import Protocol, List
from dataclasses import dataclass
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: str
    metadata: dict

class Agent(Protocol):
    name: str
    specialty: str
    
    def process(self, message: str, context: dict) -> str:
        """Process input and return specialized output"""
        ...

class SummarizerAgent:
    def __init__(self):
        self.name = "summarizer"
        self.specialty = "Create concise, accurate summaries"
    
    def process(self, document: str, context: dict) -> str:
        prompt = f"""You are a specialized summarization agent.
        Create a concise summary focusing on main points and conclusions.
        
        Document: {document}"""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class EntityExtractorAgent:
    def __init__(self):
        self.name = "entity_extractor"
        self.specialty = "Extract structured entities"
    
    def process(self, document: str, context: dict) -> str:
        prompt = f"""You are a specialized entity extraction agent.
        Extract people, organizations, dates, and locations in JSON format.
        
        Document: {document}"""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

class CoordinatorAgent:
    def __init__(self, agents: List[Agent]):
        self.name = "coordinator"
        self.agents = {agent.name: agent for agent in agents}
    
    def orchestrate(self, document: str) -> dict:
        """Coordinate specialized agents and synthesize results"""
        results = {}
        
        # Delegate to specialized agents
        for agent_name, agent in self.agents.items():
            results[agent_name] = agent.process(document, context=results)
        
        # Synthesize final analysis
        synthesis_prompt = f"""Synthesize these specialized analyses:
        
        Summary: {results.get('summarizer', '')}
        Entities: {results.get('entity_extractor', '')}
        
        Provide integrated insights and recommendations."""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        results['synthesis'] = response.content[0].text
        return results

# Usage
agents = [SummarizerAgent(), EntityExtractorAgent()]
coordinator = CoordinatorAgent(agents)
results = coordinator.orchestrate(document)
```

### Key Engineering Insights

**1. Specialization Improves Quality**: A model prompted to be a "summarization expert" produces better summaries than a generalist model juggling multiple tasks. This isn't just prompt engineering—it's cognitive load distribution.

**2. Communication Overhead Is Real**: Each agent interaction adds latency and cost. A five-agent system with sequential dependencies might take 5x longer than a single call. Design for parallelization where possible.

**3. Emergent Behavior Through Interaction**: Multiple agents debating, critiquing, or refining each other's outputs often produce superior results to any single agent—but this requires careful orchestration to avoid circular reasoning or drift.

**4. State Management Becomes Critical**: Unlike single calls, multi-agent systems must track conversation history, intermediate results, and decision paths. Poor state management leads to context loss and degraded performance.

### Why This Matters Now

**Context Window Limitations**: Even with 200K+ token context windows, structuring work across specialized agents can be more effective than cramming everything into a single prompt. Specialization beats comprehensiveness.

**Cost-Performance Trade-offs**: Run cheap, fast models for simple subtasks (classification, extraction) and expensive, powerful models only for complex reasoning. Multi-agent architectures enable granular cost optimization.

**Reliability Through Redundancy**: Critical systems can employ multiple agents as validators, fact-checkers, or parallel problem-solvers, dramatically improving reliability over single-agent approaches.

**Scaling Complex Workflows**: As AI systems move from demos to production, handling multi-step workflows with error recovery, conditional branching, and human-in-the-loop becomes essential—multi-agent patterns provide the architecture.

## Technical Components

### 1. Agent Roles and Specialization

Each agent should have a clearly defined role, specialty, and scope. Avoid generic "assistant" agents—specific roles produce better results.

**Technical Explanation**: Agent specialization works because focused system prompts create stronger priors for specific tasks. A model prompted as a "Python security auditor" will apply security-focused attention patterns differently than a general coding assistant.

**Practical Implementation**:

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field

class AgentRole(BaseModel):
    name: str = Field(description="Unique agent identifier")
    specialty: str = Field(description="Core competency")
    system_prompt: str = Field(description="Defines agent behavior")
    model: str = Field(default="claude-sonnet-4-20250514")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    
class SpecializedAgent:
    def __init__(self, role: AgentRole):
        self.role = role
        self.client = Anthropic(api_key="your-api-key")
        self.conversation_history: List[dict] = []
    
    def process(self, user_message: str, context: Optional[dict] = None) -> str:
        """Process message with agent's specialized perspective"""
        messages = self.conversation_history + [
            {"role": "user", "content": user_message}
        ]
        
        response = self.client.messages.create(
            model=self.role.model,
            max_tokens=2000,
            temperature=self.role.temperature,
            system=self.role.system_prompt,
            messages=messages
        )
        
        assistant_message = response.content[0].text
        
        # Maintain conversation context
        self.conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ])
        
        return assistant_message
    
    def reset(self):
        """Clear conversation history"""
        self.conversation_history = []

# Example specialized agents
code_reviewer = SpecializedAgent(AgentRole(
    name="code_reviewer",
    specialty="Code quality and security review",
    system_prompt="""You are an expert code reviewer focusing on:
    - Security vulnerabilities (injection, auth issues, data exposure)
    - Performance bottlenecks
    - Code maintainability and patterns
    Provide specific, actionable feedback with line references.""",
    temperature=0.3  # Lower temperature for consistent reviews
))

test_generator = SpecializedAgent(AgentRole(
    name="test_generator",
    specialty="Comprehensive test case generation",
    system_prompt="""You are a test engineering specialist. Generate:
    - Unit tests covering edge cases
    - Integration test scenarios
    - Property-based test strategies
    Provide complete, runnable test code.""",
    temperature=0.5
))

documentation_writer = SpecializedAgent(AgentRole(
    name="documentation_writer",
    specialty="Clear technical documentation",
    system_prompt="""You are a technical writer specializing in API documentation.
    Create clear, comprehensive docs with:
    - Usage examples
    - Parameter descriptions
    - Error scenarios
    - Performance considerations""",
    temperature=0.6
))
```

**Real Constraints**: Specialization has diminishing returns. Beyond 5-7 distinct roles, coordination overhead typically exceeds benefits. Start with 2-3 core agents and expand only when specific quality gaps emerge.

### 2. Communication Protocols

Agents need structured ways to exchange information. Ad-hoc string passing leads to context loss and parsing errors.

**Technical Explanation**: Well-defined message schemas enable agents to understand context, track conversation flow, and make better decisions about when and how to respond.

```python
from enum import Enum
from datetime import datetime
from typing import Any, Dict
import json

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUESTION = "question"
    ANSWER = "answer"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"

class InterAgentMessage(BaseModel):
    msg_id: str = Field(default_factory=lambda: str(datetime.utcnow().timestamp()))
    msg_type: MessageType
    sender: str
    recipient: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_response: bool = False
    in_response_to: Optional[str] = None
    
    def to_context(self) -> str:
        """Format message for LLM context"""
        return f"""[{self.msg_type.value}] From {self.sender} to {self.recipient}:
{self.content}
{f'Metadata: {json.dumps(self.metadata)}' if self.metadata else ''}"""

class MessageBus:
    """Central message routing and history"""
    def __init__(self):
        self.messages: List[InterAgentMessage] = []
        self.agent_registry: Dict[str, SpecializedAgent] = {}
    
    def register_agent(self, agent: SpecializedAgent):
        self.agent_registry[agent.role.name] = agent
    
    def send(self, message: InterAgentMessage) -> Optional[str]:
        """Route message and return response if required"""
        self.messages.append(message)
        
        recipient = self.agent_registry.get(message.recipient)
        if not recipient:
            raise ValueError(f"Unknown recipient: {message.recipient}")
        
        # Build context from relevant message history
        context = self._build_context(message)
        
        # Process message
        response_content = recipient.process(
            user_message=message.content,
            context={"history": context, "metadata": message.metadata}
        )
        
        if message.requires_response:
            response = InterAgentMessage(
                msg_type=MessageType.TASK_RESPONSE,
                sender=message.recipient,
                recipient=message.sender,
                content=response_content,
                in_response_to=message.msg_id
            )
            self.messages.append(response)
            return response.content
        
        return None
    
    def _build_context(self, current_message: InterAgentMessage) -> str:
        """Build relevant conversation context"""
        relevant_messages = [
            msg for msg in self.messages[-10:]  # Last 10 messages
            if msg.sender == current_message.recipient or 
               msg.recipient == current_message.recipient
        ]
        return "\n\n".join(msg.to_context() for msg in relevant_messages)
    
    def get_conversation_thread(self, msg_id: str) -> List[InterAgentMessage]:
        """Retrieve full conversation thread"""
        thread = []
        current_id = msg_id
        
        while current_id:
            msg = next((m for m in self.messages if m.msg_id == current_id), None)
            if not msg:
                break
            thread.insert(0, msg)
            current_id = msg.in_response_to
        
        return thread

# Usage example
bus = MessageBus()
bus.register_agent(code_reviewer)
bus.register_agent(test_generator)

# Request code review
review_request = InterAgentMessage(
    msg_type=MessageType.TASK_REQUEST,
    sender="orchestrator",
    recipient="code_reviewer",
    content="Review this function for security issues:\n\n" + code_sample,
    requires_response=True,
    metadata={"priority": "high", "file": "api.py"}
)

review_result = bus.send(review_request)
```

**Practical Implications**: Message buses centralize communication, enabling features like conversation replay, debugging, and async processing. In production systems, this becomes your audit log and debugging tool.

**Trade-offs**: Structured protocols add code complexity. For simple 2-3 agent systems, direct function calls may suffice. Implement full message buses when you need conversation history, async processing, or debugging capabilities.

### 3. Orchestration Patterns

How agents coordinate determines system capability. Three primary patterns: sequential, parallel, and hierarchical.

**Sequential Pattern** (workflow-style):

```python
class SequentialOrchestrator:
    """Execute agents in defined order, passing outputs forward"""
    def __init__(self, agents: List[SpecializedAgent]):
        self.agents = agents
    
    def execute(self, initial_input: str) -> Dict[str, str]:
        """Run agents sequentially, each receiving previous output"""
        results = {"initial_input": initial_input}
        current_input = initial_input
        
        for agent in self.agents:
            agent_output = agent.process(current_input)
            results[agent.role.name] = agent_output
            
            # Next agent receives previous output plus context
            current_input = f"""Previous analysis:
{agent_output}

Original input:
{initial_input}

Continue the analysis from your specialized perspective."""
        
        return results

# Example: Document processing pipeline
pipeline = SequentialOrchestrator([
    SpecializedAgent(AgentRole(
        name="extractor",
        specialty="Extract structured data",
        system_prompt="Extract key facts and entities as JSON"
    )),
    SpecializedAgent(AgentRole(
        name="analyzer",
        specialty="Analyze extracted data",
        system_prompt="Analyze patterns and relationships in extracted data"
    )),
    SpecializedAgent(AgentRole(
        name="recommender",
        specialty="Generate recommendations",
        system_prompt="Provide actionable recommendations based on analysis"
    ))
])

results = pipeline.execute(document)
```

**Parallel Pattern** (fan-out, fan-in):

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

class ParallelOrchestrator:
    """Execute multiple agents simultaneously, then synthesize"""
    def __init__(self, agents: List[SpecializedAgent], synthesizer: SpecializedAgent):
        self.agents = agents
        self.synthesizer = synthesizer
    
    def execute(self, input_data: str) -> Dict[str, Any]:
        """Run agents in parallel, synthesize results"""
        # Execute all agents concurrently
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                agent.role.name: executor.submit(agent.process, input_data)
                for agent in self.agents
            }
            
            results = {
                name