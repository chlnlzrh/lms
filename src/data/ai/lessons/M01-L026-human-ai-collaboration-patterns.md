# Human-AI Collaboration Patterns

## Core Concepts

Human-AI collaboration is the engineering practice of designing workflows where human intelligence and AI capabilities combine to solve problems neither could handle as effectively alone. This isn't about AI replacing humans or humans supervising AI—it's about architecting systems where each contributes their strengths at the right moments.

### Traditional vs. Modern Approach

```python
# Traditional: Sequential human-only workflow
def analyze_customer_feedback_traditional(feedback_list: list[str]) -> dict:
    """Human reads all feedback, categorizes, and summarizes manually."""
    # Hours of manual work
    # Results in spreadsheet
    # Human does: reading, categorizing, summarizing, identifying patterns
    pass

# Human-AI Collaboration: Hybrid workflow
from typing import List, Dict
import anthropic

def analyze_customer_feedback_collaborative(
    feedback_list: List[str],
    client: anthropic.Client
) -> Dict:
    """AI handles scale and pattern detection; human handles judgment and strategy."""
    
    # AI: Rapid categorization and initial analysis (seconds for 1000s of items)
    prompt = f"""Analyze this customer feedback and provide:
1. Categories with counts
2. Sentiment distribution
3. Top 5 issues by frequency
4. Unusual patterns that need human attention

Feedback:
{chr(10).join(feedback_list[:100])}  # First 100 items
"""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    ai_analysis = message.content[0].text
    
    # Human: Strategic decisions based on AI synthesis
    print(f"AI Analysis:\n{ai_analysis}\n")
    print("Now review AI findings and make strategic decisions:")
    print("- Which issues need immediate product changes?")
    print("- Which patterns suggest market opportunities?")
    print("- What requires deeper investigation?")
    
    return {
        "ai_analysis": ai_analysis,
        "feedback_count": len(feedback_list),
        "human_decision_needed": True
    }

# Result: 100x faster initial analysis, human time spent on high-value decisions
```

### Key Engineering Insights

**1. Complementary Strengths, Not Substitution**

AI excels at: pattern recognition across large datasets, consistent application of rules, rapid iteration, 24/7 availability, handling repetitive cognitive tasks.

Humans excel at: contextual judgment, ethical reasoning, creative problem-solving, handling ambiguity, strategic decision-making, understanding unstated constraints.

**2. The Interface Is the Architecture**

The quality of human-AI collaboration depends entirely on how you design the handoff points. Poor interfaces create friction; good ones create flow.

```python
# Poor interface: Opaque AI output
def poor_collaboration(data: str) -> str:
    result = ai_model.generate(data)
    return result  # Human gets raw output, no context

# Good interface: Structured output with confidence and reasoning
from typing import TypedDict

class AIRecommendation(TypedDict):
    recommendation: str
    confidence: float
    reasoning: str
    alternatives: List[str]
    human_review_needed: bool

def good_collaboration(data: str) -> AIRecommendation:
    result = ai_model.generate(data)
    return {
        "recommendation": result.primary,
        "confidence": result.confidence_score,
        "reasoning": result.explanation,
        "alternatives": result.other_options,
        "human_review_needed": result.confidence_score < 0.8
    }
```

**3. Collaboration Patterns Are Workflow Primitives**

Just as we have design patterns in software (Factory, Observer, Strategy), human-AI collaboration has its own patterns: AI-first with human verification, human-first with AI augmentation, parallel processing with human synthesis, iterative refinement, human-in-the-loop training.

### Why This Matters Now

Current AI systems (2024-2025) hit a "capability ceiling" on complex tasks when operating autonomously. GPT-4 class models achieve 60-80% accuracy on many real-world tasks alone, but 90-98% accuracy with appropriate human collaboration. This isn't a limitation—it's an architectural opportunity. Engineers who master collaboration patterns build systems that are 2-5x more effective than those who treat AI as either a black box or a complete solution.

## Technical Components

### 1. Task Decomposition & Assignment

**Technical Explanation:** Breaking complex workflows into subtasks and assigning each to human or AI based on capability matching. This requires analyzing task characteristics: scale (how many items?), complexity (how many variables?), ambiguity (how clear are the rules?), and stakes (what's the cost of errors?).

**Practical Implementation:**

```python
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any

class TaskType(Enum):
    AI_AUTONOMOUS = "ai_autonomous"  # AI handles completely
    AI_FIRST_HUMAN_VERIFY = "ai_first_human_verify"  # AI proposes, human approves
    HUMAN_FIRST_AI_AUGMENT = "human_first_ai_augment"  # Human decides, AI assists
    HUMAN_AUTONOMOUS = "human_autonomous"  # Human handles completely

@dataclass
class TaskCharacteristics:
    scale: int  # Number of items to process
    complexity: int  # 1-10 scale
    ambiguity: int  # 1-10 scale
    error_cost: int  # 1-10 scale

def assign_task_type(chars: TaskCharacteristics) -> TaskType:
    """Determine optimal human-AI collaboration pattern."""
    
    # High scale, low complexity/ambiguity, low stakes -> AI autonomous
    if chars.scale > 100 and chars.complexity < 4 and chars.error_cost < 3:
        return TaskType.AI_AUTONOMOUS
    
    # High stakes or high ambiguity -> Human in loop
    if chars.error_cost > 7 or chars.ambiguity > 7:
        return TaskType.HUMAN_FIRST_AI_AUGMENT
    
    # Moderate stakes, AI can handle but needs verification
    if chars.complexity < 7 and chars.error_cost < 7:
        return TaskType.AI_FIRST_HUMAN_VERIFY
    
    # High complexity, low scale -> Human autonomous
    return TaskType.HUMAN_AUTONOMOUS

# Example usage
content_moderation = TaskCharacteristics(
    scale=10000,
    complexity=5,
    ambiguity=6,
    error_cost=8  # Removing legitimate content is costly
)

print(assign_task_type(content_moderation))  
# Output: TaskType.AI_FIRST_HUMAN_VERIFY
```

**Real Constraints:** Task decomposition overhead. Breaking tasks incorrectly creates inefficiency. A 100-item task that takes 1 minute per item (100 minutes) shouldn't be decomposed if decomposition adds 30 minutes of overhead. Measure end-to-end time, not just processing time.

**Trade-offs:** Fine-grained decomposition increases flexibility but adds coordination cost. Coarse-grained decomposition is simpler but may not optimize each subtask.

### 2. Confidence Scoring & Escalation Thresholds

**Technical Explanation:** AI systems should expose uncertainty metrics that trigger human review. Rather than treating all AI outputs equally, implement tiered confidence levels that route low-confidence items to humans.

**Practical Implementation:**

```python
from typing import Optional, List
import json

class CollaborativeClassifier:
    def __init__(self, ai_client, confidence_threshold: float = 0.85):
        self.client = ai_client
        self.threshold = confidence_threshold
        self.human_review_queue: List[dict] = []
    
    def classify_with_escalation(self, item: str) -> dict:
        """Classify item; escalate to human if confidence is low."""
        
        prompt = f"""Classify this item into one of: [urgent, normal, low_priority].
        
Provide your response as JSON:
{{
    "classification": "your_classification",
    "confidence": 0.0-1.0,
    "reasoning": "why you chose this",
    "ambiguity_factors": ["list", "of", "unclear", "aspects"]
}}

Item: {item}"""
        
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = json.loads(message.content[0].text)
        
        if result["confidence"] < self.threshold:
            # Escalate to human review
            self.human_review_queue.append({
                "item": item,
                "ai_suggestion": result["classification"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "ambiguity": result["ambiguity_factors"]
            })
            return {
                "status": "escalated",
                "ai_suggestion": result,
                "requires_human": True
            }
        else:
            # AI handles autonomously
            return {
                "status": "resolved",
                "classification": result["classification"],
                "confidence": result["confidence"],
                "requires_human": False
            }
    
    def get_human_review_queue(self) -> List[dict]:
        """Fetch items needing human review, sorted by urgency."""
        return sorted(
            self.human_review_queue,
            key=lambda x: x["confidence"]  # Lowest confidence first
        )

# Example usage
# classifier = CollaborativeClassifier(client, confidence_threshold=0.85)
# result = classifier.classify_with_escalation("Customer wants refund for damaged item")
```

**Real Constraints:** LLMs don't have true probability distributions. "Confidence" is often a post-hoc assessment, not a statistical measure. Test calibration: do 80% confidence predictions succeed 80% of the time? Measure this on your specific use case.

**Trade-offs:** Lower thresholds (0.7) mean more human review but higher accuracy. Higher thresholds (0.95) mean less human review but more errors slip through. Optimize based on error cost vs. human time cost.

### 3. Feedback Loops & Iterative Refinement

**Technical Explanation:** Human corrections should improve AI performance over time. This doesn't necessarily mean retraining models (often impractical), but rather refining prompts, building few-shot examples, or creating guardrails based on human feedback.

**Practical Implementation:**

```python
from typing import List, Tuple
from datetime import datetime
import json

class LearningCollaborationSystem:
    def __init__(self, ai_client):
        self.client = ai_client
        self.corrections: List[dict] = []
        self.few_shot_examples: List[Tuple[str, str]] = []
    
    def process_with_learning(self, task: str) -> dict:
        """Process task using accumulated learning from corrections."""
        
        # Build prompt with few-shot examples from human corrections
        few_shot_section = ""
        if self.few_shot_examples:
            few_shot_section = "Learn from these examples:\n"
            for input_ex, output_ex in self.few_shot_examples[-5:]:  # Last 5
                few_shot_section += f"Input: {input_ex}\nCorrect Output: {output_ex}\n\n"
        
        prompt = f"""{few_shot_section}
Now process this task:
{task}"""
        
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "output": message.content[0].text,
            "timestamp": datetime.now().isoformat(),
            "examples_used": len(self.few_shot_examples)
        }
    
    def record_human_correction(
        self, 
        original_task: str,
        ai_output: str, 
        human_correction: str,
        correction_type: str
    ):
        """Record human correction to improve future performance."""
        
        self.corrections.append({
            "task": original_task,
            "ai_output": ai_output,
            "human_correction": human_correction,
            "type": correction_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add to few-shot examples if it's a clear correction
        if correction_type in ["wrong_format", "wrong_category", "missed_detail"]:
            self.few_shot_examples.append((original_task, human_correction))
    
    def get_correction_stats(self) -> dict:
        """Analyze correction patterns to identify systematic issues."""
        
        if not self.corrections:
            return {"total_corrections": 0}
        
        correction_types = {}
        for correction in self.corrections:
            ctype = correction["type"]
            correction_types[ctype] = correction_types.get(ctype, 0) + 1
        
        return {
            "total_corrections": len(self.corrections),
            "by_type": correction_types,
            "improvement_rate": self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate if error rate is decreasing over time."""
        if len(self.corrections) < 10:
            return 0.0
        
        # Compare first half vs second half
        midpoint = len(self.corrections) // 2
        first_half_rate = midpoint / midpoint  # Normalized
        second_half_rate = (len(self.corrections) - midpoint) / midpoint
        
        return (first_half_rate - second_half_rate) / first_half_rate

# Example usage showing improvement over time
# system = LearningCollaborationSystem(client)
# result1 = system.process_with_learning("Extract invoice number")
# system.record_human_correction(
#     "Extract invoice number",
#     result1["output"],
#     "INV-2024-001",
#     "wrong_format"
# )
# result2 = system.process_with_learning("Extract invoice number from similar doc")
# # result2 now benefits from the correction
```

**Real Constraints:** Few-shot learning has diminishing returns. Adding examples 1-5 dramatically improves performance. Examples 20-30 add minimal value. Context window limits how many examples you can include.

**Trade-offs:** Storing and using corrections adds complexity. Only worth it if you process similar tasks repeatedly (>50 times). For one-off tasks, the overhead exceeds the benefit.

### 4. Structured Communication Protocols

**Technical Explanation:** Define explicit formats for human-AI information exchange. Unstructured conversation is flexible but inefficient. Structured protocols reduce ambiguity and enable automation.

**Practical Implementation:**

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field

class HumanRequest(BaseModel):
    """Structured format for human requests to AI."""
    task_type: Literal["analyze", "generate", "classify", "extract"]
    input_data: str
    constraints: Optional[list[str]] = None
    output_format: Literal["json", "markdown", "plain_text"] = "plain_text"
    max_length: Optional[int] = None

class AIResponse(BaseModel):
    """Structured format for AI responses to human."""
    result: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    alternatives: Optional[list[str]] = None
    warnings: Optional[list[str]] = None
    human_review_suggested: bool

class HumanFeedback(BaseModel):
    """Structured format for human feedback on AI responses."""
    approved: bool
    corrections: Optional[str] = None
    feedback_type: Literal["format", "content", "reasoning", "complete_rewrite"]
    notes: Optional[str] = None

def collaborative_exchange(
    request: HumanRequest,
    client
) -> tuple[AIResponse, Optional[HumanFeedback]]:
    """Execute structured human-AI exchange."""
    
    # Build prompt from structured request
    prompt = f"""Task: {request.task_type}
Input: {request.input_data}
Constraints: {request.constraints or 'None'}
Output format: {request.output_format}
{"Max length: " + str(request.max