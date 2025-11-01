# Architecture Decision Records (ADRs) with AI

## Core Concepts

Architecture Decision Records (ADRs) document significant architectural decisions made during software development. They capture the context, decision, and consequences of choices that shape system design. Traditionally, ADRs have been manually crafted documents requiring deep technical knowledge, consensus-building skills, and significant time investment. AI transforms this process by accelerating research, generating structured documentation, and helping evaluate alternatives—but the engineering judgment remains human.

### Traditional vs. AI-Augmented ADR Creation

**Traditional Approach:**
```python
# Traditional ADR creation: manual research and documentation
def create_adr_traditional():
    """
    Process:
    1. Research technology options (2-4 hours)
    2. Draft ADR structure (30 min)
    3. Write context and analysis (1-2 hours)
    4. Review and revise (1 hour)
    Total: 5-8 hours per ADR
    """
    research_notes = manually_research_databases()
    trade_offs = manually_analyze_options(research_notes)
    draft = manually_write_adr(trade_offs)
    final = manually_revise_with_team(draft)
    return final  # 5-8 hours later
```

**AI-Augmented Approach:**
```python
from typing import List, Dict
import json

def create_adr_with_ai(
    decision_topic: str,
    constraints: List[str],
    current_context: str
) -> Dict[str, str]:
    """
    AI-augmented ADR creation process.
    
    Process:
    1. AI generates initial research summary (5 min)
    2. AI structures ADR with alternatives (10 min)
    3. Engineer validates and refines (30 min)
    4. Team reviews focused document (30 min)
    Total: 1-2 hours per ADR (3-4x faster)
    
    Args:
        decision_topic: The architectural decision to document
        constraints: Technical/business constraints
        current_context: Existing system architecture
    
    Returns:
        Structured ADR components
    """
    # AI accelerates research phase
    research_summary = ai_research_technologies(
        topic=decision_topic,
        constraints=constraints
    )
    
    # AI generates structured alternatives analysis
    alternatives = ai_generate_alternatives(
        research=research_summary,
        context=current_context
    )
    
    # AI drafts ADR structure (human reviews and refines)
    adr_draft = ai_structure_adr(
        topic=decision_topic,
        alternatives=alternatives,
        constraints=constraints
    )
    
    # Engineer applies judgment and domain knowledge
    validated_adr = engineer_validate_and_refine(adr_draft)
    
    return validated_adr
```

The key difference: AI handles information gathering and structure generation, while engineers focus on decision-making and validation. This shifts engineering time from documentation mechanics to critical thinking.

### Why ADRs with AI Matter Now

Three convergent factors make AI-augmented ADRs critical today:

1. **Decision Velocity**: Modern systems require architectural decisions weekly, not quarterly. AI maintains ADR quality while matching increased decision pace.

2. **Complexity Explosion**: Microservices, cloud providers, and data stores create combinatorial option spaces. AI efficiently explores alternatives humans might miss.

3. **Knowledge Distribution**: Distributed teams lack shared context. AI generates comprehensive ADRs that transfer knowledge effectively across time zones and team boundaries.

Engineers who master AI-augmented ADRs make better decisions faster, document knowledge that survives team changes, and avoid costly architectural mistakes.

## Technical Components

### 1. Structured Prompting for ADR Generation

AI generates useful ADRs when given structured prompts that mirror standard ADR formats. Poor prompts yield generic content; structured prompts yield actionable documentation.

**Technical Implementation:**

```python
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ADRStatus(Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"

@dataclass
class ADRPromptTemplate:
    """Structured template for generating ADR content with AI."""
    
    title: str
    context: str
    constraints: List[str]
    alternatives: List[str]
    decision_criteria: List[str]
    status: ADRStatus = ADRStatus.PROPOSED
    
    def to_prompt(self) -> str:
        """Convert template to AI prompt."""
        return f"""Generate an Architecture Decision Record with the following structure:

# ADR: {self.title}

## Status
{self.status.value}

## Context
{self.context}

System Constraints:
{self._format_list(self.constraints)}

## Decision Criteria
Evaluate alternatives based on:
{self._format_list(self.decision_criteria)}

## Alternatives Considered
{self._format_list(self.alternatives)}

For each alternative, provide:
1. Technical approach and implementation details
2. Pros (specific technical benefits)
3. Cons (specific technical limitations)
4. Performance characteristics (latency, throughput, resource usage)
5. Operational complexity (deployment, monitoring, maintenance)

## Recommendation
Based on the criteria, provide a specific recommendation with:
1. Which alternative to choose
2. Technical justification
3. Expected trade-offs
4. Implementation risks

## Consequences
List concrete consequences of this decision:
1. Positive consequences (what improves)
2. Negative consequences (what gets harder)
3. Neutral consequences (what changes)
4. Risks and mitigation strategies
"""
    
    def _format_list(self, items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items)


# Example usage
adr_prompt = ADRPromptTemplate(
    title="Select Database for Event Sourcing System",
    context="""
We're building an event-sourced order processing system handling 10K orders/day.
Current stack: Python 3.11, FastAPI, deployed on AWS ECS.
Team has PostgreSQL expertise but limited NoSQL experience.
Budget allows managed database services.
""",
    constraints=[
        "Must support ACID transactions for order consistency",
        "Write throughput: 200 events/second peak",
        "Read patterns: temporal queries, event replay, projections",
        "Data retention: 7 years for compliance",
        "Recovery Point Objective (RPO): 1 minute",
    ],
    decision_criteria=[
        "Write performance and scalability",
        "Query flexibility for event patterns",
        "Operational complexity and team learning curve",
        "Cost at 10K orders/day with 7-year retention",
        "Backup and point-in-time recovery capabilities",
    ],
    alternatives=[
        "PostgreSQL with event table and JSONB columns",
        "Event-specific database (EventStoreDB)",
        "Document database (MongoDB) with change streams",
        "DynamoDB with streams for event processing",
    ]
)

structured_prompt = adr_prompt.to_prompt()
# Send to AI model for generation
```

**Practical Implications:**

- Structured prompts yield consistent ADR quality across team members
- Decision criteria focus AI on relevant trade-offs, not generic comparisons
- Constraints prevent AI from suggesting impractical solutions
- Template reuse reduces prompt engineering time from 20 minutes to 2 minutes

**Trade-offs:**

- Initial template design requires 1-2 hours per organization
- Too much structure limits AI creativity in alternative generation
- Template maintenance needed as ADR standards evolve
- Balance specificity (better results) vs. flexibility (broader applicability)

### 2. Iterative Refinement and Validation

AI-generated ADRs require validation because models hallucinate technical details, miss domain-specific constraints, and cannot evaluate organizational fit. Engineers must systematically validate and refine outputs.

**Technical Implementation:**

```python
from typing import Dict, List, Set
import re

class ADRValidator:
    """Validates and refines AI-generated ADRs."""
    
    def __init__(self, domain_knowledge: Dict[str, Set[str]]):
        """
        Args:
            domain_knowledge: Known constraints, e.g.,
                {"approved_technologies": {"postgres", "redis", "kafka"},
                 "performance_requirements": {"p99_latency < 100ms"},
                 "compliance_requirements": {"GDPR", "SOC2"}}
        """
        self.domain_knowledge = domain_knowledge
        
    def validate_adr(self, adr_content: str) -> Dict[str, List[str]]:
        """
        Validate ADR for hallucinations, inconsistencies, and completeness.
        
        Returns:
            Dictionary with validation issues by category
        """
        issues = {
            "hallucinations": [],
            "missing_criteria": [],
            "inconsistencies": [],
            "compliance_gaps": []
        }
        
        # Check for unrealistic performance claims
        performance_claims = self._extract_performance_claims(adr_content)
        for claim in performance_claims:
            if not self._validate_performance_claim(claim):
                issues["hallucinations"].append(
                    f"Unrealistic performance claim: {claim}"
                )
        
        # Check for unapproved technologies
        technologies = self._extract_technologies(adr_content)
        approved = self.domain_knowledge.get("approved_technologies", set())
        for tech in technologies:
            if approved and tech.lower() not in approved:
                issues["hallucinations"].append(
                    f"Unapproved technology mentioned: {tech}"
                )
        
        # Check for required decision criteria
        required_criteria = [
            "performance", "scalability", "operational complexity",
            "cost", "team expertise"
        ]
        for criterion in required_criteria:
            if criterion.lower() not in adr_content.lower():
                issues["missing_criteria"].append(
                    f"Missing evaluation criterion: {criterion}"
                )
        
        # Check compliance requirements
        compliance_reqs = self.domain_knowledge.get("compliance_requirements", set())
        for req in compliance_reqs:
            if req.lower() not in adr_content.lower():
                issues["compliance_gaps"].append(
                    f"No mention of compliance requirement: {req}"
                )
        
        return {k: v for k, v in issues.items() if v}
    
    def _extract_performance_claims(self, content: str) -> List[str]:
        """Extract performance metrics from ADR."""
        patterns = [
            r'(\d+(?:,\d+)*)\s*(?:requests?|queries?|ops?)/s(?:ec(?:ond)?)?',
            r'p\d+\s*[<>=]+\s*\d+\s*ms',
            r'latency[:\s]+\d+\s*ms',
        ]
        claims = []
        for pattern in patterns:
            claims.extend(re.findall(pattern, content, re.IGNORECASE))
        return claims
    
    def _validate_performance_claim(self, claim: str) -> bool:
        """Validate if performance claim is realistic."""
        # Simple heuristic validation
        # In production, compare against benchmarks database
        numbers = re.findall(r'\d+', claim)
        if not numbers:
            return True
        
        value = int(numbers[0])
        # Flag suspiciously high claims
        if 'requests/s' in claim.lower() and value > 1_000_000:
            return False
        if 'ms' in claim.lower() and value < 1:
            return False
        
        return True
    
    def _extract_technologies(self, content: str) -> Set[str]:
        """Extract technology names from ADR."""
        # Simple extraction - production version would use NER
        common_tech = {
            'postgresql', 'postgres', 'mysql', 'mongodb', 'redis',
            'kafka', 'rabbitmq', 'dynamodb', 'eventstoredb',
            'elasticsearch', 'cassandra'
        }
        found = set()
        content_lower = content.lower()
        for tech in common_tech:
            if tech in content_lower:
                found.add(tech)
        return found
    
    def generate_refinement_prompt(
        self,
        original_adr: str,
        validation_issues: Dict[str, List[str]]
    ) -> str:
        """Generate prompt to refine ADR based on validation issues."""
        issues_text = "\n\n".join(
            f"**{category.replace('_', ' ').title()}:**\n" +
            "\n".join(f"- {issue}" for issue in issues_list)
            for category, issues_list in validation_issues.items()
        )
        
        return f"""Review and refine the following ADR based on validation issues:

{original_adr}

## Validation Issues Found:

{issues_text}

## Refinement Instructions:

1. Address each validation issue specifically
2. Replace unrealistic claims with evidence-based estimates
3. Add missing evaluation criteria with concrete analysis
4. Ensure all compliance requirements are addressed
5. Keep technical depth and specificity
6. Maintain structured format

Provide the refined ADR:
"""


# Example usage
validator = ADRValidator(
    domain_knowledge={
        "approved_technologies": {"postgresql", "redis", "kafka", "dynamodb"},
        "performance_requirements": {"p99_latency < 100ms"},
        "compliance_requirements": {"GDPR", "SOC2", "PCI-DSS"}
    }
)

ai_generated_adr = """
# ADR: Select Database for Event Sourcing

We recommend EventStoreDB which can handle 10 million events/second
with sub-millisecond latency...
"""

issues = validator.validate_adr(ai_generated_adr)
if issues:
    refinement_prompt = validator.generate_refinement_prompt(
        ai_generated_adr, issues
    )
    # Send refinement_prompt back to AI for iteration
```

**Practical Implications:**

- Automated validation catches 70-80% of AI hallucinations before human review
- Refinement prompts improve ADR quality without starting from scratch
- Domain knowledge encoding is reusable across all ADRs
- Validation step takes 2-3 minutes vs. 15-20 minutes manual review

**Trade-offs:**

- False positives require human override decisions
- Domain knowledge maintenance overhead
- Validation rules must evolve with team standards
- Cannot catch subtle logical inconsistencies

### 3. Alternative Analysis and Trade-off Matrices

Effective ADRs require systematic comparison of alternatives. AI generates comprehensive trade-off analyses but engineers must weight criteria based on business context.

**Technical Implementation:**

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class Rating(Enum):
    POOR = 1
    FAIR = 2
    GOOD = 3
    EXCELLENT = 4

@dataclass
class Alternative:
    name: str
    description: str
    ratings: Dict[str, Rating]
    estimated_cost_monthly: float
    implementation_time_weeks: int
    
@dataclass
class DecisionCriteria:
    name: str
    weight: float  # 0.0 to 1.0, sum should be 1.0
    description: str

class TradeoffAnalyzer:
    """Analyzes alternatives using weighted decision criteria."""
    
    def __init__(self, criteria: List[DecisionCriteria]):
        """
        Args:
            criteria: List of decision criteria with weights
        """
        total_weight = sum(c.weight for c in criteria)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Criteria weights must sum to 1.0, got {total_weight}")
        self.criteria = criteria
    
    def analyze_alternatives(
        self,
        alternatives: List[Alternative]
    ) -> Dict[str, any]:
        """
        Score alternatives and generate recommendation.
        
        Returns:
            Analysis with scores, rankings, and recommendation
        """
        scored_alternatives = []
        
        for alt in alternatives:
            # Calculate weighted score
            score = 0.0
            criterion_scores = {}
            
            for criterion in self.criteria:
                if criterion.name in alt.ratings:
                    rating_value = alt.ratings[criterion.name].value
                    weighted_score = (rating_value / 4.0) * criterion.weight
                    score += weighted_score
                    criterion_scores[criterion.name] = {
                        "rating": alt.ratings[criterion.name].name,
                        "weight": criterion.weight,
                