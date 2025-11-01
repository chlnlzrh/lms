# Project DNA & AI-Native Architecture: Engineering Systems That Learn

## Core Concepts

### Technical Definition

**Project DNA** is an architectural pattern where system behavior and knowledge are encoded as structured, version-controlled configuration rather than hardcoded logic. Think of it as infrastructure-as-code extended to business logic, decision-making rules, and domain knowledge. An **AI-native system** treats language models as first-class architectural components—not bolt-on features—where natural language becomes a primary interface for both humans and systems.

Traditional systems separate data, logic, and knowledge into rigid layers. AI-native systems with Project DNA collapse these boundaries: a single source of truth (the DNA) defines what the system knows, how it behaves, and how it evolves, while LLMs provide the execution engine that interprets this DNA contextually.

### Engineering Analogy: Traditional vs. AI-Native

**Traditional Approach:**

```python
# Hardcoded business logic embedded in code
class OrderProcessor:
    def validate_order(self, order: dict) -> tuple[bool, str]:
        # Business rules scattered across codebase
        if order['total'] > 10000:
            if order['customer_tier'] != 'premium':
                return False, "Large orders require premium tier"
        
        if order['shipping_country'] == 'US':
            if 'tax_id' not in order:
                return False, "US orders require tax ID"
        
        # Adding new rules requires code changes, testing, deployment
        return True, "Valid"
    
    def calculate_discount(self, order: dict) -> float:
        # More hardcoded logic
        if order['customer_tier'] == 'premium':
            return 0.15
        elif order['total'] > 5000:
            return 0.10
        return 0.0
```

**AI-Native with Project DNA:**

```python
from typing import TypedDict
import anthropic
import json

# Project DNA: Declarative, version-controlled configuration
PROJECT_DNA = {
    "domain": "order_processing",
    "version": "2.1.0",
    "business_rules": {
        "order_validation": {
            "large_order_threshold": 10000,
            "large_order_requirements": "Orders exceeding threshold require premium tier or manual approval",
            "regional_requirements": {
                "US": ["tax_id", "state"],
                "EU": ["vat_number"],
                "default": []
            }
        },
        "pricing": {
            "discount_tiers": {
                "premium": {"base": 0.15, "description": "Premium customers get 15% base discount"},
                "standard": {"threshold": 5000, "rate": 0.10, "description": "10% on orders over $5000"}
            }
        }
    },
    "policies": """
    Order Processing Guidelines:
    1. Prioritize customer experience while maintaining compliance
    2. Flag edge cases for human review rather than rejecting automatically
    3. Provide clear, actionable feedback on validation failures
    4. Consider customer history and context when applying rules
    """
}

class AIOrderProcessor:
    def __init__(self, dna: dict):
        self.dna = dna
        self.client = anthropic.Anthropic()
    
    def process_with_context(self, order: dict, operation: str) -> dict:
        """Single interface for any order operation using DNA as context"""
        prompt = f"""You are an order processing system. Use the following system DNA to {operation}:

<system_dna>
{json.dumps(self.dna, indent=2)}
</system_dna>

<order>
{json.dumps(order, indent=2)}
</order>

Analyze this order according to our business rules and policies. Return JSON:
{{
    "valid": true/false,
    "issues": ["list of problems"],
    "recommendations": ["suggested actions"],
    "calculated_discount": 0.00,
    "reasoning": "brief explanation"
}}
"""
        
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(message.content[0].text)

# Usage: Logic lives in DNA, not code
processor = AIOrderProcessor(PROJECT_DNA)

order1 = {
    "total": 12000,
    "customer_tier": "standard",
    "customer_history": {"total_lifetime_value": 50000, "orders": 15},
    "shipping_country": "US"
}

result = processor.process_with_context(order1, "validate and price")
print(json.dumps(result, indent=2))
```

### Key Insights That Change Engineering Thinking

1. **Configuration as Intelligence**: In AI-native systems, configuration isn't just settings—it's compressed knowledge. Your DNA encodes decades of business logic that an LLM can interpret contextually, handling edge cases you never explicitly programmed.

2. **Natural Language as Schema**: Traditional systems require rigid schemas (JSON Schema, Protocol Buffers). AI-native systems use natural language descriptions alongside structure, making the system self-documenting and interpretable by both machines and humans.

3. **Behavior Versioning Over Code Versioning**: Instead of versioning code that implements behavior, you version the DNA that describes behavior. The execution engine (LLM) remains constant while the system evolves through DNA updates.

4. **Context-Aware Execution**: Traditional if-then logic evaluates rules in isolation. LLMs evaluate rules with full context, balancing competing constraints naturally. The system can reason: "This order violates the threshold rule, but customer history suggests approval with review."

### Why This Matters NOW

Three convergent forces make AI-native architecture with Project DNA urgent:

**1. LLM Capabilities Crossed Production Threshold**: Models like Claude 3.5 Sonnet reliably interpret complex instructions, follow structured output formats, and reason over 200K+ tokens. This enables treating natural language descriptions as executable specifications.

**2. Regulatory Velocity Exceeds Code Velocity**: Financial regulations, privacy laws, and compliance requirements change faster than traditional development cycles. DNA-based systems let domain experts update behavior by editing configuration, not waiting for developer bandwidth.

**3. Cost-Benefit Inflection Point**: Two years ago, calling an LLM for every business decision was prohibitively expensive. Today, with ~$1/million input tokens and caching, the cost is negligible compared to the engineering cost of maintaining hardcoded logic.

## Technical Components

### 1. DNA Schema Structure

**Technical Explanation**: The DNA schema is a hierarchical data structure combining machine-readable configuration (JSON/YAML) with human-readable descriptions (natural language). It must be simultaneously parseable by code (for versioning, diffing, validation) and interpretable by LLMs (for execution).

**Practical Implementation**:

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Rule(BaseModel):
    """Individual business rule with LLM-interpretable description"""
    id: str
    condition: str  # Natural language condition
    action: str     # Natural language action
    priority: int = Field(ge=1, le=10)
    active: bool = True
    rationale: str  # Why this rule exists
    examples: list[dict] = Field(default_factory=list)

class DomainDNA(BaseModel):
    """Version-controlled system behavior specification"""
    domain: str
    version: str
    updated: datetime = Field(default_factory=datetime.now)
    
    # Structured data for programmatic access
    parameters: dict[str, float | int | str]
    
    # Natural language for LLM interpretation
    rules: list[Rule]
    policies: str  # Overarching guidelines
    examples: list[dict]  # Few-shot examples
    
    # Metadata for governance
    changelog: list[str]
    owner: str
    compliance_refs: list[str] = Field(default_factory=list)

# Example: Credit approval DNA
credit_dna = DomainDNA(
    domain="credit_approval",
    version="3.2.1",
    owner="risk-team@example.com",
    parameters={
        "min_credit_score": 650,
        "max_debt_to_income": 0.43,
        "min_employment_months": 24
    },
    rules=[
        Rule(
            id="high_score_fast_track",
            condition="Credit score >= 750 AND stable employment >= 24 months",
            action="Auto-approve up to $50,000 with prime rate",
            priority=9,
            rationale="Excellent credit history indicates low risk",
            examples=[
                {"score": 780, "employment_months": 36, "approved": True, "amount": 45000}
            ]
        ),
        Rule(
            id="thin_file_manual_review",
            condition="Credit history < 12 months OR fewer than 3 tradelines",
            action="Flag for manual underwriter review with additional documentation",
            priority=7,
            rationale="Insufficient history to assess risk algorithmically",
            examples=[
                {"history_months": 8, "tradelines": 2, "action": "manual_review"}
            ]
        )
    ],
    policies="""
    Credit Approval Philosophy:
    - Prioritize responsible lending; avoid predatory patterns
    - Consider full financial picture, not just scores
    - Provide clear decline reasons with improvement paths
    - Flag borderline cases for human judgment
    """,
    examples=[
        {
            "scenario": "First-time borrower, excellent income",
            "data": {"score": 720, "income": 120000, "history_months": 6},
            "decision": "Approve $25,000 with rate +1.5% and require co-signer"
        }
    ],
    changelog=[
        "3.2.1: Reduced min employment from 36 to 24 months per fair lending review",
        "3.2.0: Added thin-file manual review rule",
        "3.1.0: Increased fast-track threshold to $50k"
    ],
    compliance_refs=["ECOA", "FCRA", "Reg B"]
)

# Save as version-controlled JSON
with open(f"credit_dna_v{credit_dna.version}.json", "w") as f:
    f.write(credit_dna.model_dump_json(indent=2))
```

**Real Constraints/Trade-offs**:
- **Size limits**: DNA must fit in LLM context window. For Claude's 200K tokens, this allows ~500KB of compressed DNA. Beyond that, implement hierarchical loading.
- **Update cadence**: Frequent DNA changes create version proliferation. Batch related updates; use semantic versioning strictly.
- **Interpretation variance**: Same DNA may yield slightly different interpretations across LLM versions. Pin model versions in production; test DNA updates against multiple model versions.

### 2. DNA Injection Patterns

**Technical Explanation**: How you inject DNA into prompts dramatically impacts system behavior. Three primary patterns exist: full injection (entire DNA in every prompt), selective injection (filter relevant sections), and layered injection (base DNA + request-specific overlay).

**Pattern Implementation**:

```python
import anthropic
from typing import Literal

class DNAInjector:
    """Manages DNA injection strategies for different scenarios"""
    
    def __init__(self, dna: DomainDNA, client: anthropic.Anthropic):
        self.dna = dna
        self.client = client
    
    def full_injection(self, user_request: dict) -> str:
        """Simple: Inject entire DNA. Use for: complex decisions, rare operations"""
        prompt = f"""<system_dna version="{self.dna.version}">
{self.dna.model_dump_json(indent=2)}
</system_dna>

<request>
{json.dumps(user_request, indent=2)}
</request>

Process this request according to system DNA. Consider all rules, policies, and examples."""
        return prompt
    
    def selective_injection(self, user_request: dict, relevant_rule_ids: list[str]) -> str:
        """Optimized: Inject only relevant DNA sections. Use for: high-frequency operations"""
        relevant_rules = [r for r in self.dna.rules if r.id in relevant_rule_ids]
        
        prompt = f"""<system_context>
Domain: {self.dna.domain} v{self.dna.version}
Parameters: {json.dumps(self.dna.parameters)}

Active Rules:
{json.dumps([r.model_dump() for r in relevant_rules], indent=2)}

Policies:
{self.dna.policies}
</system_context>

<request>
{json.dumps(user_request, indent=2)}
</request>

Apply relevant rules to this request."""
        return prompt
    
    def layered_injection(self, user_request: dict, session_context: dict) -> str:
        """Advanced: Base DNA + session overrides. Use for: user-specific customization"""
        prompt = f"""<base_system>
{self.dna.model_dump_json(indent=2)}
</base_system>

<session_overrides>
{json.dumps(session_context, indent=2)}
</session_overrides>

<request>
{json.dumps(user_request, indent=2)}
</request>

Apply base system DNA with session-specific overrides. Session context takes precedence."""
        return prompt

# Performance comparison
injector = DNAInjector(credit_dna, anthropic.Anthropic())

request = {
    "applicant": {"credit_score": 720, "income": 85000},
    "loan_amount": 30000
}

# Full injection: ~2000 tokens, higher accuracy for complex cases
full_prompt = injector.full_injection(request)

# Selective: ~800 tokens, 60% cost reduction, sufficient for standard cases
selective_prompt = injector.selective_injection(request, ["high_score_fast_track"])

# Layered: ~1500 tokens, enables personalization
layered_prompt = injector.layered_injection(
    request,
    {"user_segment": "returning_customer", "risk_tolerance": "moderate"}
)
```

**Real Constraints**:
- **Prompt caching**: Use Claude's prompt caching by keeping DNA in system message. First call costs full price; subsequent calls within 5 minutes cost 90% less for cached DNA.
- **Token efficiency**: Full injection uses 2-10× more tokens than selective. For >1000 QPS systems, selective injection is mandatory for cost control.
- **Consistency**: Layered injection can create ambiguity if base and override conflict. Establish clear precedence rules.

### 3. DNA Evolution & Version Control

**Technical Explanation**: DNA isn't static—it evolves as business requirements change. Treating DNA as code enables git-based workflows: branches for experiments, diffs for review, rollbacks for failures.

**Evolution Workflow**:

```python
import git
from pathlib import Path
import difflib

class DNAVersionControl:
    """Manage DNA evolution with git integration"""
    
    def __init__(self, repo_path: str):
        self.repo = git.Repo(repo_path)
        self.dna_dir = Path(repo_path) / "dna"
        self.dna_dir.mkdir(exist_ok=True)
    
    def propose_change(self, domain: str, changes: dict, rationale: str) -> str:
        """Create feature branch with proposed DNA changes"""
        # Create feature branch
        branch_name = f"dna-update/{domain}/{datetime.now():%Y%m%d-%H%M%S}"
        current = self.repo.head.reference
        new_branch = self.repo.create_head(branch_name)
        new_branch.checkout()
        
        # Load current DNA
        dna_file = self.dna_dir / f"{domain}.json"
        current_dna = DomainDNA.model_validate_json(dna_file.read_text())
        
        # Apply changes
        updated_dna = current_dna.model_copy(deep=True)
        for key, value in changes.items():
            setattr(updated_dna, key, value)
        
        # Bump version
        major, minor, patch = map(int, updated_dna.version.split('.'))
        updated_dna.version = f"{major}.{minor}.{patch + 1}"
        updated_dna.changelog.insert(0, f"{updated_dna.version}: {rationale}")
        
        # Save and commit
        dna_file.write_text(updated_dna.model_dump_json(indent