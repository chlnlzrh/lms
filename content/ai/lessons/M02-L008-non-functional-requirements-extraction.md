# Non-Functional Requirements Extraction with LLMs

## Core Concepts

Non-functional requirements (NFRs) extraction is the process of identifying system qualities, constraints, and operational characteristics that aren't directly related to specific features or behaviors. Traditional software engineering has always struggled with NFR elicitation—stakeholders articulate features easily ("users should be able to search products") but rarely volunteer requirements about scalability, maintainability, security, or performance until problems emerge.

LLMs transform NFR extraction from an interview-intensive, expert-driven process into a systematic analysis capability. They excel at pattern recognition across unstructured text, understanding implicit constraints, and mapping domain language to technical requirements.

### Traditional vs. Modern Approach

**Traditional NFR Extraction:**

```python
# Manual template-based extraction
def extract_nfrs_traditional(requirements_doc: str) -> dict:
    """Parse requirements using keyword matching and templates."""
    nfrs = {
        'performance': [],
        'security': [],
        'scalability': []
    }
    
    # Brittle keyword matching
    performance_keywords = ['fast', 'quick', 'responsive', 'latency']
    security_keywords = ['secure', 'encrypted', 'authenticated']
    
    lines = requirements_doc.lower().split('\n')
    for line in lines:
        if any(kw in line for kw in performance_keywords):
            nfrs['performance'].append(line)
        if any(kw in line for kw in security_keywords):
            nfrs['security'].append(line)
    
    # Misses: implicit requirements, contextual constraints,
    # industry standards, cross-cutting concerns
    return nfrs

# Result: Catches explicit mentions, misses 60-80% of actual NFRs
```

**LLM-Powered NFR Extraction:**

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json

class NFRCategory(Enum):
    PERFORMANCE = "performance"
    SECURITY = "security"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    COMPLIANCE = "compliance"

@dataclass
class NonFunctionalRequirement:
    category: NFRCategory
    requirement: str
    source: str  # Where in the doc this was derived from
    confidence: float
    measurable_criteria: Optional[str] = None
    rationale: str = ""

def extract_nfrs_with_llm(
    requirements_doc: str,
    domain_context: str = "",
    llm_client = None  # Your LLM API client
) -> List[NonFunctionalRequirement]:
    """
    Extract both explicit and implicit NFRs using LLM analysis.
    Understands context, industry standards, and cross-cutting concerns.
    """
    
    prompt = f"""Analyze the following requirements document and extract ALL non-functional requirements, including:
1. Explicitly stated NFRs (directly mentioned)
2. Implicit NFRs (implied by context, domain, or industry standards)
3. Derived NFRs (logical consequences of functional requirements)

Domain context: {domain_context}

Requirements document:
{requirements_doc}

For each NFR, provide:
- Category (performance/security/scalability/maintainability/reliability/usability/compliance)
- Specific requirement statement
- Source (quote or paraphrase what implied this)
- Confidence (0.0-1.0)
- Measurable criteria (if determinable)
- Rationale (why this is needed)

Return as JSON array with structure:
[{{"category": "...", "requirement": "...", "source": "...", "confidence": 0.0, "measurable_criteria": "...", "rationale": "..."}}]
"""

    response = llm_client.complete(prompt, temperature=0.2)
    nfrs_data = json.loads(response)
    
    return [
        NonFunctionalRequirement(
            category=NFRCategory(nfr['category']),
            requirement=nfr['requirement'],
            source=nfr['source'],
            confidence=nfr['confidence'],
            measurable_criteria=nfr.get('measurable_criteria'),
            rationale=nfr['rationale']
        )
        for nfr in nfrs_data
    ]

# Result: Captures explicit + implicit NFRs with justification
```

### Key Engineering Insights

**1. Context Understanding Bridges Domain-Technical Gap**

LLMs can translate domain language into technical constraints. When a healthcare document mentions "patient records," the LLM infers HIPAA compliance requirements, encryption at rest, audit logging, and access control—requirements rarely explicitly stated but legally mandatory.

**2. NFRs Are Network Effects, Not Lists**

Traditional extraction treats NFRs as independent items. LLMs recognize that "must handle 10,000 concurrent users" has cascading implications: database connection pooling, horizontal scaling architecture, caching strategy, session management, and load balancer configuration. One stated requirement implies a dozen technical constraints.

**3. Ambiguity Detection Is as Valuable as Extraction**

The most dangerous NFRs are vague: "system must be fast." LLMs can flag these, suggest measurable alternatives ("95th percentile response time < 200ms"), and prompt stakeholders for clarification before architecture decisions lock in assumptions.

### Why This Matters Now

Production systems fail primarily from violated NFRs, not missing features. Performance degradation, security breaches, compliance violations, and maintenance nightmares emerge from implicit requirements that were never captured, discussed, or architected for. LLMs make systematic NFR discovery economically feasible for every project, not just those that can afford dedicated requirements engineers and multi-week stakeholder workshops.

## Technical Components

### 1. Context-Aware Categorization

NFR extraction isn't just text classification—it requires understanding technical implications in domain context.

**Technical Explanation:**

LLMs leverage their pre-training across millions of technical documents, architectural patterns, and domain-specific texts to map natural language to NFR categories. They understand that "real-time trading platform" inherently requires sub-millisecond latency (performance), financial regulation compliance (compliance), and fault tolerance (reliability)—even if the requirements doc never uses those terms.

**Implementation Pattern:**

```python
from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class DomainContext:
    industry: str
    system_type: str
    regulatory_environment: List[str] = field(default_factory=list)
    user_base_characteristics: Dict[str, str] = field(default_factory=dict)
    integration_requirements: List[str] = field(default_factory=list)

def build_context_aware_prompt(
    requirements: str,
    domain: DomainContext
) -> str:
    """Construct prompt that activates relevant domain knowledge."""
    
    context_primer = f"""
Domain: {domain.industry}
System Type: {domain.system_type}
Regulatory: {', '.join(domain.regulatory_environment) if domain.regulatory_environment else 'None specified'}
User Base: {json.dumps(domain.user_base_characteristics)}
Integrations: {', '.join(domain.integration_requirements)}

Industry-standard NFRs for {domain.industry} {domain.system_type}s typically include:
"""
    
    # Add domain-specific NFR guidance
    domain_guidance = {
        'healthcare': '- HIPAA compliance\n- Data retention policies\n- Audit trail requirements\n- PHI encryption',
        'financial': '- PCI-DSS compliance\n- Transaction consistency\n- Audit logging\n- Disaster recovery < 4hr RTO',
        'ecommerce': '- PCI compliance for payments\n- 99.9% uptime SLA\n- Black Friday scalability\n- Cart persistence'
    }
    
    guidance = domain_guidance.get(domain.industry.lower(), '')
    
    return f"""{context_primer}
{guidance}

Given this domain context, analyze the following requirements and extract ALL applicable NFRs,
including those implied by industry standards and regulatory requirements:

{requirements}

For each NFR, specify:
1. Whether it's explicit (stated) or implicit (industry standard/regulatory/derived)
2. The confidence level (1.0 for explicit/regulatory, 0.7-0.9 for implied)
3. Specific measurable acceptance criteria
"""

# Usage example
domain = DomainContext(
    industry='healthcare',
    system_type='patient portal',
    regulatory_environment=['HIPAA', 'HITECH'],
    user_base_characteristics={'size': '50000+', 'tech_savvy': 'low to moderate'},
    integration_requirements=['EHR system', 'lab systems', 'billing']
)

prompt = build_context_aware_prompt(requirements_text, domain)
```

**Practical Implications:**

- **Reduced NFR gaps:** Domain-aware extraction surfaces 40-60% more requirements than keyword matching
- **Earlier risk identification:** Regulatory NFRs caught in requirements phase, not during compliance audit
- **Better stakeholder communication:** LLM translates technical NFRs back to business language for validation

**Constraints:**

- Domain knowledge quality varies: Healthcare and financial services have extensive training data; niche industries may need few-shot examples
- Hallucination risk: LLM might infer regulatory requirements that don't apply to your jurisdiction or have changed
- Context window limits: Large requirements documents need chunking strategies

### 2. Implicit Requirement Inference

Most critical NFRs are never explicitly stated—they're assumed, implied, or consequences of other requirements.

**Technical Explanation:**

LLMs use causal reasoning and world knowledge to derive implicit requirements. "Mobile app for field technicians" implies offline capability, battery efficiency, rugged device support, and GPS integration—even if the requirements doc only mentions "job assignment and completion tracking."

**Implementation Pattern:**

```python
from typing import List, Tuple
import json

@dataclass
class ImplicitNFR:
    requirement: str
    inference_chain: List[str]  # Step-by-step reasoning
    upstream_requirement: str  # What functional req triggered this
    risk_if_missed: str
    
def extract_implicit_nfrs(
    functional_requirements: List[str],
    domain: DomainContext,
    llm_client
) -> List[ImplicitNFR]:
    """
    For each functional requirement, derive implicit NFRs through
    chain-of-thought reasoning.
    """
    
    prompt = f"""For each functional requirement below, identify implicit non-functional requirements
using chain-of-thought reasoning. Show your work.

Domain: {domain.industry} - {domain.system_type}

For each functional requirement, analyze:
1. What technical capabilities are needed to deliver this?
2. What performance, security, or reliability constraints are implied?
3. What could go wrong if these implicit NFRs aren't addressed?

Functional Requirements:
{json.dumps(functional_requirements, indent=2)}

Return JSON array:
[{{
    "functional_req": "original requirement",
    "implicit_nfr": "derived NFR statement",
    "reasoning_chain": ["step 1", "step 2", "step 3"],
    "risk_if_missed": "specific failure scenario"
}}]

Example reasoning chain:
Functional: "Users can upload profile photos"
→ Photos are user-generated content → need content moderation
→ Photos contain PII (faces) → need privacy controls and consent
→ Photos are stored permanently → need storage scaling strategy
→ Photos displayed on every page → need CDN and caching
Implicit NFRs: content moderation system, privacy policy compliance, 
scalable storage (S3-class), CDN integration, image optimization pipeline
"""

    response = llm_client.complete(prompt, temperature=0.3)
    implicit_nfrs_data = json.loads(response)
    
    return [
        ImplicitNFR(
            requirement=item['implicit_nfr'],
            inference_chain=item['reasoning_chain'],
            upstream_requirement=item['functional_req'],
            risk_if_missed=item['risk_if_missed']
        )
        for item in implicit_nfrs_data
    ]

def validate_implicit_nfrs(
    implicit_nfrs: List[ImplicitNFR],
    llm_client
) -> List[Tuple[ImplicitNFR, float]]:
    """
    Second-pass validation: Check if inferred NFRs are actually necessary
    or if LLM over-extrapolated.
    """
    
    validation_prompt = f"""Review these inferred NFRs and assess necessity.
For each, rate:
- Necessity (0.0-1.0): Is this truly required or nice-to-have?
- Universality (0.0-1.0): Does this apply to most systems of this type?

{json.dumps([{
    'nfr': nfr.requirement,
    'reasoning': nfr.inference_chain,
    'source': nfr.upstream_requirement
} for nfr in implicit_nfrs], indent=2)}

Return JSON: [{{"nfr_index": 0, "necessity_score": 0.0, "universality_score": 0.0, "rationale": "..."}}]
"""
    
    validation_data = json.loads(llm_client.complete(validation_prompt, temperature=0.1))
    
    return [
        (implicit_nfrs[v['nfr_index']], v['necessity_score'] * v['universality_score'])
        for v in validation_data
    ]
```

**Practical Implications:**

- **Catches architectural oversights early:** "Real-time dashboard" implies WebSocket infrastructure, not REST polling
- **Surfaces hidden costs:** Seemingly simple features imply significant NFR investments
- **Prioritization support:** Not all implicit NFRs are equal—validation scoring helps decide what's MVP vs. future

**Constraints:**

- Over-inference risk: LLM might derive NFRs for edge cases that won't occur in your context
- Requires validation: Implicit NFRs need review by domain experts—treat as suggestions, not facts
- Reasoning opacity: Even with chain-of-thought, some inferences may seem arbitrary

### 3. Measurability Translation

NFRs like "system should be fast" or "must be secure" are useless without measurable criteria. LLMs can translate vague statements into testable requirements.

**Technical Explanation:**

LLMs have learned associations between qualitative descriptions and quantitative metrics from thousands of SLAs, performance benchmarks, and technical specifications. They can suggest appropriate metrics, realistic thresholds, and measurement methodologies.

**Implementation Pattern:**

```python
from typing import Optional
from dataclasses import dataclass

@dataclass
class MeasurableNFR:
    original_statement: str
    category: NFRCategory
    metric: str
    threshold: str
    measurement_method: str
    rationale: str
    alternatives: List[Dict[str, str]]  # Alternative metrics/thresholds

def make_nfrs_measurable(
    vague_nfrs: List[str],
    domain: DomainContext,
    llm_client
) -> List[MeasurableNFR]:
    """
    Convert vague NFR statements into specific, measurable requirements
    with appropriate metrics and thresholds.
    """
    
    prompt = f"""Convert these vague NFRs into measurable, testable requirements.

Domain: {domain.industry} - {domain.system_type}
User base: {domain.user_base_characteristics}

For each vague NFR, provide:
1. Specific metric (with units)
2. Threshold/target value (based on industry standards for this domain)
3. How to measure it (tools, methodology)
4. Why this threshold is appropriate
5. 2-3 alternative metrics/thresholds (more/less stringent)

Vague NFRs:
{json.dumps(vague_nfrs, indent=2)}

Return JSON:
[{{
    "original": "...",
    "category": "performance|security|...",
    "metric": "95th percentile response time",
    "threshold": "< 200ms",
    "measurement_method": "APM tool measuring server response time, excluding network latency",
    "rationale": "Users perceive < 200ms as instant. Industry standard for interactive web apps.",
    "alternatives": [
        {{"metric": "median response time", "threshold": "< 100ms", "stringency": "more"}},
        {{"metric": "99th percentile response time", "threshold": "< 500ms", "stringency": "less"}}
    ]
}}]

Examples:
"System must be fast" → "95th percentile API response time < 200ms under normal load (100 req/sec)"
"Must be secure