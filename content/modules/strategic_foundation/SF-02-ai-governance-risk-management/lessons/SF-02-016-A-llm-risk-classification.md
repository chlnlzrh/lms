# LLM Risk Classification: Engineering Safety into AI Systems

## Core Concepts

LLM risk classification is the systematic process of identifying, categorizing, and prioritizing potential failure modes in language model applications. Unlike traditional software where bugs produce deterministic errors, LLM failures are probabilistic, context-dependent, and often emerge only under specific input conditions or cumulative usage patterns.

### Traditional vs. Modern Risk Management

In traditional software systems, risk management focuses on input validation, type safety, and exception handling:

```python
# Traditional API risk management
def process_payment(amount: float, account_id: str) -> dict:
    # Deterministic validation
    if amount <= 0:
        raise ValueError("Amount must be positive")
    if not account_id.isalnum():
        raise ValueError("Invalid account ID format")
    
    # Predictable failure modes
    try:
        result = payment_gateway.charge(amount, account_id)
        return {"status": "success", "transaction_id": result.id}
    except NetworkError as e:
        return {"status": "retry", "error": str(e)}
    except InsufficientFunds as e:
        return {"status": "declined", "error": str(e)}
```

LLM applications require a fundamentally different approach because failures are non-deterministic and multidimensional:

```python
from typing import Literal, Optional
from enum import Enum
import re

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class RiskCategory(Enum):
    CONTENT_SAFETY = "content_safety"
    DATA_LEAKAGE = "data_leakage"
    HALLUCINATION = "hallucination"
    MANIPULATION = "manipulation"
    BIAS = "bias"
    AVAILABILITY = "availability"

class LLMRiskClassifier:
    def __init__(self):
        self.risk_patterns = {
            RiskCategory.DATA_LEAKAGE: [
                r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card
            ],
            RiskCategory.MANIPULATION: [
                r'ignore previous instructions',
                r'disregard your guidelines',
                r'pretend you are',
            ]
        }
    
    def classify_interaction(
        self, 
        user_input: str, 
        llm_output: str,
        context: Optional[dict] = None
    ) -> dict:
        """
        Classify risks in both input and output.
        Unlike deterministic validation, this produces risk probabilities.
        """
        risks = []
        
        # Input risk assessment
        for category, patterns in self.risk_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    risks.append({
                        "category": category.value,
                        "level": RiskLevel.HIGH,
                        "location": "input",
                        "confidence": 0.85,
                        "mitigation": self._get_mitigation(category)
                    })
        
        # Output risk assessment (requires different techniques)
        hallucination_risk = self._detect_hallucination(llm_output, context)
        if hallucination_risk > 0.5:
            risks.append({
                "category": RiskCategory.HALLUCINATION.value,
                "level": RiskLevel.MEDIUM if hallucination_risk < 0.7 else RiskLevel.HIGH,
                "location": "output",
                "confidence": hallucination_risk,
                "mitigation": "Require citation verification"
            })
        
        return {
            "risks": risks,
            "max_risk_level": max([r["level"] for r in risks], 
                                  default=RiskLevel.LOW),
            "safe_to_proceed": all(r["level"].value <= RiskLevel.MEDIUM.value 
                                   for r in risks)
        }
    
    def _detect_hallucination(self, output: str, context: Optional[dict]) -> float:
        """
        Probabilistic hallucination detection.
        Real implementation would use embeddings, fact-checking, etc.
        """
        confidence_indicators = [
            ("i think", 0.3),
            ("probably", 0.4),
            ("might be", 0.4),
            ("not sure", 0.6),
        ]
        
        risk_score = 0.0
        for phrase, weight in confidence_indicators:
            if phrase in output.lower():
                risk_score = max(risk_score, weight)
        
        return risk_score
    
    def _get_mitigation(self, category: RiskCategory) -> str:
        mitigations = {
            RiskCategory.DATA_LEAKAGE: "Sanitize input before processing",
            RiskCategory.MANIPULATION: "Use instruction hierarchy with system prompts",
            RiskCategory.HALLUCINATION: "Implement fact verification pipeline",
        }
        return mitigations.get(category, "Manual review required")
```

### Why This Matters Now

The shift to LLM-based applications introduces failure modes that traditional software engineering practices don't address:

1. **Non-deterministic outputs**: Same input can produce different outputs, making unit testing insufficient
2. **Emergent behaviors**: Risks appear from interaction patterns, not individual requests
3. **Semantic attacks**: Malicious inputs exploit model understanding, not code vulnerabilities
4. **Responsibility gaps**: When an LLM produces harmful content, liability is unclear

Engineers must adopt risk classification as a first-class design concern, not an afterthought. A medical diagnosis assistant and a creative writing tool require completely different risk profiles and mitigation strategies.

## Technical Components

### 1. Risk Taxonomy and Severity Matrices

A robust risk taxonomy maps failure modes to business impact. Unlike generic severity labels, effective taxonomies are domain-specific:

```python
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class RiskDefinition:
    category: RiskCategory
    severity_threshold: float
    business_impact: str
    detection_method: Callable
    mitigation_strategy: str
    sla_target: float  # Acceptable failure rate

class DomainSpecificRiskTaxonomy:
    """
    Different domains require different risk profiles.
    Healthcare chatbot vs. creative writing tool have vastly different tolerances.
    """
    
    @staticmethod
    def healthcare_taxonomy() -> List[RiskDefinition]:
        return [
            RiskDefinition(
                category=RiskCategory.HALLUCINATION,
                severity_threshold=0.01,  # 99% accuracy required
                business_impact="Patient harm, liability, regulatory violation",
                detection_method=lambda x: verify_against_medical_database(x),
                mitigation_strategy="Require human-in-loop for all diagnoses",
                sla_target=0.001  # 0.1% acceptable failure rate
            ),
            RiskDefinition(
                category=RiskCategory.DATA_LEAKAGE,
                severity_threshold=0.0,  # Zero tolerance
                business_impact="HIPAA violation, lawsuit, reputation damage",
                detection_method=lambda x: scan_for_phi(x),
                mitigation_strategy="Encrypt all data, audit logs, automated scanning",
                sla_target=0.0
            ),
        ]
    
    @staticmethod
    def creative_writing_taxonomy() -> List[RiskDefinition]:
        return [
            RiskDefinition(
                category=RiskCategory.HALLUCINATION,
                severity_threshold=0.5,  # Creativity expected, facts less critical
                business_impact="User disappointment, minor credibility loss",
                detection_method=lambda x: check_coherence(x),
                mitigation_strategy="Disclaimer about creative interpretation",
                sla_target=0.2  # 20% acceptable "creative liberty"
            ),
            RiskDefinition(
                category=RiskCategory.CONTENT_SAFETY,
                severity_threshold=0.05,  # Moderate tolerance
                business_impact="Platform policy violation, user complaints",
                detection_method=lambda x: content_filter(x),
                mitigation_strategy="Post-generation filtering, user reporting",
                sla_target=0.01
            ),
        ]

def verify_against_medical_database(output: str) -> float:
    """Stub: Real implementation would query medical knowledge base"""
    return 0.95

def scan_for_phi(text: str) -> float:
    """Stub: Real implementation would use NER for Protected Health Information"""
    return 0.0

def check_coherence(text: str) -> float:
    """Stub: Real implementation would use coherence scoring models"""
    return 0.85

def content_filter(text: str) -> float:
    """Stub: Real implementation would use content moderation API"""
    return 0.02
```

**Practical Implications**: Your risk taxonomy drives architecture decisions. A zero-tolerance data leakage policy might require on-premise deployment, while a creative tool can use third-party APIs.

**Real Constraints**: Overly restrictive risk thresholds increase false positives, degrading user experience. Underfitting thresholds expose your system to real harms. Calibrate based on production data, not assumptions.

### 2. Multi-Layer Risk Detection

Effective risk classification operates at multiple layers: input sanitization, prompt injection defense, output validation, and behavioral monitoring.

```python
from typing import Protocol
import hashlib
import time

class RiskDetector(Protocol):
    def detect(self, data: dict) -> dict:
        """Returns risk assessment dict"""
        ...

class InputSanitizer(RiskDetector):
    """Layer 1: Prevent malicious inputs from reaching the model"""
    
    def __init__(self):
        self.blocked_patterns = [
            r'(?i)ignore\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions|prompts|rules)',
            r'(?i)system\s*:\s*you\s+are',
            r'<\s*script\s*>',  # XSS attempts
        ]
        self.pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
            (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', 'CREDIT_CARD'),
        ]
    
    def detect(self, data: dict) -> dict:
        user_input = data.get('input', '')
        risks = []
        
        # Injection attack detection
        for pattern in self.blocked_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                risks.append({
                    'type': 'PROMPT_INJECTION',
                    'severity': RiskLevel.CRITICAL,
                    'action': 'BLOCK',
                    'reason': 'Detected instruction manipulation attempt'
                })
        
        # PII detection
        sanitized_input = user_input
        for pattern, pii_type in self.pii_patterns:
            matches = re.finditer(pattern, user_input)
            for match in matches:
                risks.append({
                    'type': 'PII_DETECTED',
                    'pii_type': pii_type,
                    'severity': RiskLevel.HIGH,
                    'action': 'SANITIZE',
                    'reason': f'Detected {pii_type} in input'
                })
                # Replace with hash for logging purposes
                sanitized_input = sanitized_input.replace(
                    match.group(0), 
                    f"[REDACTED_{pii_type}]"
                )
        
        return {
            'risks': risks,
            'sanitized_input': sanitized_input,
            'block': any(r['action'] == 'BLOCK' for r in risks)
        }

class OutputValidator(RiskDetector):
    """Layer 2: Validate model outputs before returning to user"""
    
    def __init__(self):
        self.toxic_threshold = 0.7
        self.factuality_threshold = 0.6
    
    def detect(self, data: dict) -> dict:
        output = data.get('output', '')
        context = data.get('context', {})
        risks = []
        
        # Toxicity check (stub - would use actual toxicity model)
        toxicity_score = self._check_toxicity(output)
        if toxicity_score > self.toxic_threshold:
            risks.append({
                'type': 'TOXIC_CONTENT',
                'severity': RiskLevel.HIGH,
                'score': toxicity_score,
                'action': 'FILTER',
                'mitigation': 'Replace with safe alternative response'
            })
        
        # Factuality check
        if context.get('requires_factuality', False):
            factuality_score = self._check_factuality(output, context)
            if factuality_score < self.factuality_threshold:
                risks.append({
                    'type': 'LOW_FACTUALITY',
                    'severity': RiskLevel.MEDIUM,
                    'score': factuality_score,
                    'action': 'WARN',
                    'mitigation': 'Add uncertainty disclaimer'
                })
        
        return {
            'risks': risks,
            'safe': len([r for r in risks if r['severity'].value >= RiskLevel.HIGH.value]) == 0
        }
    
    def _check_toxicity(self, text: str) -> float:
        """Stub: Would use actual toxicity detection model"""
        toxic_keywords = ['hate', 'violence', 'explicit']
        return sum(kw in text.lower() for kw in toxic_keywords) / len(toxic_keywords)
    
    def _check_factuality(self, output: str, context: dict) -> float:
        """Stub: Would use fact-checking against knowledge base"""
        hedging_phrases = ['might', 'possibly', 'perhaps', 'i think']
        hedging_count = sum(phrase in output.lower() for phrase in hedging_phrases)
        return max(0.0, 1.0 - (hedging_count * 0.2))

class BehavioralMonitor(RiskDetector):
    """Layer 3: Detect abuse patterns across multiple interactions"""
    
    def __init__(self):
        self.user_history = {}
        self.rate_limit = 100  # requests per hour
        self.pattern_window = 3600  # 1 hour
    
    def detect(self, data: dict) -> dict:
        user_id = data.get('user_id')
        timestamp = data.get('timestamp', time.time())
        
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        # Clean old history
        self.user_history[user_id] = [
            t for t in self.user_history[user_id] 
            if timestamp - t < self.pattern_window
        ]
        
        self.user_history[user_id].append(timestamp)
        
        risks = []
        request_count = len(self.user_history[user_id])
        
        # Rate limiting
        if request_count > self.rate_limit:
            risks.append({
                'type': 'RATE_LIMIT_EXCEEDED',
                'severity': RiskLevel.MEDIUM,
                'count': request_count,
                'action': 'THROTTLE',
                'reason': f'User exceeded {self.rate_limit} requests/hour'
            })
        
        # Pattern detection
        if request_count > 50:
            avg_interval = self.pattern_window / request_count
            if avg_interval < 5:  # Less than 5 seconds between requests
                risks.append({
                    'type': 'AUTOMATED_ABUSE',
                    'severity': RiskLevel.HIGH,
                    'action': 'CHALLENGE',
                    'reason': 'Suspected bot activity'
                })
        
        return {'risks': risks}

class LayeredRiskDetection:
    """Orchestrate multiple detection layers"""
    
    def __init__(self):
        self.layers = [