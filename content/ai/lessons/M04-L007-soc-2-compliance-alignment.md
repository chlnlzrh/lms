# SOC 2 & Compliance Alignment for AI Systems

## Core Concepts

### Technical Definition

SOC 2 compliance for AI systems requires demonstrating documented controls across five Trust Services Criteria (TSC): Security, Availability, Processing Integrity, Confidentiality, and Privacy. For LLM-based systems, this translates to proving you can consistently control what data enters models, how inference occurs, what outputs are generated, and maintaining auditable evidence of all three—even when dealing with non-deterministic systems.

The engineering challenge: traditional compliance frameworks assume deterministic systems where input + process = predictable output. LLMs introduce probabilistic outputs, emergent behaviors, and data leakage vectors that traditional audit controls weren't designed to handle.

### Engineering Analogy: Traditional vs. AI Compliance

**Traditional Web Application Compliance:**

```python
from typing import Dict, Any
import hashlib
import logging

class TraditionalCompliantService:
    """Deterministic service with predictable audit trails"""
    
    def __init__(self, audit_logger: logging.Logger):
        self.audit_logger = audit_logger
        
    def process_sensitive_data(
        self, 
        user_id: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Input validation - deterministic
        if not self._validate_input(data):
            self.audit_logger.error(f"Invalid input: {user_id}")
            raise ValueError("Input validation failed")
        
        # Processing - deterministic transformation
        result = self._transform_data(data)
        
        # Output validation - deterministic
        if not self._validate_output(result):
            self.audit_logger.error(f"Output validation failed: {user_id}")
            raise ValueError("Output validation failed")
        
        # Audit trail - complete and deterministic
        self.audit_logger.info(
            f"user={user_id}, input_hash={self._hash(data)}, "
            f"output_hash={self._hash(result)}, status=success"
        )
        
        return result
    
    def _hash(self, data: Dict[str, Any]) -> str:
        return hashlib.sha256(str(data).encode()).hexdigest()
```

**AI/LLM Compliant Service:**

```python
from typing import Dict, Any, List, Optional
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class InferenceAuditLog:
    """Comprehensive audit log for non-deterministic LLM operations"""
    timestamp: str
    user_id: str
    session_id: str
    input_hash: str
    input_pii_detected: List[str]
    prompt_template_version: str
    model_version: str
    temperature: float
    max_tokens: int
    output_hash: str
    output_pii_detected: List[str]
    output_toxicity_score: float
    output_blocked: bool
    latency_ms: int
    token_count: int
    
class LLMCompliantService:
    """Non-deterministic service requiring probabilistic controls"""
    
    def __init__(self, audit_logger: logging.Logger):
        self.audit_logger = audit_logger
        self.pii_patterns = self._compile_pii_patterns()
        
    def process_with_llm(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        model_client: Any
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        # Pre-inference controls - detect and redact PII
        pii_found = self._detect_pii(user_input)
        sanitized_input = self._redact_pii(user_input, pii_found)
        
        # Build auditable prompt with version control
        prompt_template_version = "v2.3.1"
        full_prompt = self._build_prompt(sanitized_input, prompt_template_version)
        
        # Controlled inference with locked parameters
        model_params = {
            "model": "gpt-4-0613",  # Pinned version for auditability
            "temperature": 0.0,      # Minimize non-determinism
            "max_tokens": 500,
            "top_p": 1.0
        }
        
        response = model_client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            **model_params
        )
        
        output_text = response.choices[0].message.content
        
        # Post-inference controls - validate output safety
        output_pii = self._detect_pii(output_text)
        toxicity_score = self._calculate_toxicity(output_text)
        
        # Block unsafe outputs
        output_blocked = False
        if len(output_pii) > 0 or toxicity_score > 0.8:
            output_blocked = True
            output_text = "Output blocked due to policy violation"
        
        # Create comprehensive audit trail
        audit_log = InferenceAuditLog(
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            session_id=session_id,
            input_hash=self._hash(user_input),
            input_pii_detected=[p[0] for p in pii_found],
            prompt_template_version=prompt_template_version,
            model_version=model_params["model"],
            temperature=model_params["temperature"],
            max_tokens=model_params["max_tokens"],
            output_hash=self._hash(output_text),
            output_pii_detected=[p[0] for p in output_pii],
            toxicity_score=toxicity_score,
            output_blocked=output_blocked,
            latency_ms=int((time.time() - start_time) * 1000),
            token_count=response.usage.total_tokens
        )
        
        # Structured logging for audit queries
        self.audit_logger.info(json.dumps(asdict(audit_log)))
        
        return {
            "output": output_text,
            "blocked": output_blocked,
            "audit_id": audit_log.timestamp + "_" + user_id
        }
    
    def _compile_pii_patterns(self) -> Dict[str, re.Pattern]:
        return {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')
        }
    
    def _detect_pii(self, text: str) -> List[tuple[str, str]]:
        """Returns list of (pii_type, matched_value) tuples"""
        found = []
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            found.extend([(pii_type, match) for match in matches])
        return found
    
    def _redact_pii(self, text: str, pii_found: List[tuple[str, str]]) -> str:
        redacted = text
        for pii_type, value in pii_found:
            redacted = redacted.replace(value, f"[REDACTED_{pii_type.upper()}]")
        return redacted
    
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _calculate_toxicity(self, text: str) -> float:
        # Simplified toxicity check - production would use ML model
        toxic_words = ["hate", "violent", "explicit"]
        score = sum(1 for word in toxic_words if word in text.lower())
        return min(score / len(toxic_words), 1.0)
    
    def _build_prompt(self, user_input: str, version: str) -> str:
        return f"""You are a helpful assistant. Respond professionally and safely.

User query: {user_input}

Guidelines:
- Do not include personal information in responses
- Keep responses factual and appropriate
- Template version: {version}"""
```

The key difference: traditional systems audit the deterministic path, AI systems must audit the control boundaries around probabilistic processes. You're proving controls work statistically, not deterministically.

### Why This Matters Now

Three critical drivers:

1. **Regulatory enforcement is active**: GDPR fines for AI systems now exceed $500M annually, with SOC 2 becoming table stakes for B2B AI companies. Auditors are actively testing for LLM-specific controls.

2. **Model training data leakage**: LLMs can memorize and regurgitate training data. Without controls, your compliant application becomes a vector for exposing other users' PII or proprietary data.

3. **Retroactive compliance is 10x harder**: Building audit trails after deployment requires reconstructing evidence. Baking compliance into architecture from day one reduces audit prep from months to weeks.

### Key Insights That Change Engineering Thinking

**Insight 1: Compliance is a Data Lineage Problem**

Your audit trail must prove data minimization and purpose limitation through the entire LLM pipeline. This means:
- What data entered the context window (not just API inputs)
- How retrieval systems selected context (not just what they returned)
- Why the model accessed specific data sources (not just that it could)

**Insight 2: Model Outputs Are Controlled Artifacts**

SOC 2 requires demonstrating "processing integrity"—that your system produces outputs meeting defined quality criteria. For LLMs, this means treating outputs as untrusted until validated, with statistical guarantees about safety rates.

**Insight 3: Temperature=0 Is Not Deterministic**

Even with temperature 0, LLMs exhibit variance. Auditors understand this—they want evidence you've characterized the variance and control for edge cases, not proof of impossible determinism.

## Technical Components

### Component 1: Input Sanitization and PII Detection Pipeline

**Technical Explanation:**

Pre-inference PII detection prevents sensitive data from entering the model context, addressing both GDPR's data minimization principle and SOC 2's confidentiality criteria. This requires real-time classification of incoming text before prompt construction.

**Implementation:**

```python
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import re
from enum import Enum

class PIIType(Enum):
    EMAIL = "email"
    SSN = "ssn"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    
@dataclass
class PIIMatch:
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    
class ProductionPIIDetector:
    """Production-grade PII detector with configurable sensitivity"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.patterns = self._build_patterns()
        self.false_positive_filters = self._build_filters()
        
    def _build_patterns(self) -> Dict[PIIType, List[Tuple[re.Pattern, float]]]:
        """Returns dict of PIIType -> [(pattern, confidence_score)]"""
        return {
            PIIType.EMAIL: [
                (re.compile(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
                ), 0.95)
            ],
            PIIType.SSN: [
                # Format: 123-45-6789
                (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), 0.90),
                # Format: 123456789
                (re.compile(r'\b\d{9}\b'), 0.70)  # Lower confidence, more FPs
            ],
            PIIType.PHONE: [
                # Format: (123) 456-7890
                (re.compile(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'), 0.95),
                # Format: 123-456-7890
                (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), 0.85)
            ],
            PIIType.CREDIT_CARD: [
                # Basic Luhn check would go here in production
                (re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'), 0.85)
            ],
            PIIType.IP_ADDRESS: [
                (re.compile(
                    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
                ), 0.80)
            ],
            PIIType.API_KEY: [
                # Common API key patterns
                (re.compile(r'\b[A-Za-z0-9_-]{32,}\b'), 0.70),
                (re.compile(r'\bsk-[A-Za-z0-9]{20,}\b'), 0.95)  # OpenAI style
            ]
        }
    
    def _build_filters(self) -> Dict[PIIType, Set[str]]:
        """Known false positives to exclude"""
        return {
            PIIType.SSN: {
                "123-45-6789",  # Common example SSN
                "000-00-0000"
            },
            PIIType.IP_ADDRESS: {
                "127.0.0.1",
                "0.0.0.0",
                "255.255.255.255"
            }
        }
    
    def detect(self, text: str) -> List[PIIMatch]:
        """Detect all PII in text with confidence scores"""
        matches = []
        
        for pii_type, pattern_list in self.patterns.items():
            for pattern, confidence in pattern_list:
                for match in pattern.finditer(text):
                    value = match.group()
                    
                    # Apply false positive filters
                    if self._is_false_positive(pii_type, value):
                        continue
                    
                    # Additional validation
                    validated_confidence = self._validate_match(
                        pii_type, value, confidence
                    )
                    
                    if validated_confidence >= self.confidence_threshold:
                        matches.append(PIIMatch(
                            pii_type=pii_type,
                            value=value,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=validated_confidence
                        ))
        
        # Remove overlapping matches (keep highest confidence)
        return self._deduplicate_overlapping(matches)
    
    def _is_false_positive(self, pii_type: PIIType, value: str) -> bool:
        filters = self.false_positive_filters.get(pii_type, set())
        return value in filters
    
    def _validate_match(
        self, 
        pii_type: PIIType, 
        value: str, 
        base_confidence: float
    ) -> float:
        """Additional validation logic per PII type"""
        if pii_type == PIIType.CREDIT_CARD:
            # Implement Luhn algorithm check
            if not self._luhn_check(value.replace(" ", "").replace("-", "")):
                return 0.0
        
        if pii_type == PIIType.IP_ADDRESS:
            # Validate IP ranges
            octets = [int(x) for x in value.split(".")]
            if any(o > 255 for o in octets):
                return 0.0
        
        return base_confidence
    
    def _luhn_check(self, card_number: str) -> bool