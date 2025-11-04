# Data Residency & Compliance Requirements for LLM Systems

## Core Concepts

Data residency and compliance in LLM systems refers to the architectural and operational requirements that govern where data is stored, processed, and transmitted when using AI models—and the legal, regulatory, and contractual obligations that constrain these operations. Unlike traditional database systems where data residency is primarily about storage location, LLM systems introduce unique challenges: data flows through prompts, embeddings, fine-tuning datasets, model weights, inference logs, and cached results across multiple jurisdictions and processing boundaries.

### Traditional vs. Modern Data Flow

**Traditional Three-Tier Application:**

```python
# Traditional architecture - clear data boundaries
class TraditionalApp:
    def __init__(self, db_region: str):
        self.db = Database(region=db_region)  # Data stays in region
        self.cache = RedisCache(region=db_region)
        
    def process_user_request(self, user_id: str, query: str) -> dict:
        # Data retrieved and processed in same region
        user_data = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
        result = self.business_logic(user_data, query)
        self.cache.set(f"result:{user_id}", result, ttl=3600)
        return result
    
    def business_logic(self, user_data: dict, query: str) -> dict:
        # Processing happens in-region, deterministic data flow
        return {"status": "processed", "data": user_data}
```

**LLM-Augmented Architecture:**

```python
from typing import List, Optional
import hashlib
import json
from datetime import datetime
from enum import Enum

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    REGULATED = "regulated"  # PII, PHI, financial data

class ComplianceViolation(Exception):
    pass

class LLMApp:
    def __init__(
        self, 
        db_region: str,
        llm_endpoint_region: str,
        embedding_region: str,
        allowed_data_classifications: List[DataClassification]
    ):
        self.db = Database(region=db_region)
        self.llm_client = LLMClient(endpoint_region=llm_endpoint_region)
        self.embedding_service = EmbeddingService(region=embedding_region)
        self.vector_db = VectorDatabase(region=db_region)
        self.allowed_classifications = allowed_data_classifications
        self.audit_log = AuditLog(region=db_region)
        
    def process_user_request(
        self, 
        user_id: str, 
        query: str,
        data_classification: DataClassification
    ) -> dict:
        """
        Data now flows through multiple processing boundaries:
        1. User data retrieved from regional DB
        2. Query + context sent to LLM endpoint (potentially different region)
        3. Embeddings generated (another potential region)
        4. Results stored in vector DB
        5. Inference logs retained by LLM provider
        """
        
        # Compliance gate - check before any processing
        if data_classification not in self.allowed_classifications:
            raise ComplianceViolation(
                f"Data classification {data_classification} not allowed for LLM processing"
            )
        
        # Retrieve user data - this stays in region
        user_data = self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
        
        # CRITICAL DECISION POINT: What data crosses regional boundaries?
        if data_classification == DataClassification.REGULATED:
            # Option 1: Don't send regulated data to LLM
            sanitized_context = self._sanitize_regulated_data(user_data)
            prompt = self._build_prompt(sanitized_context, query)
        else:
            # Option 2: Full context to LLM (data leaves region)
            prompt = self._build_prompt(user_data, query)
        
        # Data crosses boundary to LLM provider's infrastructure
        llm_response = self.llm_client.complete(prompt)
        
        # Embedding generation - another boundary crossing
        embedding = self.embedding_service.embed(query)
        
        # Store results and audit trail
        self.vector_db.store(embedding, llm_response, metadata={
            "user_id": user_id,
            "classification": data_classification.value,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self.audit_log.record({
            "user_id": user_id,
            "data_classification": data_classification.value,
            "llm_endpoint": self.llm_client.endpoint_region,
            "data_sent_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "response_hash": hashlib.sha256(llm_response.encode()).hexdigest()
        })
        
        return {"response": llm_response, "audit_id": self.audit_log.last_id}
    
    def _sanitize_regulated_data(self, user_data: dict) -> dict:
        """Remove PII/PHI before sending to LLM"""
        # This is where compliance actually happens
        sanitized = {
            k: v for k, v in user_data.items() 
            if k not in ['ssn', 'medical_record', 'credit_card', 'email']
        }
        return sanitized
```

### Key Insights That Change Engineering Thinking

**1. Data Leaves Your Control Multiple Times**

In traditional systems, you control the entire data path. With LLMs, data crosses boundaries you don't control:
- Training data may be retained by providers for model improvement
- Inference logs often stored for 30-90 days minimum
- Embeddings persist in vector databases
- Cache layers at CDN/provider edge locations
- Model fine-tuning creates persistent copies

**2. "Processed" Doesn't Mean "Deleted"**

The LLM doesn't just process and discard your data. It may be:
- Used in future training runs (unless explicitly opted out)
- Stored for abuse monitoring
- Cached for performance optimization
- Logged for debugging and quality improvement

**3. Compliance Requires Architectural Changes, Not Configuration**

You cannot configure your way to compliance with regulated data and third-party LLMs. You must architect different data flows:

```python
from typing import Protocol
from abc import abstractmethod

class LLMStrategy(Protocol):
    @abstractmethod
    def process(self, data: dict, query: str) -> str:
        """Process data according to compliance requirements"""
        pass

class PublicCloudLLM:
    """For non-sensitive data - use managed service"""
    def process(self, data: dict, query: str) -> str:
        # Data can leave your infrastructure
        return external_llm_api.complete(
            prompt=f"Context: {data}\n\nQuery: {query}"
        )

class PrivateDeploymentLLM:
    """For sensitive data - self-hosted inference"""
    def __init__(self, model_path: str, gpu_region: str):
        self.model = load_model_in_region(model_path, gpu_region)
    
    def process(self, data: dict, query: str) -> str:
        # Data never leaves your VPC/region
        return self.model.generate(
            prompt=f"Context: {data}\n\nQuery: {query}",
            max_tokens=500
        )

class HybridLLM:
    """Route based on data classification"""
    def __init__(self):
        self.public_llm = PublicCloudLLM()
        self.private_llm = PrivateDeploymentLLM(
            model_path="/models/llama-7b",
            gpu_region="eu-central-1"
        )
    
    def process(
        self, 
        data: dict, 
        query: str,
        classification: DataClassification
    ) -> str:
        if classification in [DataClassification.REGULATED, DataClassification.CONFIDENTIAL]:
            return self.private_llm.process(data, query)
        else:
            return self.public_llm.process(data, query)
```

### Why This Matters NOW

**Regulatory Enforcement Has Begun:**
- GDPR fines for AI systems exceeded €2B in 2023
- HIPAA violation cases specifically citing LLM data leakage
- Financial services regulators issuing specific AI data handling requirements
- China's Personal Information Protection Law (PIPL) requires in-country processing

**Technical Debt Compounds Rapidly:**
- Embeddings generated from regulated data cannot be easily "deleted"
- Fine-tuned models may encode sensitive information in weights
- Audit trails must prove data never left jurisdiction—impossible to retrofit
- Migration costs from non-compliant to compliant architecture often exceed 6-12 months

## Technical Components

### 1. Data Classification and Boundary Enforcement

**Technical Explanation:**

Every piece of data entering your LLM pipeline must be classified and routed through appropriate processing boundaries. This requires runtime enforcement, not just policy documentation.

```python
from dataclasses import dataclass
from typing import Set, Callable, Optional
import re

@dataclass
class DataBoundary:
    """Defines what can cross a processing boundary"""
    name: str
    allowed_regions: Set[str]
    allowed_classifications: Set[DataClassification]
    data_validators: List[Callable[[str], bool]]
    retention_policy_days: Optional[int]

class BoundaryEnforcer:
    """Runtime enforcement of data boundaries"""
    
    def __init__(self):
        self.boundaries = {
            "managed_llm_api": DataBoundary(
                name="managed_llm_api",
                allowed_regions={"us-east-1", "us-west-2", "eu-west-1"},
                allowed_classifications={
                    DataClassification.PUBLIC,
                    DataClassification.INTERNAL
                },
                data_validators=[
                    self._no_email_addresses,
                    self._no_ssn,
                    self._no_credit_cards
                ],
                retention_policy_days=30
            ),
            "private_inference": DataBoundary(
                name="private_inference",
                allowed_regions={"eu-central-1"},  # GDPR compliance
                allowed_classifications=set(DataClassification),  # All allowed
                data_validators=[],  # No restrictions when self-hosted
                retention_policy_days=90
            ),
            "embedding_service": DataBoundary(
                name="embedding_service",
                allowed_regions={"eu-central-1", "eu-west-1"},
                allowed_classifications={
                    DataClassification.PUBLIC,
                    DataClassification.INTERNAL,
                    DataClassification.CONFIDENTIAL
                },
                data_validators=[self._no_ssn, self._no_phi],
                retention_policy_days=None  # Indefinite
            )
        }
    
    def can_cross_boundary(
        self,
        data: str,
        classification: DataClassification,
        source_region: str,
        destination_boundary: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if data can cross to destination boundary.
        Returns (allowed, reason_if_denied)
        """
        boundary = self.boundaries.get(destination_boundary)
        if not boundary:
            return False, f"Unknown boundary: {destination_boundary}"
        
        # Check classification
        if classification not in boundary.allowed_classifications:
            return False, f"Classification {classification} not allowed for {destination_boundary}"
        
        # Check region
        if source_region not in boundary.allowed_regions:
            return False, f"Region {source_region} not allowed for {destination_boundary}"
        
        # Run validators
        for validator in boundary.data_validators:
            if not validator(data):
                return False, f"Data validation failed: {validator.__name__}"
        
        return True, None
    
    @staticmethod
    def _no_email_addresses(data: str) -> bool:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return not re.search(email_pattern, data)
    
    @staticmethod
    def _no_ssn(data: str) -> bool:
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        return not re.search(ssn_pattern, data)
    
    @staticmethod
    def _no_credit_cards(data: str) -> bool:
        # Simplified Luhn algorithm check
        cc_pattern = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        return not re.search(cc_pattern, data)
    
    @staticmethod
    def _no_phi(data: str) -> bool:
        # Check for common PHI patterns
        phi_patterns = [
            r'\bMRN[:\s]+\d+\b',  # Medical record numbers
            r'\b(diagnosis|diagnosed with|condition)[:\s]+\w+\b'
        ]
        return not any(re.search(p, data, re.IGNORECASE) for p in phi_patterns)

# Usage in production
class CompliantLLMService:
    def __init__(self):
        self.enforcer = BoundaryEnforcer()
        self.audit_log = AuditLog()
    
    def process_with_compliance(
        self,
        user_query: str,
        context_data: str,
        classification: DataClassification,
        source_region: str
    ) -> dict:
        # Attempt to use managed API first (cheaper, faster)
        can_use_api, denial_reason = self.enforcer.can_cross_boundary(
            data=context_data,
            classification=classification,
            source_region=source_region,
            destination_boundary="managed_llm_api"
        )
        
        if can_use_api:
            response = self._call_managed_api(user_query, context_data)
            processing_method = "managed_api"
        else:
            # Fall back to private inference
            self.audit_log.record({
                "event": "boundary_denied",
                "reason": denial_reason,
                "fallback": "private_inference"
            })
            response = self._call_private_inference(user_query, context_data)
            processing_method = "private_inference"
        
        return {
            "response": response,
            "processing_method": processing_method,
            "compliance_verified": True
        }
```

**Practical Implications:**

- Performance penalty: Private inference typically 3-10x slower than managed APIs
- Cost impact: Self-hosted inference costs $0.50-$2.00 per 1M tokens vs $0.002-$0.02 for managed
- Operational complexity: Managing GPU infrastructure vs API keys
- Real-time classification required—cannot batch process without classification

**Real Constraints:**

You cannot retroactively change where data was processed. Once context containing PII is sent to a managed API, you must assume:
- It's stored in provider logs for minimum retention period
- It may have crossed geographic boundaries
- You cannot fulfill "right to deletion" requests with certainty

### 2. Model Weight and Training Data Provenance

**Technical Explanation:**

Model weights encode information from training data. For compliance, you must track:
- What data was used to train/fine-tune models
- Where that training occurred
- Whether regulated data was included
- Ability to prove data lineage for audits

```python
from datetime import datetime
from typing import List, Dict, Any
import hashlib
import json

@dataclass
class TrainingDataset:
    dataset_id: str
    data_classification: DataClassification
    source_regions: Set[str]
    record_count: int
    data_hash: str  # Hash of dataset for verification
    contains_pii: bool
    consent_obtained: bool
    retention_expires: Optional[datetime]

@dataclass
class ModelProvenance:
    model_id: str
    base_model: str
    training_datasets: List[TrainingDataset]
    fine_tuning_datasets: List[TrainingDataset]
    training_region: str
    training_timestamp: datetime
    model_weights_hash: str
    compliance_certifications: List[str]
    