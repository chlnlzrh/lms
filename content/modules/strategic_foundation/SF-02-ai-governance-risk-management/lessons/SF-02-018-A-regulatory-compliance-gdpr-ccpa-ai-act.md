# Regulatory Compliance for AI Systems: GDPR, CCPA, and the AI Act

## Core Concepts

Regulatory compliance in AI isn't about checking legal boxes—it's about architecting systems with specific technical capabilities that enable auditability, user control, and risk management. Three major regulations shape how you build AI systems: GDPR (General Data Protection Regulation) affects anyone processing EU citizens' data, CCPA (California Consumer Privacy Act) covers California residents, and the emerging EU AI Act classifies AI systems by risk level and imposes corresponding technical requirements.

### Engineering Analogy: Traditional vs. Compliance-Aware Architecture

**Traditional AI System:**
```python
from typing import Dict, Any
import json

class TraditionalAISystem:
    """Simple AI system without compliance considerations"""
    
    def __init__(self, model):
        self.model = model
        self.data_store = []
    
    def process_request(self, user_data: Dict[str, Any]) -> str:
        # Store everything indefinitely
        self.data_store.append(user_data)
        
        # Process with model
        result = self.model.predict(user_data['input'])
        
        # Return result
        return result
    
    def get_user_data(self, user_id: str) -> list:
        # No granular access control
        return [d for d in self.data_store if d.get('user_id') == user_id]
```

**Compliance-Aware AI System:**
```python
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json

class DataPurpose(Enum):
    """Explicit purposes for data processing - GDPR Article 5"""
    MODEL_INFERENCE = "model_inference"
    MODEL_TRAINING = "model_training"
    QUALITY_IMPROVEMENT = "quality_improvement"
    LEGAL_COMPLIANCE = "legal_compliance"

class ProcessingRecord:
    """Audit trail for all data processing - GDPR Article 30"""
    def __init__(self, user_id: str, purpose: DataPurpose, 
                 data_categories: List[str], retention_days: int):
        self.timestamp = datetime.utcnow()
        self.user_id = self._pseudonymize(user_id)
        self.purpose = purpose
        self.data_categories = data_categories
        self.retention_until = self.timestamp + timedelta(days=retention_days)
        self.processing_id = self._generate_id()
    
    def _pseudonymize(self, identifier: str) -> str:
        """One-way hash for privacy - GDPR Article 32"""
        return hashlib.sha256(identifier.encode()).hexdigest()
    
    def _generate_id(self) -> str:
        data = f"{self.user_id}{self.timestamp.isoformat()}{self.purpose.value}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

class ComplianceAwareAISystem:
    """AI system with built-in compliance capabilities"""
    
    def __init__(self, model, retention_policy: Dict[DataPurpose, int]):
        self.model = model
        self.data_store = {}  # user_id -> list of data points
        self.processing_log = []  # Audit trail
        self.retention_policy = retention_policy
        self.consent_registry = {}  # user_id -> purposes consented to
    
    def process_request(self, user_id: str, user_data: Dict[str, Any], 
                       purpose: DataPurpose) -> Dict[str, Any]:
        """Process with purpose limitation and audit trail"""
        
        # Check consent (GDPR Article 6, CCPA opt-in for sensitive data)
        if not self._has_consent(user_id, purpose):
            return {
                "error": "insufficient_consent",
                "required_purpose": purpose.value
            }
        
        # Log processing activity (GDPR Article 30)
        record = ProcessingRecord(
            user_id=user_id,
            purpose=purpose,
            data_categories=list(user_data.keys()),
            retention_days=self.retention_policy[purpose]
        )
        self.processing_log.append(record)
        
        # Process only what's necessary (data minimization - GDPR Article 5)
        minimal_data = self._minimize_data(user_data, purpose)
        
        # Store with retention metadata
        self._store_with_retention(user_id, minimal_data, record.retention_until)
        
        # Process with model
        result = self.model.predict(minimal_data['input'])
        
        return {
            "result": result,
            "processing_id": record.processing_id,
            "retention_until": record.retention_until.isoformat()
        }
    
    def _has_consent(self, user_id: str, purpose: DataPurpose) -> bool:
        """Check if user consented to this purpose"""
        return purpose in self.consent_registry.get(user_id, set())
    
    def _minimize_data(self, data: Dict[str, Any], 
                      purpose: DataPurpose) -> Dict[str, Any]:
        """Keep only fields necessary for the purpose"""
        # Define what each purpose needs
        purpose_requirements = {
            DataPurpose.MODEL_INFERENCE: ['input'],
            DataPurpose.MODEL_TRAINING: ['input', 'expected_output'],
            DataPurpose.QUALITY_IMPROVEMENT: ['input', 'actual_output', 'feedback']
        }
        
        required_fields = purpose_requirements.get(purpose, [])
        return {k: v for k, v in data.items() if k in required_fields}
    
    def _store_with_retention(self, user_id: str, data: Dict[str, Any], 
                             retention_until: datetime):
        """Store data with automatic deletion date"""
        if user_id not in self.data_store:
            self.data_store[user_id] = []
        
        self.data_store[user_id].append({
            'data': data,
            'retention_until': retention_until,
            'stored_at': datetime.utcnow()
        })
    
    def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Right to erasure - GDPR Article 17, CCPA deletion right"""
        deleted_count = 0
        
        if user_id in self.data_store:
            deleted_count = len(self.data_store[user_id])
            del self.data_store[user_id]
        
        # Remove from consent registry
        if user_id in self.consent_registry:
            del self.consent_registry[user_id]
        
        # Log the deletion
        self.processing_log.append({
            'action': 'user_data_deleted',
            'user_id': hashlib.sha256(user_id.encode()).hexdigest(),
            'timestamp': datetime.utcnow(),
            'records_deleted': deleted_count
        })
        
        return {
            'status': 'completed',
            'records_deleted': deleted_count,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Right to data portability - GDPR Article 20, CCPA disclosure"""
        user_data = self.data_store.get(user_id, [])
        user_processing = [
            log for log in self.processing_log 
            if getattr(log, 'user_id', None) == 
               hashlib.sha256(user_id.encode()).hexdigest()
        ]
        
        return {
            'user_id': user_id,
            'data_points': [
                {
                    'data': point['data'],
                    'stored_at': point['stored_at'].isoformat(),
                    'retention_until': point['retention_until'].isoformat()
                }
                for point in user_data
            ],
            'processing_history': [
                {
                    'timestamp': record.timestamp.isoformat(),
                    'purpose': record.purpose.value,
                    'data_categories': record.data_categories
                }
                for record in user_processing
            ],
            'export_timestamp': datetime.utcnow().isoformat()
        }
    
    def cleanup_expired_data(self) -> int:
        """Automated retention enforcement"""
        now = datetime.utcnow()
        deleted_count = 0
        
        for user_id in list(self.data_store.keys()):
            original_count = len(self.data_store[user_id])
            self.data_store[user_id] = [
                point for point in self.data_store[user_id]
                if point['retention_until'] > now
            ]
            deleted_count += original_count - len(self.data_store[user_id])
            
            # Remove user entry if no data remains
            if not self.data_store[user_id]:
                del self.data_store[user_id]
        
        return deleted_count
```

### Key Insights

1. **Compliance is architecture, not afterthought**: You can't bolt GDPR compliance onto an existing system. Rights like deletion and data export require designing your data model from scratch with these operations in mind.

2. **Purpose limitation drives design**: Every data collection needs an explicit, documented purpose. This forces you to think about data flows and prevents scope creep that creates compliance risks.

3. **Audit trails are mandatory infrastructure**: Both GDPR Article 30 and the AI Act require detailed records of processing activities. This isn't optional logging—it's a core system requirement that affects performance and storage.

4. **AI Act introduces risk-based requirements**: The AI Act classifies systems into risk categories (unacceptable, high, limited, minimal), with high-risk systems requiring conformity assessments, technical documentation, and human oversight mechanisms.

### Why This Matters Now

Penalties are substantial and actively enforced: GDPR fines reach €20 million or 4% of global revenue (whichever is higher), CCPA allows $7,500 per intentional violation, and the AI Act proposes up to €30 million or 6% of global revenue. More importantly, non-compliant systems face operational shutdowns. The AI Act, expected to be enforced by 2025-2026, will require retroactive compliance for deployed systems classified as high-risk.

## Technical Components

### 1. Data Subject Rights Implementation

**Technical Explanation:**
GDPR Articles 15-22 and CCPA Sections 1798.100-1798.125 grant users specific rights over their data: access, rectification, erasure, portability, and restriction of processing. These aren't just API endpoints—they require specific data architecture decisions.

**Practical Implementation:**
```python
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import sqlite3

@dataclass
class DataSubjectRequest:
    """Represents a user's rights request"""
    request_id: str
    user_id: str
    request_type: str  # 'access', 'delete', 'rectify', 'restrict', 'export'
    timestamp: datetime
    status: str  # 'pending', 'completed', 'rejected'
    completion_deadline: datetime  # 30 days for GDPR, 45 days for CCPA

class DataSubjectRightsManager:
    """Centralized handler for all data subject rights"""
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Create tables with compliance requirements in mind"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                user_id TEXT,
                data_category TEXT,
                data_value TEXT,
                purpose TEXT,
                legal_basis TEXT,  -- 'consent', 'contract', 'legitimate_interest'
                collected_at TIMESTAMP,
                retention_until TIMESTAMP,
                is_restricted BOOLEAN DEFAULT 0,
                PRIMARY KEY (user_id, data_category)
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS processing_purposes (
                user_id TEXT,
                purpose TEXT,
                legal_basis TEXT,
                consented_at TIMESTAMP,
                consent_version TEXT,
                is_active BOOLEAN DEFAULT 1,
                PRIMARY KEY (user_id, purpose)
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS data_requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT,
                request_type TEXT,
                requested_at TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT,
                response_data TEXT
            )
        ''')
        self.conn.commit()
    
    def handle_access_request(self, user_id: str) -> Dict[str, Any]:
        """GDPR Article 15 - Right of access"""
        cursor = self.conn.execute('''
            SELECT data_category, data_value, purpose, legal_basis, 
                   collected_at, retention_until
            FROM user_data
            WHERE user_id = ? AND is_restricted = 0
        ''', (user_id,))
        
        data_inventory = []
        for row in cursor.fetchall():
            data_inventory.append({
                'category': row[0],
                'value': row[1],
                'purpose': row[2],
                'legal_basis': row[3],
                'collected_at': row[4],
                'retention_until': row[5]
            })
        
        # Get processing purposes
        cursor = self.conn.execute('''
            SELECT purpose, legal_basis, consented_at
            FROM processing_purposes
            WHERE user_id = ? AND is_active = 1
        ''', (user_id,))
        
        purposes = [
            {
                'purpose': row[0],
                'legal_basis': row[1],
                'consented_at': row[2]
            }
            for row in cursor.fetchall()
        ]
        
        return {
            'user_id': user_id,
            'data_inventory': data_inventory,
            'processing_purposes': purposes,
            'data_recipients': self._get_data_recipients(user_id),
            'retention_period': self._get_retention_info(user_id),
            'rights_information': {
                'right_to_rectification': True,
                'right_to_erasure': True,
                'right_to_restrict': True,
                'right_to_portability': True,
                'right_to_object': True
            }
        }
    
    def handle_erasure_request(self, user_id: str, 
                              reason: Optional[str] = None) -> Dict[str, Any]:
        """GDPR Article 17 - Right to erasure"""
        # Check if erasure is permissible
        legal_obligations = self._check_legal_obligations(user_id)
        if legal_obligations:
            return {
                'status': 'rejected',
                'reason': 'legal_obligation',
                'details': legal_obligations
            }
        
        # Mark data for deletion (soft delete for audit trail)
        cursor = self.conn.execute('''
            UPDATE user_data 
            SET data_value = '[DELETED]',
                retention_until = datetime('now')
            WHERE user_id = ?
        ''', (user_id,))
        
        deleted_count = cursor.rowcount
        
        # Deactivate processing purposes
        self.conn.execute('''
            UPDATE processing_purposes 
            SET is_active = 0
            WHERE user_id = ?
        ''', (user_id,))
        
        self.conn.commit()
        
        # Notify third parties (GDPR Article 19)
        self._notify_third_parties_of_erasure(user_id)
        
        return {
            'status': 'completed',
            'records_deleted': deleted_count,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id
        }
    
    def handle_portability_request(self, user_id: str) -> Dict[str, Any]:
        """GDPR Article 20