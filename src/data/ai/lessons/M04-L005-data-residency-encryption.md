# Data Residency & Encryption for LLM Systems

## Core Concepts

Data residency and encryption in LLM systems address where your data physically exists and how it's protected throughout its lifecycle. Unlike traditional web applications where you control the entire stack, LLM systems introduce a fundamental architectural shift: your most sensitive data—customer conversations, proprietary documents, business logic embedded in prompts—flows through third-party infrastructure you don't control.

### The Architectural Shift

Traditional application architecture keeps data within your security perimeter:

```python
# Traditional architecture: data stays in your infrastructure
from typing import Dict
import hashlib

class TraditionalDataProcessor:
    """Data never leaves your infrastructure"""
    
    def __init__(self, database_connection: str):
        self.db = database_connection
    
    def process_sensitive_data(self, customer_data: Dict[str, str]) -> Dict[str, str]:
        # Hash PII before storing
        processed = {
            "customer_id": customer_data["id"],
            "email_hash": hashlib.sha256(
                customer_data["email"].encode()
            ).hexdigest(),
            "query": customer_data["query"]
        }
        # Store in your database
        self.store_locally(processed)
        return self.analyze_locally(processed)
    
    def store_locally(self, data: Dict[str, str]) -> None:
        # Your database, your servers, your jurisdiction
        pass
    
    def analyze_locally(self, data: Dict[str, str]) -> Dict[str, str]:
        # Processing happens in your infrastructure
        return {"result": "analyzed"}
```

Modern LLM architecture sends data to external providers:

```python
import os
from typing import Dict, List
import json

class LLMDataProcessor:
    """Data flows through third-party LLM providers"""
    
    def __init__(self, api_key: str, api_endpoint: str):
        self.api_key = api_key
        self.endpoint = api_endpoint
    
    def process_with_llm(self, customer_data: Dict[str, str]) -> str:
        # CRITICAL: This data leaves your infrastructure
        prompt = f"""
        Customer: {customer_data['email']}
        Account: {customer_data['account_id']}
        Query: {customer_data['query']}
        Purchase History: {customer_data['purchase_history']}
        
        Provide personalized recommendation.
        """
        
        # Data transmitted to third-party servers
        # - Crosses network boundaries
        # - Stored in provider's infrastructure (temporarily or permanently)
        # - Subject to provider's jurisdiction and policies
        # - May be used for model training (check ToS)
        response = self.call_llm_api(prompt)
        
        return response
    
    def call_llm_api(self, prompt: str) -> str:
        # Simplified API call
        # Real implementation uses requests/httpx
        return "recommendation"
```

The difference: in traditional systems, you control data locality. In LLM systems, data locality becomes a negotiation with your provider, governed by contracts, regional regulations, and technical capabilities.

### Why This Matters Now

Three converging forces make data residency and encryption urgent:

1. **Regulatory enforcement is accelerating**: GDPR fines averaged €17 million in 2023. Non-compliance with data localization requirements can shut down operations in entire regions.

2. **LLM providers operate globally distributed infrastructure**: Your API call might hit servers in Virginia, but the actual model inference could happen in Oregon, Singapore, or Ireland. Training data aggregation could occur anywhere.

3. **Prompt injection and data exfiltration attacks are mature**: Attackers can extract training data, leak conversation history, or manipulate outputs to reveal sensitive information from other users' sessions.

The engineering challenge: build LLM systems that deliver AI capabilities while maintaining regulatory compliance and security posture—without rewriting your application for every jurisdiction.

## Technical Components

### 1. Data Residency: Physical Location and Legal Jurisdiction

Data residency defines where data physically resides and which legal framework governs it. For LLM systems, this spans multiple layers:

**Technical explanation**: When you send a prompt to an LLM API, your data traverses multiple infrastructure layers: edge networks for routing, regional API gateways for load balancing, inference clusters for model execution, and potentially storage systems for logging, fine-tuning, or model improvement. Each layer may exist in different physical locations.

**Practical implications**:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

class DataRegion(Enum):
    """Supported data residency regions"""
    EU_WEST = "eu-west"
    US_EAST = "us-east"
    ASIA_PACIFIC = "asia-pacific"
    
@dataclass
class ResidencyConfig:
    """Configuration for data residency requirements"""
    primary_region: DataRegion
    allowed_regions: List[DataRegion]
    prohibited_regions: List[DataRegion]
    require_regional_inference: bool = True
    require_regional_storage: bool = True
    
class ResidencyAwareLLMClient:
    """LLM client with data residency enforcement"""
    
    def __init__(self, config: ResidencyConfig):
        self.config = config
        self.regional_endpoints = {
            DataRegion.EU_WEST: "https://api-eu.example.com/v1",
            DataRegion.US_EAST: "https://api-us.example.com/v1",
            DataRegion.ASIA_PACIFIC: "https://api-ap.example.com/v1",
        }
    
    def get_endpoint(self) -> str:
        """Select endpoint based on residency requirements"""
        endpoint = self.regional_endpoints.get(self.config.primary_region)
        if not endpoint:
            raise ValueError(
                f"No endpoint available for region {self.config.primary_region}"
            )
        return endpoint
    
    def validate_residency(self, data_region: DataRegion) -> None:
        """Validate data doesn't cross prohibited boundaries"""
        if data_region in self.config.prohibited_regions:
            raise ValueError(
                f"Data processing in {data_region} violates residency policy"
            )
        
        if (self.config.require_regional_inference and 
            data_region not in self.config.allowed_regions):
            raise ValueError(
                f"Regional inference required but {data_region} not allowed"
            )
    
    def process_with_residency(
        self, 
        prompt: str, 
        customer_region: DataRegion
    ) -> str:
        """Process data with residency constraints"""
        # Validate before sending
        self.validate_residency(customer_region)
        
        # Route to regional endpoint
        endpoint = self.get_endpoint()
        
        # Add residency headers
        headers = {
            "X-Data-Region": customer_region.value,
            "X-Require-Regional-Processing": "true"
        }
        
        # Make API call (simplified)
        # Real implementation would use requests with proper error handling
        return f"Processed in {self.config.primary_region.value}"

# Usage example
config = ResidencyConfig(
    primary_region=DataRegion.EU_WEST,
    allowed_regions=[DataRegion.EU_WEST],
    prohibited_regions=[DataRegion.US_EAST, DataRegion.ASIA_PACIFIC],
    require_regional_inference=True,
    require_regional_storage=True
)

client = ResidencyAwareLLMClient(config)

# This succeeds
response = client.process_with_residency(
    "Analyze customer sentiment", 
    DataRegion.EU_WEST
)

# This raises ValueError
try:
    response = client.process_with_residency(
        "Analyze customer sentiment",
        DataRegion.US_EAST
    )
except ValueError as e:
    print(f"Residency violation: {e}")
```

**Real constraints**: Regional endpoints add latency (50-200ms for cross-region routing), may have limited model availability, and cost 10-30% more than default endpoints. Not all providers offer regional guarantees—read the Data Processing Agreement (DPA) carefully.

### 2. Encryption in Transit: Protecting Data During Transmission

Data moving between your infrastructure and LLM providers must be encrypted to prevent interception.

**Technical explanation**: TLS 1.3 provides encryption for data in transit, but implementation details matter. Certificate validation, cipher suite selection, and connection pooling affect both security and performance.

```python
import httpx
import ssl
from typing import Dict, Optional
import certifi

class SecureLLMClient:
    """LLM client with strict transport security"""
    
    def __init__(
        self, 
        api_key: str,
        endpoint: str,
        min_tls_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_3
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        
        # Create SSL context with strict requirements
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.minimum_version = min_tls_version
        ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Disable insecure ciphers
        ssl_context.set_ciphers(
            'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
        )
        
        # Verify certificates strictly
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Create HTTP client with security settings
        self.client = httpx.Client(
            verify=ssl_context,
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10
            )
        )
    
    def send_prompt(
        self, 
        prompt: str,
        sanitize: bool = True
    ) -> Dict[str, str]:
        """Send prompt with encrypted transport"""
        
        # Remove potentially sensitive debugging info
        if sanitize:
            prompt = self._sanitize_prompt(prompt)
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Prevent caching of sensitive data
            "Cache-Control": "no-store, no-cache, must-revalidate",
            # Additional security headers
            "X-Content-Type-Options": "nosniff"
        }
        
        payload = {
            "prompt": prompt,
            "max_tokens": 500
        }
        
        try:
            # HTTPS ensures encryption in transit
            response = self.client.post(
                f"{self.endpoint}/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            # Don't log full request/response (may contain sensitive data)
            print(f"API error: {e.response.status_code}")
            raise
        except httpx.TransportError as e:
            print(f"Transport error: {type(e).__name__}")
            raise
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Remove debugging artifacts that might leak info"""
        # Remove common debugging patterns
        sanitized = prompt.replace("[DEBUG]", "")
        sanitized = sanitized.replace("[TRACE]", "")
        return sanitized.strip()
    
    def close(self) -> None:
        """Clean up connections"""
        self.client.close()

# Usage
client = SecureLLMClient(
    api_key="sk-...",
    endpoint="https://api.provider.com/v1"
)

try:
    response = client.send_prompt("Summarize quarterly results")
    print(response)
finally:
    client.close()
```

**Trade-offs**: TLS 1.3 adds 1-2ms latency per request. Strict certificate validation prevents MITM attacks but can cause issues with corporate proxies that perform SSL inspection. Connection pooling amortizes TLS handshake costs but requires careful management.

### 3. Encryption at Rest: Protecting Stored Data

Data stored by LLM providers (logs, fine-tuning datasets, embeddings) must be encrypted when not actively processed.

**Technical explanation**: Encryption at rest uses symmetric encryption (typically AES-256) to protect data on disk. Key management determines who can decrypt: provider-managed keys (PMK), customer-managed keys (CMK), or customer-provided keys (CPK).

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import os
import base64
from typing import Tuple

class ClientSideEncryption:
    """Encrypt data before sending to LLM provider"""
    
    def __init__(self, master_key: bytes):
        """
        Initialize with master key (32 bytes for AES-256)
        In production, retrieve from key management service
        """
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes for AES-256")
        self.master_key = master_key
    
    def encrypt_prompt(self, prompt: str) -> Tuple[str, bytes, bytes]:
        """
        Encrypt prompt before sending to LLM
        Returns: (encrypted_base64, iv, tag)
        """
        # Generate random IV (initialization vector)
        iv = os.urandom(12)  # 96 bits for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.master_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt
        prompt_bytes = prompt.encode('utf-8')
        ciphertext = encryptor.update(prompt_bytes) + encryptor.finalize()
        
        # Get authentication tag
        tag = encryptor.tag
        
        # Return base64-encoded ciphertext for transmission
        encrypted_b64 = base64.b64encode(ciphertext).decode('utf-8')
        
        return encrypted_b64, iv, tag
    
    def decrypt_response(
        self, 
        encrypted_b64: str, 
        iv: bytes, 
        tag: bytes
    ) -> str:
        """Decrypt LLM response"""
        # Decode from base64
        ciphertext = base64.b64decode(encrypted_b64)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.master_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt and verify
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext.decode('utf-8')

class EncryptedLLMWorkflow:
    """Complete workflow with client-side encryption"""
    
    def __init__(self, encryption: ClientSideEncryption):
        self.encryption = encryption
    
    def process_sensitive_data(self, sensitive_prompt: str) -> str:
        """
        Encrypt prompt, send to LLM, decrypt response
        LLM provider never sees plaintext
        """
        # Encrypt before transmission
        encrypted_prompt, iv, tag = self.encryption.encrypt_prompt(
            sensitive_prompt
        )
        
        # Metadata for LLM provider
        request_metadata = {
            "encrypted": True,
            "algorithm": "AES-256-GCM",
            "iv": base64.b64encode(iv).decode('utf-8'),
            "tag": base64.b64encode(tag).decode('utf-8')
        }
        
        # Send encrypted prompt to LLM
        # Provider processes encrypted data (limited utility)
        # Or you decrypt in your infrastructure after retrieval
        encrypted_response = self._call_llm_with_encrypted_data(
            