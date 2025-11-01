# Infrastructure-as-Code Generation with LLMs

## Core Concepts

Infrastructure-as-Code (IaC) generation using LLMs represents a paradigm shift from template-based provisioning to intent-driven infrastructure specification. Instead of manually writing Terraform, CloudFormation, or Kubernetes manifests, engineers provide high-level requirements and constraints, letting LLMs generate syntactically correct, production-ready configurations that encode best practices.

### Traditional vs. LLM-Driven Approach

```python
# Traditional: Manual Terraform authoring
# Engineer writes ~200 lines, references docs, handles edge cases manually

"""
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "production-vpc"
  }
}

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "private-subnet-${count.index + 1}"
  }
}
# ... 150+ more lines for security groups, routing, NAT gateways, etc.
"""

# LLM-Driven: Intent specification
from typing import Dict, Any, List
import anthropic
import json

def generate_infrastructure(
    requirements: Dict[str, Any],
    constraints: List[str],
    provider: str = "terraform"
) -> str:
    """
    Generate IaC from high-level requirements.
    
    Args:
        requirements: Structured infrastructure needs
        constraints: Security, compliance, cost constraints
        provider: IaC tool (terraform, pulumi, cloudformation)
    
    Returns:
        Complete, validated IaC configuration
    """
    client = anthropic.Anthropic()
    
    prompt = f"""Generate production-ready {provider} configuration for:

Requirements:
{json.dumps(requirements, indent=2)}

Constraints:
{chr(10).join(f'- {c}' for c in constraints)}

Include:
1. All necessary resources with proper dependencies
2. Security best practices (least privilege, encryption)
3. High availability across multiple AZs
4. Monitoring and logging configuration
5. Cost optimization (appropriate instance sizing)
6. Inline comments explaining design decisions

Output only valid {provider} HCL, no markdown wrappers."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Usage
requirements = {
    "environment": "production",
    "workload": "web application",
    "expected_traffic": "10K requests/minute",
    "database": "PostgreSQL with read replicas",
    "caching": "Redis cluster",
    "regions": ["us-east-1"]
}

constraints = [
    "PCI-DSS compliance required",
    "Budget: $2000/month",
    "99.9% uptime SLA",
    "Data must be encrypted at rest and in transit"
]

terraform_config = generate_infrastructure(requirements, constraints)
# Output: 200+ lines of Terraform with VPC, subnets, security groups,
# RDS with encryption, ElastiCache, ALB, auto-scaling, CloudWatch alarms
```

The measurable difference: **5 minutes to specify intent vs. 2-3 hours manual authoring**, with fewer security misconfigurations and built-in best practices.

### Why This Matters Now

Three converging factors make LLM-based IaC generation critical:

1. **Configuration complexity explosion**: Modern applications require 10-20x more infrastructure resources than five years ago (service mesh, observability, secrets management, compliance controls). Manual management doesn't scale.

2. **Security-by-default is non-negotiable**: LLMs trained on security best practices generate configurations with encryption, least privilege, and audit logging by default—things humans forget under pressure.

3. **Infrastructure literacy gap**: Platform teams can't bottleneck on infrastructure experts. LLMs democratize infrastructure provisioning while maintaining consistency through engineered prompts and validation.

### Key Insight That Changes Thinking

**IaC generation isn't about replacing infrastructure engineers—it's about elevating them from syntax writers to constraint architects.** The valuable skill becomes defining comprehensive requirement specifications and validation frameworks, not remembering Terraform resource argument syntax.

## Technical Components

### 1. Structured Requirement Specification

LLMs need unambiguous, machine-parseable requirements to generate deterministic infrastructure. Free-form descriptions produce inconsistent results.

**Technical Explanation:**

Use JSON schemas to enforce requirement completeness. This transforms IaC generation from probabilistic text generation to constrained, validated synthesis.

```python
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum

class Environment(str, Enum):
    DEV = "development"
    STAGING = "staging"
    PROD = "production"

class ComputeRequirements(BaseModel):
    type: Literal["containers", "serverless", "vms"]
    min_instances: int = Field(ge=1, le=100)
    max_instances: int = Field(ge=1, le=1000)
    cpu_cores: int = Field(ge=1, le=64)
    memory_gb: int = Field(ge=1, le=256)
    
    @validator('max_instances')
    def max_greater_than_min(cls, v, values):
        if 'min_instances' in values and v < values['min_instances']:
            raise ValueError('max_instances must be >= min_instances')
        return v

class DatabaseRequirements(BaseModel):
    engine: Literal["postgres", "mysql", "mongodb"]
    version: str
    storage_gb: int = Field(ge=20, le=16000)
    backup_retention_days: int = Field(ge=1, le=35)
    multi_az: bool = True
    read_replicas: int = Field(ge=0, le=5, default=0)

class NetworkRequirements(BaseModel):
    vpc_cidr: str = Field(regex=r'^\d{1,3}(\.\d{1,3}){3}/\d{1,2}$')
    availability_zones: int = Field(ge=2, le=6)
    public_subnets: bool = False
    private_subnets: bool = True
    nat_gateway: bool = True

class SecurityRequirements(BaseModel):
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    compliance_frameworks: List[Literal["PCI-DSS", "HIPAA", "SOC2", "GDPR"]]
    secrets_management: bool = True
    vulnerability_scanning: bool = True

class InfrastructureSpec(BaseModel):
    """Complete, validated infrastructure specification."""
    project_name: str = Field(min_length=3, max_length=50)
    environment: Environment
    cloud_provider: Literal["aws", "gcp", "azure"]
    region: str
    compute: ComputeRequirements
    database: Optional[DatabaseRequirements]
    network: NetworkRequirements
    security: SecurityRequirements
    tags: dict[str, str] = Field(default_factory=dict)
    
    @validator('tags')
    def required_tags(cls, v, values):
        required = {'Environment', 'ManagedBy', 'CostCenter'}
        if not required.issubset(set(v.keys())):
            raise ValueError(f'Missing required tags: {required - set(v.keys())}')
        return v

# Usage with validation
spec = InfrastructureSpec(
    project_name="payment-api",
    environment=Environment.PROD,
    cloud_provider="aws",
    region="us-east-1",
    compute=ComputeRequirements(
        type="containers",
        min_instances=3,
        max_instances=20,
        cpu_cores=4,
        memory_gb=16
    ),
    database=DatabaseRequirements(
        engine="postgres",
        version="15.3",
        storage_gb=500,
        backup_retention_days=30,
        multi_az=True,
        read_replicas=2
    ),
    network=NetworkRequirements(
        vpc_cidr="10.0.0.0/16",
        availability_zones=3,
        public_subnets=True,
        private_subnets=True
    ),
    security=SecurityRequirements(
        compliance_frameworks=["PCI-DSS", "SOC2"],
        secrets_management=True
    ),
    tags={
        "Environment": "production",
        "ManagedBy": "terraform",
        "CostCenter": "payments"
    }
)

# Converts to validated JSON for LLM consumption
validated_requirements = spec.model_dump_json(indent=2)
```

**Practical Implications:**

- **Determinism**: Same spec always generates functionally equivalent infrastructure
- **Validation**: Catches errors before expensive LLM calls
- **Auditability**: Requirements are versioned, reviewable artifacts
- **Testing**: Generate test fixtures programmatically

**Real Constraints:**

Schema design is critical. Too rigid and you block valid use cases; too loose and LLMs make dangerous assumptions. Start with strict schemas, relax based on production feedback.

### 2. Multi-Stage Generation Pipeline

Production IaC generation requires iterative refinement: base generation → security hardening → cost optimization → validation.

```python
from dataclasses import dataclass
from typing import Protocol
import subprocess
import tempfile
import os

class IaCValidator(Protocol):
    """Interface for IaC validation tools."""
    def validate(self, code: str) -> tuple[bool, List[str]]: ...

@dataclass
class GenerationResult:
    code: str
    warnings: List[str]
    estimated_monthly_cost: float
    security_score: float
    
class TerraformValidator:
    """Validates Terraform syntax and runs security checks."""
    
    def validate(self, code: str) -> tuple[bool, List[str]]:
        errors = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tf_file = os.path.join(tmpdir, "main.tf")
            with open(tf_file, 'w') as f:
                f.write(code)
            
            # Syntax validation
            result = subprocess.run(
                ["terraform", "fmt", "-check", tmpdir],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                errors.append(f"Formatting issues: {result.stderr}")
            
            result = subprocess.run(
                ["terraform", "validate", tmpdir],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                errors.append(f"Validation failed: {result.stderr}")
            
            # Security scanning (using tfsec as example)
            result = subprocess.run(
                ["tfsec", tmpdir, "--format", "json"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                security_issues = json.loads(result.stdout)
                errors.extend([
                    f"{issue['rule_id']}: {issue['description']}"
                    for issue in security_issues.get('results', [])
                ])
        
        return len(errors) == 0, errors

class IaCGenerationPipeline:
    """Multi-stage pipeline for production-ready IaC generation."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.validator = TerraformValidator()
    
    def generate(
        self,
        spec: InfrastructureSpec,
        max_iterations: int = 3
    ) -> GenerationResult:
        """Generate IaC with iterative refinement."""
        
        # Stage 1: Base generation
        code = self._generate_base(spec)
        
        # Stage 2: Iterative validation and fixing
        for iteration in range(max_iterations):
            is_valid, errors = self.validator.validate(code)
            if is_valid:
                break
            
            code = self._fix_issues(code, errors)
        
        if not is_valid:
            raise ValueError(f"Failed to generate valid IaC after {max_iterations} attempts")
        
        # Stage 3: Security hardening
        code = self._apply_security_hardening(code, spec.security)
        
        # Stage 4: Cost optimization
        code, cost_estimate = self._optimize_costs(code, spec)
        
        # Stage 5: Final validation
        is_valid, warnings = self.validator.validate(code)
        security_score = self._calculate_security_score(code)
        
        return GenerationResult(
            code=code,
            warnings=warnings,
            estimated_monthly_cost=cost_estimate,
            security_score=security_score
        )
    
    def _generate_base(self, spec: InfrastructureSpec) -> str:
        """Initial infrastructure generation."""
        prompt = f"""Generate Terraform configuration for this specification:

{spec.model_dump_json(indent=2)}

Requirements:
1. Use latest provider versions
2. Include all required dependencies
3. Use variables for configurable values
4. Add outputs for important resource IDs
5. Include terraform backend configuration

Output only HCL code, no markdown."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _fix_issues(self, code: str, errors: List[str]) -> str:
        """Fix validation errors through LLM iteration."""
        prompt = f"""This Terraform code has validation errors:

```hcl
{code}
```

Errors:
{chr(10).join(f'- {e}' for e in errors)}

Fix all errors while preserving functionality. Output corrected HCL only."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _apply_security_hardening(
        self,
        code: str,
        security: SecurityRequirements
    ) -> str:
        """Apply security best practices based on compliance requirements."""
        hardening_rules = {
            "PCI-DSS": [
                "Enable VPC flow logs",
                "Enforce SSL/TLS 1.2+ only",
                "Enable CloudTrail with log file validation",
                "Encrypt all storage with customer-managed keys"
            ],
            "HIPAA": [
                "Enable encryption at rest for all data stores",
                "Configure audit logging for all data access",
                "Implement network segmentation",
                "Enable automated backups with encryption"
            ],
            "SOC2": [
                "Enable multi-factor authentication",
                "Implement least privilege access",
                "Configure security monitoring",
                "Enable change tracking"
            ]
        }
        
        required_controls = []
        for framework in security.compliance_frameworks:
            required_controls.extend(hardening_rules.get(framework, []))
        
        prompt = f"""Harden this Terraform code for compliance:

```hcl
{code}
```

Required security controls:
{chr(10).join(f'- {c}' for c in required_controls)}

Add necessary resources and configurations. Output complete HCL."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _optimize_costs(
        self,
        code: str,
        spec: InfrastructureSpec
    ) -> tuple[str, float]:
        """Optimize infrastructure costs while meeting requirements."""
        prompt = f"""Optimize this Terraform for cost efficiency: