"""
Generate M0 lessons in parallel batches (3 at a time) with 10-second pauses between batches using OpenAI
"""
import os
import sys
import time
import asyncio
from datetime import datetime
from openai import AsyncOpenAI

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Read OpenAI API key
try:
    with open("openaiapikey.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except:
    api_key = None

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("="*80)
    print("ERROR: OpenAI API key not found!")
    print("="*80)
    print("Please add your OpenAI API key to openaiapikey.txt")
    print("The file exists but is currently empty.")
    print("\nTo fix:")
    print("1. Open openaiapikey.txt")
    print("2. Paste your OpenAI API key (starts with sk-)")
    print("3. Save the file")
    print("4. Run this script again")
    print("="*80)
    raise ValueError("OpenAI API key required in openaiapikey.txt")

client = AsyncOpenAI(api_key=api_key)
# Use gpt-4o-mini for faster/cheaper generation, or gpt-4o for higher quality
# Using GPT-5 Mini as primary, GPT-5 Nano as fallback
MODEL_NAME = "gpt-5-mini"  # Primary model
FALLBACK_MODEL = "gpt-5-nano"  # Fallback if gpt-5-mini not available

# Read the prompt template (it's a Python f-string template)
with open("LESSON_GENERATION_PROMPT_GENERIC.md", "r", encoding="utf-8") as f:
    prompt_template = f.read()

# Read the content structure (curriculum map) - will be referenced in the prompt
with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
    content_structure = f.read()

# Read the 10-section template structure if it exists separately
# (If it's in LESSON_GENERATION_PROMPT_GENERIC.md, we don't need it separately)
# For now, we'll let the prompt template reference it implicitly

# Load M1-M4 lessons from extracted file
import re
import json

# Read the extracted lessons file
with open("scripts/m1_m2_m3_m4_lessons.txt", "r", encoding="utf-8") as f:
    m1_m4_content = f.read()

# Extract the missing_lessons list using regex
lessons_match = re.search(r'missing_lessons = \[(.*?)\]', m1_m4_content, re.DOTALL)
if lessons_match:
    # Extract just the list content, not the variable assignment
    lessons_str = "[" + lessons_match.group(1) + "]"
    # Use eval safely here since we control the input
    m1_m4_lessons = eval(lessons_str)
else:
    m1_m4_lessons = []

# All remaining M0 lessons to generate (L017-L072) - 56 lessons total
# Imported from build_remaining_m0_lessons_fixed.py output
m0_remaining_lessons = [
    {"LESSON_CODE": "M00-L017", "LESSON_TITLE": "Architecture Fitness Functions & Continuous Verification", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L016 (Evolutionary Architecture), understanding of software metrics, continuous integration", "RELATED_LESSON_CODES": "M00-L016, M00-L018", "SPECIFIC_FOCUS": "Core Architectural Patterns"},
    {"LESSON_CODE": "M00-L018", "LESSON_TITLE": "Ubiquitous Language & Context Mapping", "COMPLEXITY": "F", "TIME": "45", "LIST_PREREQUISITES": "M00-L017 (Architecture Fitness Functions), understanding of domain-driven design basics", "RELATED_LESSON_CODES": "M00-L017, M00-L019", "SPECIFIC_FOCUS": "Bounded Contexts & Domain Modeling"},
    {"LESSON_CODE": "M00-L019", "LESSON_TITLE": "Entities, Aggregates, and Value Objects: Getting Invariants Right", "COMPLEXITY": "F", "TIME": "45", "LIST_PREREQUISITES": "M00-L018 (Ubiquitous Language & Context Mapping)", "RELATED_LESSON_CODES": "M00-L018, M00-L020", "SPECIFIC_FOCUS": "Bounded Contexts & Domain Modeling"},
    {"LESSON_CODE": "M00-L020", "LESSON_TITLE": "Aggregate Design Patterns (Consistency & Transaction Boundaries)", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L019 (Entities, Aggregates, and Value Objects: Getting Invariants Right)", "RELATED_LESSON_CODES": "M00-L019, M00-L021", "SPECIFIC_FOCUS": "Bounded Contexts & Domain Modeling"},
    {"LESSON_CODE": "M00-L021", "LESSON_TITLE": "Domain Events: Publishing, Ordering, and Replay", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L020 (Aggregate Design Patterns (Consistency & Transaction Boundaries))", "RELATED_LESSON_CODES": "M00-L020, M00-L022", "SPECIFIC_FOCUS": "Bounded Contexts & Domain Modeling"},
    {"LESSON_CODE": "M00-L022", "LESSON_TITLE": "Anti-Corruption Layers for Legacy & Third-Party Systems", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L021 (Domain Events: Publishing, Ordering, and Replay)", "RELATED_LESSON_CODES": "M00-L021, M00-L023", "SPECIFIC_FOCUS": "Bounded Contexts & Domain Modeling"},
    {"LESSON_CODE": "M00-L023", "LESSON_TITLE": "Context Maps: Shared Kernel, Customer–Supplier, Conformist", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L022 (Anti-Corruption Layers for Legacy & Third-Party Systems)", "RELATED_LESSON_CODES": "M00-L022, M00-L024", "SPECIFIC_FOCUS": "Bounded Contexts & Domain Modeling"},
    {"LESSON_CODE": "M00-L024", "LESSON_TITLE": "Event Storming to Discover Workflows and Hot Spots", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L023 (Context Maps: Shared Kernel, Customer–Supplier, Conformist)", "RELATED_LESSON_CODES": "M00-L023, M00-L025", "SPECIFIC_FOCUS": "Bounded Contexts & Domain Modeling"},
    {"LESSON_CODE": "M00-L025", "LESSON_TITLE": "Documenting Decisions with ADRs & Lightweight RFCs", "COMPLEXITY": "F", "TIME": "45", "LIST_PREREQUISITES": "M00-L024 (Event Storming to Discover Workflows and Hot Spots)", "RELATED_LESSON_CODES": "M00-L024, M00-L026", "SPECIFIC_FOCUS": "Bounded Contexts & Domain Modeling"},
    {"LESSON_CODE": "M00-L026", "LESSON_TITLE": "CAP & PACELC: Practical Impact on Design Choices", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L025 (Documenting Decisions with ADRs & Lightweight RFCs), understanding of distributed systems, consistency models", "RELATED_LESSON_CODES": "M00-L025, M00-L027", "SPECIFIC_FOCUS": "Consistency, Resiliency & Failure Semantics"},
    {"LESSON_CODE": "M00-L027", "LESSON_TITLE": "Consistency Models: Strong, Eventual, Causal, and Read-Your-Writes", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L026 (CAP & PACELC: Practical Impact on Design Choices), understanding of distributed systems, consistency models", "RELATED_LESSON_CODES": "M00-L026, M00-L028", "SPECIFIC_FOCUS": "Consistency, Resiliency & Failure Semantics"},
    {"LESSON_CODE": "M00-L028", "LESSON_TITLE": "Designing Idempotent APIs, Jobs, and Webhooks", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L027 (Consistency Models: Strong, Eventual, Causal, and Read-Your-Writes), understanding of distributed systems, consistency models", "RELATED_LESSON_CODES": "M00-L027, M00-L029", "SPECIFIC_FOCUS": "Consistency, Resiliency & Failure Semantics"},
    {"LESSON_CODE": "M00-L029", "LESSON_TITLE": "Retry Policies: Exponential Backoff, Jitter, and Hedging", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L028 (Designing Idempotent APIs, Jobs, and Webhooks), understanding of distributed systems, consistency models", "RELATED_LESSON_CODES": "M00-L028, M00-L030", "SPECIFIC_FOCUS": "Consistency, Resiliency & Failure Semantics"},
    {"LESSON_CODE": "M00-L030", "LESSON_TITLE": "Backpressure: Queue Depth, Shed Load, and Brownouts", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L029 (Retry Policies: Exponential Backoff, Jitter, and Hedging), understanding of distributed systems, consistency models", "RELATED_LESSON_CODES": "M00-L029, M00-L031", "SPECIFIC_FOCUS": "Consistency, Resiliency & Failure Semantics"},
    {"LESSON_CODE": "M00-L031", "LESSON_TITLE": "Dead-Letter Queues: Triage, Replay, and Data Repair", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L030 (Backpressure: Queue Depth, Shed Load, and Brownouts), understanding of distributed systems, consistency models", "RELATED_LESSON_CODES": "M00-L030, M00-L032", "SPECIFIC_FOCUS": "Consistency, Resiliency & Failure Semantics"},
    {"LESSON_CODE": "M00-L032", "LESSON_TITLE": "Exactly-Once Delivery Myths & Achievable Guarantees", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L031 (Dead-Letter Queues: Triage, Replay, and Data Repair), understanding of distributed systems, consistency models", "RELATED_LESSON_CODES": "M00-L031, M00-L033", "SPECIFIC_FOCUS": "Consistency, Resiliency & Failure Semantics"},
    {"LESSON_CODE": "M00-L033", "LESSON_TITLE": "Failure Injection & Chaos Experiments for Distributed Systems", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L032 (Exactly-Once Delivery Myths & Achievable Guarantees), understanding of distributed systems, consistency models", "RELATED_LESSON_CODES": "M00-L032, M00-L034", "SPECIFIC_FOCUS": "Consistency, Resiliency & Failure Semantics"},
    {"LESSON_CODE": "M00-L034", "LESSON_TITLE": "HTTP Caching Deep Dive: ETag, Last-Modified, Cache-Control", "COMPLEXITY": "F", "TIME": "45", "LIST_PREREQUISITES": "M00-L033 (Failure Injection & Chaos Experiments for Distributed Systems), understanding of HTTP protocols, caching strategies", "RELATED_LESSON_CODES": "M00-L033, M00-L035", "SPECIFIC_FOCUS": "Caching Strategy (HTTP, CDN, App, DB)"},
    {"LESSON_CODE": "M00-L035", "LESSON_TITLE": "CDN Patterns: Cache Keys, Purge, and Signed URLs", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L034 (HTTP Caching Deep Dive: ETag, Last-Modified, Cache-Control), understanding of HTTP protocols, caching strategies", "RELATED_LESSON_CODES": "M00-L034, M00-L036", "SPECIFIC_FOCUS": "Caching Strategy (HTTP, CDN, App, DB)"},
    {"LESSON_CODE": "M00-L036", "LESSON_TITLE": "Application-Level Caching: In-Memory vs Distributed", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L035 (CDN Patterns: Cache Keys, Purge, and Signed URLs), understanding of HTTP protocols, caching strategies", "RELATED_LESSON_CODES": "M00-L035, M00-L037", "SPECIFIC_FOCUS": "Caching Strategy (HTTP, CDN, App, DB)"},
    {"LESSON_CODE": "M00-L037", "LESSON_TITLE": "Database Caching & Materialized Views", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L036 (Application-Level Caching: In-Memory vs Distributed), understanding of HTTP protocols, caching strategies", "RELATED_LESSON_CODES": "M00-L036, M00-L038", "SPECIFIC_FOCUS": "Caching Strategy (HTTP, CDN, App, DB)"},
    {"LESSON_CODE": "M00-L038", "LESSON_TITLE": "Cache Invalidation & Stale-While-Revalidate at Scale", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L037 (Database Caching & Materialized Views), understanding of HTTP protocols, caching strategies", "RELATED_LESSON_CODES": "M00-L037, M00-L039", "SPECIFIC_FOCUS": "Caching Strategy (HTTP, CDN, App, DB)"},
    {"LESSON_CODE": "M00-L039", "LESSON_TITLE": "Multi-Layer Caching: Ordering, Coherency, and TTLs", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L038 (Cache Invalidation & Stale-While-Revalidate at Scale), understanding of HTTP protocols, caching strategies", "RELATED_LESSON_CODES": "M00-L038, M00-L040", "SPECIFIC_FOCUS": "Caching Strategy (HTTP, CDN, App, DB)"},
    {"LESSON_CODE": "M00-L040", "LESSON_TITLE": "Observability for Caches: Hit Ratios & Tail Latency", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L039 (Multi-Layer Caching: Ordering, Coherency, and TTLs), understanding of HTTP protocols, caching strategies", "RELATED_LESSON_CODES": "M00-L039, M00-L041", "SPECIFIC_FOCUS": "Caching Strategy (HTTP, CDN, App, DB)"},
    {"LESSON_CODE": "M00-L041", "LESSON_TITLE": "Cost Modeling: CDN & Cache Economics", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L040 (Observability for Caches: Hit Ratios & Tail Latency), understanding of HTTP protocols, caching strategies", "RELATED_LESSON_CODES": "M00-L040, M00-L042", "SPECIFIC_FOCUS": "Caching Strategy (HTTP, CDN, App, DB)"},
    {"LESSON_CODE": "M00-L042", "LESSON_TITLE": "Algorithms: Token Bucket vs Leaky Bucket", "COMPLEXITY": "F", "TIME": "45", "LIST_PREREQUISITES": "M00-L041 (Cost Modeling: CDN & Cache Economics), understanding of rate limiting algorithms, traffic management", "RELATED_LESSON_CODES": "M00-L041, M00-L043", "SPECIFIC_FOCUS": "Rate Limiting & Fairness"},
    {"LESSON_CODE": "M00-L043", "LESSON_TITLE": "Global vs Tenant-Scoped Limits (Per-Key & Per-Org)", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L042 (Algorithms: Token Bucket vs Leaky Bucket), understanding of rate limiting algorithms, traffic management", "RELATED_LESSON_CODES": "M00-L042, M00-L044", "SPECIFIC_FOCUS": "Rate Limiting & Fairness"},
    {"LESSON_CODE": "M00-L044", "LESSON_TITLE": "Fairness & Abuse Prevention: Priority Queues and Quotas", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L043 (Global vs Tenant-Scoped Limits (Per-Key & Per-Org)), understanding of rate limiting algorithms, traffic management", "RELATED_LESSON_CODES": "M00-L043, M00-L045", "SPECIFIC_FOCUS": "Rate Limiting & Fairness"},
    {"LESSON_CODE": "M00-L045", "LESSON_TITLE": "Adaptive Limits from Live Signals (Autoscaling & SLOs)", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L044 (Fairness & Abuse Prevention: Priority Queues and Quotas), understanding of rate limiting algorithms, traffic management", "RELATED_LESSON_CODES": "M00-L044, M00-L046", "SPECIFIC_FOCUS": "Rate Limiting & Fairness"},
    {"LESSON_CODE": "M00-L046", "LESSON_TITLE": "Throttling for Webhooks & Partner APIs", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L045 (Adaptive Limits from Live Signals (Autoscaling & SLOs)), understanding of rate limiting algorithms, traffic management", "RELATED_LESSON_CODES": "M00-L045, M00-L047", "SPECIFIC_FOCUS": "Rate Limiting & Fairness"},
    {"LESSON_CODE": "M00-L047", "LESSON_TITLE": "Circuit Breakers vs Rate Limits: Complementary Roles", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L046 (Throttling for Webhooks & Partner APIs), understanding of rate limiting algorithms, traffic management", "RELATED_LESSON_CODES": "M00-L046, M00-L048", "SPECIFIC_FOCUS": "Rate Limiting & Fairness"},
    {"LESSON_CODE": "M00-L048", "LESSON_TITLE": "Legal/Compliance Considerations for Traffic Shaping", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L047 (Circuit Breakers vs Rate Limits: Complementary Roles), understanding of rate limiting algorithms, traffic management", "RELATED_LESSON_CODES": "M00-L047, M00-L049", "SPECIFIC_FOCUS": "Rate Limiting & Fairness"},
    {"LESSON_CODE": "M00-L049", "LESSON_TITLE": "Style Guides: Naming, Errors, and Pagination Contracts", "COMPLEXITY": "F", "TIME": "45", "LIST_PREREQUISITES": "M00-L048 (Legal/Compliance Considerations for Traffic Shaping), understanding of API design, REST principles", "RELATED_LESSON_CODES": "M00-L048, M00-L050", "SPECIFIC_FOCUS": "API Lifecycle: Standards, Versioning & Governance"},
    {"LESSON_CODE": "M00-L050", "LESSON_TITLE": "Versioning Tactics: URI, Headers, and Content Negotiation", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L049 (Style Guides: Naming, Errors, and Pagination Contracts), understanding of API design, REST principles", "RELATED_LESSON_CODES": "M00-L049, M00-L051", "SPECIFIC_FOCUS": "API Lifecycle: Standards, Versioning & Governance"},
    {"LESSON_CODE": "M00-L051", "LESSON_TITLE": "Deprecation Policy: Timelines, Comms, and Enforcement", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L050 (Versioning Tactics: URI, Headers, and Content Negotiation), understanding of API design, REST principles", "RELATED_LESSON_CODES": "M00-L050, M00-L052", "SPECIFIC_FOCUS": "API Lifecycle: Standards, Versioning & Governance"},
    {"LESSON_CODE": "M00-L052", "LESSON_TITLE": "Backward Compatibility & the Tolerant Reader Pattern", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L051 (Deprecation Policy: Timelines, Comms, and Enforcement), understanding of API design, REST principles", "RELATED_LESSON_CODES": "M00-L051, M00-L053", "SPECIFIC_FOCUS": "API Lifecycle: Standards, Versioning & Governance"},
    {"LESSON_CODE": "M00-L053", "LESSON_TITLE": "SDK Generation & Contract Testing (OpenAPI/Pact)", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L052 (Backward Compatibility & the Tolerant Reader Pattern), understanding of API design, REST principles", "RELATED_LESSON_CODES": "M00-L052, M00-L054", "SPECIFIC_FOCUS": "API Lifecycle: Standards, Versioning & Governance"},
    {"LESSON_CODE": "M00-L054", "LESSON_TITLE": "API Gateway Governance: Auth, Rate, and Observability", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L053 (SDK Generation & Contract Testing (OpenAPI/Pact)), understanding of API design, REST principles", "RELATED_LESSON_CODES": "M00-L053, M00-L055", "SPECIFIC_FOCUS": "API Lifecycle: Standards, Versioning & Governance"},
    {"LESSON_CODE": "M00-L055", "LESSON_TITLE": "Sunset, Data Export, and Customer Retention Paths", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L054 (API Gateway Governance: Auth, Rate, and Observability), understanding of API design, REST principles", "RELATED_LESSON_CODES": "M00-L054, M00-L056", "SPECIFIC_FOCUS": "API Lifecycle: Standards, Versioning & Governance"},
    {"LESSON_CODE": "M00-L056", "LESSON_TITLE": "API Monetization & Entitlement Mapping", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L055 (Sunset, Data Export, and Customer Retention Paths), understanding of API design, REST principles", "RELATED_LESSON_CODES": "M00-L055, M00-L057", "SPECIFIC_FOCUS": "API Lifecycle: Standards, Versioning & Governance"},
    {"LESSON_CODE": "M00-L057", "LESSON_TITLE": "Defining SLIs for Web, API, Jobs, and Data Pipelines", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L056 (API Monetization & Entitlement Mapping), understanding of reliability engineering, SLOs", "RELATED_LESSON_CODES": "M00-L056, M00-L058", "SPECIFIC_FOCUS": "Cost & Reliability Objectives (SLI/SLO/SLA, Multi-Region)"},
    {"LESSON_CODE": "M00-L058", "LESSON_TITLE": "Setting SLOs & Error Budgets: Trade-off Frameworks", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L057 (Defining SLIs for Web, API, Jobs, and Data Pipelines), understanding of reliability engineering, SLOs", "RELATED_LESSON_CODES": "M00-L057, M00-L059", "SPECIFIC_FOCUS": "Cost & Reliability Objectives (SLI/SLO/SLA, Multi-Region)"},
    {"LESSON_CODE": "M00-L059", "LESSON_TITLE": "Alert Philosophy: Symptom-Based, Multi-Signal, Low-Noise", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L058 (Setting SLOs & Error Budgets: Trade-off Frameworks), understanding of reliability engineering, SLOs", "RELATED_LESSON_CODES": "M00-L058, M00-L060", "SPECIFIC_FOCUS": "Cost & Reliability Objectives (SLI/SLO/SLA, Multi-Region)"},
    {"LESSON_CODE": "M00-L060", "LESSON_TITLE": "Release Safety: Blue/Green, Canary, Shadow Traffic", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L059 (Alert Philosophy: Symptom-Based, Multi-Signal, Low-Noise), understanding of reliability engineering, SLOs", "RELATED_LESSON_CODES": "M00-L059, M00-L061", "SPECIFIC_FOCUS": "Cost & Reliability Objectives (SLI/SLO/SLA, Multi-Region)"},
    {"LESSON_CODE": "M00-L061", "LESSON_TITLE": "Disaster Recovery: RTO/RPO, Backups, & Game Days", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L060 (Release Safety: Blue/Green, Canary, Shadow Traffic), understanding of reliability engineering, SLOs", "RELATED_LESSON_CODES": "M00-L060, M00-L062", "SPECIFIC_FOCUS": "Cost & Reliability Objectives (SLI/SLO/SLA, Multi-Region)"},
    {"LESSON_CODE": "M00-L062", "LESSON_TITLE": "Multi-Region Topologies & Data Residency Constraints", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L061 (Disaster Recovery: RTO/RPO, Backups, & Game Days), understanding of reliability engineering, SLOs", "RELATED_LESSON_CODES": "M00-L061, M00-L063", "SPECIFIC_FOCUS": "Cost & Reliability Objectives (SLI/SLO/SLA, Multi-Region)"},
    {"LESSON_CODE": "M00-L063", "LESSON_TITLE": "Unit Economics: Showback/Chargeback & Cost Budgets", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L062 (Multi-Region Topologies & Data Residency Constraints), understanding of reliability engineering, SLOs", "RELATED_LESSON_CODES": "M00-L062, M00-L064", "SPECIFIC_FOCUS": "Cost & Reliability Objectives (SLI/SLO/SLA, Multi-Region)"},
    {"LESSON_CODE": "M00-L064", "LESSON_TITLE": "Cost vs Reliability Experiments (When to Pay for 9s)", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L063 (Unit Economics: Showback/Chargeback & Cost Budgets), understanding of reliability engineering, SLOs", "RELATED_LESSON_CODES": "M00-L063, M00-L065", "SPECIFIC_FOCUS": "Cost & Reliability Objectives (SLI/SLO/SLA, Multi-Region)"},
    {"LESSON_CODE": "M00-L065", "LESSON_TITLE": "Golden Paths & Paved Roads: Starter Kits that Scale", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L064 (Cost vs Reliability Experiments (When to Pay for 9s)), understanding of platform engineering, organizational patterns", "RELATED_LESSON_CODES": "M00-L064, M00-L066", "SPECIFIC_FOCUS": "Portfolio / Platform Thinking"},
    {"LESSON_CODE": "M00-L066", "LESSON_TITLE": "Reusable Modules: Auth, Billing, Observability as Products", "COMPLEXITY": "I", "TIME": "60", "LIST_PREREQUISITES": "M00-L065 (Golden Paths & Paved Roads: Starter Kits that Scale), understanding of platform engineering, organizational patterns", "RELATED_LESSON_CODES": "M00-L065, M00-L067", "SPECIFIC_FOCUS": "Portfolio / Platform Thinking"},
    {"LESSON_CODE": "M00-L067", "LESSON_TITLE": "Platform SLAs and Internal Product Management", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L066 (Reusable Modules: Auth, Billing, Observability as Products), understanding of platform engineering, organizational patterns", "RELATED_LESSON_CODES": "M00-L066, M00-L068", "SPECIFIC_FOCUS": "Portfolio / Platform Thinking"},
    {"LESSON_CODE": "M00-L068", "LESSON_TITLE": "Standards & Guardrails: Tech Radar, ADR Catalogs", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L067 (Platform SLAs and Internal Product Management), understanding of platform engineering, organizational patterns", "RELATED_LESSON_CODES": "M00-L067, M00-L069", "SPECIFIC_FOCUS": "Portfolio / Platform Thinking"},
    {"LESSON_CODE": "M00-L069", "LESSON_TITLE": "Balancing Standardization vs Team Autonomy", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L068 (Standards & Guardrails: Tech Radar, ADR Catalogs), understanding of platform engineering, organizational patterns", "RELATED_LESSON_CODES": "M00-L068, M00-L070", "SPECIFIC_FOCUS": "Portfolio / Platform Thinking"},
    {"LESSON_CODE": "M00-L070", "LESSON_TITLE": "Measuring Platform ROI & Developer Experience", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L069 (Balancing Standardization vs Team Autonomy), understanding of platform engineering, organizational patterns", "RELATED_LESSON_CODES": "M00-L069, M00-L071", "SPECIFIC_FOCUS": "Portfolio / Platform Thinking"},
    {"LESSON_CODE": "M00-L071", "LESSON_TITLE": "Migration Playbooks for Org-Wide Change", "COMPLEXITY": "A", "TIME": "75", "LIST_PREREQUISITES": "M00-L070 (Measuring Platform ROI & Developer Experience), understanding of platform engineering, organizational patterns", "RELATED_LESSON_CODES": "M00-L070, M00-L072", "SPECIFIC_FOCUS": "Portfolio / Platform Thinking"},
    {"LESSON_CODE": "M00-L072", "LESSON_TITLE": "Org Design: Platform/Enablement/Stream-Aligned Teams", "COMPLEXITY": "E", "TIME": "90", "LIST_PREREQUISITES": "M00-L071 (Migration Playbooks for Org-Wide Change), understanding of platform engineering, organizational patterns", "RELATED_LESSON_CODES": "M00-L071", "SPECIFIC_FOCUS": "Portfolio / Platform Thinking"},
]

# Filter out already generated lessons
import os
import re

lesson_dir = "src/data/saas/lessons"
generated = set()
if os.path.exists(lesson_dir):
    files = [f for f in os.listdir(lesson_dir) if f.endswith('.md')]
    for f in files:
        match = re.match(r'(M[1-4]-L\d{3})', f)
        if match:
            generated.add(match.group(1))

print(f"Already generated: {len(generated)} M1-M4 lessons")
remaining_lessons = [l for l in m1_m4_lessons if l['LESSON_CODE'] not in generated]
print(f"Remaining to generate: {len(remaining_lessons)} lessons")

if remaining_lessons:
    counts = {}
    for lesson in remaining_lessons:
        mod = lesson['MODULE_CODE']
        counts[mod] = counts.get(mod, 0) + 1
    print("Breakdown by module:")
    for mod in ['M1', 'M2', 'M3', 'M4']:
        print(f"  {mod}: {counts.get(mod, 0)} lessons")

# Use remaining lessons
missing_lessons = remaining_lessons

# Module metadata will be extracted from lesson_details for each lesson
# Default module metadata (will be overridden per lesson)
DEFAULT_AUDIENCE = "Software engineers building SaaS platforms"
DEFAULT_FIRM_TYPE = "Technology companies and engineering teams"
DEFAULT_INDUSTRY = "SaaS platform development"

# Module-specific metadata
MODULE_METADATA = {
    "M1": {
        "MODULE_NAME": "Frontend Foundations (TypeScript/React/Next.js)",
        "AUDIENCE_DESCRIPTION": "Frontend engineers and developers building modern web applications",
        "INDUSTRY_DOMAIN": "Frontend development, React, Next.js, TypeScript"
    },
    "M2": {
        "MODULE_NAME": "Backend & API Development",
        "AUDIENCE_DESCRIPTION": "Backend engineers and API developers building scalable services",
        "INDUSTRY_DOMAIN": "Backend development, API design, Node.js, microservices"
    },
    "M3": {
        "MODULE_NAME": "Data, Storage & ORM",
        "AUDIENCE_DESCRIPTION": "Database engineers and developers working with data persistence",
        "INDUSTRY_DOMAIN": "Database design, SQL, ORM, data modeling, PostgreSQL"
    },
    "M4": {
        "MODULE_NAME": "Search, Retrieval & Recommendations",
        "AUDIENCE_DESCRIPTION": "Search engineers and developers building search and recommendation systems",
        "INDUSTRY_DOMAIN": "Search engines, information retrieval, vector search, recommendations"
    }
}

async def generate_lesson(lesson_details, lesson_num, total_in_batch):
    # Reload prompt template fresh for each lesson to ensure we always use the latest version
    with open("LESSON_GENERATION_PROMPT_GENERIC.md", "r", encoding="utf-8") as f:
        current_prompt_template = f.read()
    
    # Prepare variables for the prompt template (LESSON_GENERATION_PROMPT_GENERIC.md is a Python f-string)
    complexity_label = {'F': 'Foundation', 'I': 'Intermediate', 'A': 'Advanced', 'E': 'Expert'}.get(lesson_details['COMPLEXITY'], 'Foundation')
    module_code = lesson_details.get('MODULE_CODE', 'M0')
    module_name = lesson_details.get('MODULE_NAME', MODULE_METADATA.get(module_code, {}).get('MODULE_NAME', 'Unknown Module'))
    audience_desc = MODULE_METADATA.get(module_code, {}).get('AUDIENCE_DESCRIPTION', DEFAULT_AUDIENCE)
    industry_domain = MODULE_METADATA.get(module_code, {}).get('INDUSTRY_DOMAIN', DEFAULT_INDUSTRY)
    
    # Evaluate the prompt template as an f-string
    # The template file contains: prompt = f"""..."""
    # We need to extract the f-string content and evaluate it
    template_content = current_prompt_template.strip()
    if template_content.startswith('prompt = f"""'):
        template_content = template_content[13:]  # Remove 'prompt = f"""'
    if template_content.endswith('"""'):
        template_content = template_content[:-3]  # Remove trailing '"""'
    template_content = template_content.strip()
    
    # Now evaluate it as an f-string by creating a namespace with all needed variables
    # Replace all the f-string expressions manually for safety
    formatted_prompt = template_content
    formatted_prompt = formatted_prompt.replace("{lesson_details['LESSON_CODE']}", lesson_details['LESSON_CODE'])
    formatted_prompt = formatted_prompt.replace("{lesson_details['LESSON_TITLE']}", lesson_details['LESSON_TITLE'])
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('MODULE_CODE', 'M0')}", module_code)
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('MODULE_NAME', MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('MODULE_NAME', 'Unknown Module'))}", module_name)
    formatted_prompt = formatted_prompt.replace("{lesson_details.get('SPECIFIC_FOCUS', 'General')}", lesson_details.get('SPECIFIC_FOCUS', 'General'))
    formatted_prompt = formatted_prompt.replace("{'Foundation' if lesson_details['COMPLEXITY'] == 'F' else 'Intermediate' if lesson_details['COMPLEXITY'] == 'I' else 'Advanced' if lesson_details['COMPLEXITY'] == 'A' else 'Expert'}", complexity_label)
    formatted_prompt = formatted_prompt.replace("{lesson_details['COMPLEXITY']}", lesson_details['COMPLEXITY'])
    formatted_prompt = formatted_prompt.replace("{lesson_details['TIME']}", str(lesson_details['TIME']))
    formatted_prompt = formatted_prompt.replace("{MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('AUDIENCE_DESCRIPTION', DEFAULT_AUDIENCE)}", audience_desc)
    formatted_prompt = formatted_prompt.replace("{DEFAULT_FIRM_TYPE}", DEFAULT_FIRM_TYPE)
    formatted_prompt = formatted_prompt.replace("{lesson_details['LIST_PREREQUISITES']}", lesson_details['LIST_PREREQUISITES'])
    formatted_prompt = formatted_prompt.replace("{lesson_details['RELATED_LESSON_CODES']}", lesson_details['RELATED_LESSON_CODES'])
    formatted_prompt = formatted_prompt.replace("{MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('INDUSTRY_DOMAIN', DEFAULT_INDUSTRY)}", industry_domain)
    
    # Reload content structure fresh for each lesson (though this changes less frequently)
    with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
        current_content_structure = f.read()
    
    # Add the content structure reference (limited to avoid token limits)
    content_structure_preview = current_content_structure[:8000] if len(current_content_structure) > 8000 else current_content_structure
    truncation_note = '\n\n[Content structure truncated for token limits. Focus on generating the lesson based on the template structure.]' if len(current_content_structure) > 8000 else ''
    
    # Construct the final prompt
    prompt = f"""You are generating a polished, publication-ready lesson for an AI-Native SaaS Curriculum.
Read and internalize these two inputs before writing anything:

1. LESSON_GENERATION_PROMPT_GENERIC.md — this defines the canonical 10-section structure.
2. content_structure_ai-native-saas-curriculum-lesson-maps.md — this defines where the lesson fits in the broader curriculum.

### CONTENT STRUCTURE REFERENCE (Curriculum Map - first 8000 chars for context):
{content_structure_preview}{truncation_note}

---

{formatted_prompt}"""

    print(f"\n[Batch {lesson_num}/{total_in_batch}] Starting: {lesson_details['LESSON_CODE']} - {lesson_details['LESSON_TITLE']}")
    print(f"  Complexity: {lesson_details['COMPLEXITY']} | Duration: {lesson_details['TIME']} minutes")
    
    # Use gpt-5-mini as primary (it uses max_completion_tokens, not max_tokens)
    # gpt-5-mini doesn't support temperature (only default value of 1)
    model_to_use = MODEL_NAME
    print(f"  Using model: {model_to_use}")
    
    # gpt-5-mini uses max_completion_tokens instead of max_tokens
    api_params = {
        "model": model_to_use,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    # gpt-5-mini requires max_completion_tokens (not max_tokens) and doesn't support temperature
    if model_to_use == "gpt-5-mini":
        api_params["max_completion_tokens"] = 16384
        # gpt-5-mini only supports default temperature (1), don't set it
    # gpt-5-nano: no max_tokens or max_completion_tokens, temperature defaults to 1
    
    try:
        # OpenAI async streaming
        stream = await client.chat.completions.create(**api_params)
        
        lesson_content_parts = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                lesson_content_parts.append(content)
        lesson_content = "".join(lesson_content_parts)
        
    except Exception as e:
        print(f"  [WARNING] Streaming failed for {lesson_details['LESSON_CODE']}, trying non-streaming: {e}")
        # Non-streaming attempt
        non_stream_params = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": prompt}]
        }
        if model_to_use == "gpt-5-mini":
            non_stream_params["max_completion_tokens"] = 16384  # gpt-5-mini uses max_completion_tokens, not max_tokens
            # gpt-5-mini only supports default temperature (1), don't set it
        
        response = await client.chat.completions.create(**non_stream_params)
        lesson_content = response.choices[0].message.content

    date_str = datetime.now().strftime("%Y-%m-%d")
    # Clean filename: remove/replace all special characters
    import re
    filename_slug = lesson_details['LESSON_TITLE'].lower()
    # Replace Unicode arrows and special chars
    filename_slug = filename_slug.replace('→', 'to').replace('→', 'to')
    filename_slug = filename_slug.replace('(', '').replace(')', '').replace('/', '-')
    filename_slug = filename_slug.replace(':', '').replace(',', '').replace('&', 'and')
    # Replace multiple spaces/hyphens with single hyphen
    filename_slug = re.sub(r'[-\s]+', '-', filename_slug)
    # Remove any remaining non-alphanumeric except hyphens
    filename_slug = re.sub(r'[^a-z0-9\-]', '', filename_slug)
    filename = f"{lesson_details['LESSON_CODE']}-{filename_slug}--{date_str}.md"

    output_path = f"src/data/saas/lessons/{filename}"
    os.makedirs("src/data/saas/lessons", exist_ok=True)

    # Post-process to remove any prompt artifacts that might have been included
    cleaned_content = lesson_content
    
    # Remove LLM prompt headers if present
    cleaned_content = re.sub(r'^# LLM Prompt:.*?\n', '', cleaned_content, flags=re.MULTILINE)
    
    # Remove "Context & Parameters" sections
    cleaned_content = re.sub(r'^## \*\*Context & Parameters\*\*.*?\n(?=##|$)', '', cleaned_content, flags=re.MULTILINE | re.DOTALL)
    
    # Remove "Content Output Schema" sections  
    cleaned_content = re.sub(r'^## \*\*Content Output Schema.*?\n(?=##|#)', '', cleaned_content, flags=re.MULTILINE | re.DOTALL)
    
    # Remove "Quality and governance notes" at the end
    cleaned_content = re.sub(r'\n---\s*\nQuality and governance notes.*$', '', cleaned_content, flags=re.DOTALL)
    
    # Ensure content starts with lesson title or Section 1
    if not re.match(r'^(# Lesson |# Section 1:|## Section 1:|Section 1:)', cleaned_content):
        # Try to find where actual lesson starts
        match = re.search(r'(^# Lesson .*|^# Section 1:.*|^## Section 1:.*|^Section 1:)', cleaned_content, re.MULTILINE)
        if match:
            cleaned_content = cleaned_content[match.start():]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)

    lines = len(lesson_content.split('\n'))
    chars = len(lesson_content)
    
    print(f"  [SUCCESS] {lesson_details['LESSON_CODE']}: {lines} lines, {chars:,} chars -> {filename}")
    
    return output_path

async def main():
    print("="*80)
    print("GENERATING LESSONS IN PARALLEL BATCHES")
    print(f"Total lessons to generate: {len(missing_lessons)}")
    print("Batch size: 3 lessons in parallel")
    print("Delay between batches: 10 seconds")
    print("="*80)
    
    generated_files = []
    batch_size = 3
    
    # Process lessons in batches of 3
    for batch_start in range(0, len(missing_lessons), batch_size):
        batch_num = (batch_start // batch_size) + 1
        total_batches = (len(missing_lessons) + batch_size - 1) // batch_size
        batch_lessons = missing_lessons[batch_start:batch_start + batch_size]
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_num}/{total_batches}: Generating {len(batch_lessons)} lessons in parallel")
        print("="*80)
        
        # Create tasks for parallel execution
        tasks = []
        for idx, lesson in enumerate(batch_lessons, 1):
            task = generate_lesson(lesson, idx, len(batch_lessons))
            tasks.append(task)
        
        # Execute all tasks in parallel
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    lesson_code = batch_lessons[i]['LESSON_CODE']
                    print(f"  [ERROR] Failed to generate {lesson_code}: {result}")
                else:
                    generated_files.append(result)
            
        except Exception as e:
            print(f"[ERROR] Batch {batch_num} failed: {e}")
        
        # Wait 10 seconds before next batch (except after the last batch)
        if batch_start + batch_size < len(missing_lessons):
            print(f"\nWaiting 10 seconds before next batch...")
            await asyncio.sleep(10)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Successfully generated {len(generated_files)} lessons:")
    for f in generated_files:
        print(f"  - {f}")

if __name__ == "__main__":
    asyncio.run(main())

