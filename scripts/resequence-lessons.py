#!/usr/bin/env python3
"""
AI Lesson Resequencing Script
Reorders lesson files according to the Content Structure.md pedagogical flow
"""

import os
import shutil
from pathlib import Path

# Base path for lessons
LESSONS_PATH = Path("src/data/ai/lessons")
BACKUP_PATH = Path("backup-lessons")

# Module 1 Resequencing Map (Current -> Correct)
MODULE_1_MAPPING = {
    # AI Literacy & Mindset - Understanding Generative AI Architecture (1-10)
    "M01-L028-generative-ai-fundamentals.md": "M01-L001-generative-ai-fundamentals.md",
    "M01-L062-transformer-model-architecture.md": "M01-L002-transformer-model-architecture.md",
    "M01-L061-tokenization-input-structure.md": "M01-L003-tokenization-input-structure.md",
    "M01-L015-context-window-memory-architecture.md": "M01-L004-context-window-memory-architecture.md",
    "M01-L039-model-lifecycle-training-paradigms.md": "M01-L005-model-lifecycle-training-paradigms.md",
    "M01-L035-inference-runtime-dynamics.md": "M01-L006-inference-runtime-dynamics.md",
    "M01-L030-gpu-infrastructure-computational-requirements.md": "M01-L007-gpu-infrastructure-computational-requirements.md",
    "M01-L038-model-efficiency-compression-techniques.md": "M01-L008-model-efficiency-compression-techniques.md",
    "M01-L006-attention-mechanisms-self-attention.md": "M01-L009-attention-mechanisms-self-attention.md",
    "M01-L022-embedding-spaces-semantic-representation.md": "M01-L010-embedding-spaces-semantic-representation.md",
    
    # Prompt Engineering & Interaction Patterns (11-19)
    "M01-L050-prompt-engineering-principles.md": "M01-L011-prompt-engineering-principles.md",
    "M01-L066-zero-shot-few-shot-chain-of-thought-prompting.md": "M01-L012-zero-shot-few-shot-chain-of-thought-prompting.md",
    "M01-L049-prompt-design-patterns-templates.md": "M01-L013-prompt-design-patterns-templates.md",
    "M01-L014-context-engineering-instruction-clarity.md": "M01-L014-context-engineering-instruction-clarity.md",  # No change
    "M01-L037-iterative-refinement-strategies.md": "M01-L015-iterative-refinement-strategies.md",
    "M01-L043-multi-turn-conversation-design.md": "M01-L016-multi-turn-conversation-design.md",
    "M01-L051-prompt-security-injection-prevention.md": "M01-L017-prompt-security-injection-prevention.md",
    "M01-L060-temperature-top-p-sampling-parameters.md": "M01-L018-temperature-top-p-sampling-parameters.md",
    "M01-L059-system-messages-role-definition.md": "M01-L019-system-messages-role-definition.md",
    
    # Model Reliability & Quality Assurance (20-25)
    "M01-L032-hallucinations-reliability-engineering.md": "M01-L020-hallucinations-reliability-engineering.md",
    "M01-L040-model-output-validation-techniques.md": "M01-L021-model-output-validation-techniques.md",
    "M01-L031-grounding-strategies-fact-checking.md": "M01-L022-grounding-strategies-fact-checking.md",
    "M01-L013-confidence-scoring-uncertainty-quantification.md": "M01-L023-confidence-scoring-uncertainty-quantification.md",
    "M01-L007-bias-detection-mitigation.md": "M01-L024-bias-detection-mitigation.md",
    "M01-L001-adversarial-testing-methodologies.md": "M01-L025-adversarial-testing-methodologies.md",
    
    # Human-AI Collaboration (26-32)
    "M01-L034-human-ai-collaboration-patterns.md": "M01-L026-human-ai-collaboration-patterns.md",
    "M01-L003-ai-as-thought-partner-vs.-executor.md": "M01-L027-ai-as-thought-partner-vs.-executor.md",
    "M01-L021-effective-delegation-to-ai-systems.md": "M01-L028-effective-delegation-to-ai-systems.md",
    "M01-L058-skill-augmentation-vs.-replacement.md": "M01-L029-skill-augmentation-vs.-replacement.md",
    "M01-L012-cognitive-load-management-with-ai.md": "M01-L030-cognitive-load-management-with-ai.md",
    "M01-L024-ethical-considerations-failure-modes.md": "M01-L031-ethical-considerations-failure-modes.md",
    "M01-L056-responsible-ai-development-principles.md": "M01-L032-responsible-ai-development-principles.md",
    
    # AI Tools Hands-On - Core Platform Mastery (33-35)
    "M01-L011-claude-enterprise-for-analysis-ideation.md": "M01-L033-claude-enterprise-for-analysis-ideation.md",
    "M01-L016-cursor-pro-for-ai-assisted-development.md": "M01-L034-cursor-pro-for-ai-assisted-development.md",
    "M01-L010-claude-code-for-command-line-agency.md": "M01-L035-claude-code-for-command-line-agency.md",
    
    # Collaboration & Documentation Tools (36-38)
    "M01-L029-github-team-for-collaboration.md": "M01-L036-github-team-for-collaboration.md",
    "M01-L026-gamma.app-for-presentation-generation.md": "M01-L037-gamma.app-for-presentation-generation.md",
    "M01-L063-typora-markdown-for-documentation.md": "M01-L038-typora-markdown-for-documentation.md",
    
    # AI-Specific Tools & Utilities (39-43)
    "M01-L048-perplexity-for-research-discovery.md": "M01-L039-perplexity-for-research-discovery.md",
    "M01-L045-notebooklm-for-knowledge-synthesis.md": "M01-L040-notebooklm-for-knowledge-synthesis.md",
    "M01-L009-chatgpt-gpt-4-api-integration.md": "M01-L041-chatgpt-gpt-4-api-integration.md",
    "M01-L027-gemini-for-multi-modal-analysis.md": "M01-L042-gemini-for-multi-modal-analysis.md",
    "M01-L046-open-source-llm-deployment-ollama-lm-studio.md": "M01-L043-open-source-llm-deployment-ollama-lm-studio.md",
    
    # Security & Best Practices (44-45)
    "M01-L004-ai-usage-hygiene-security.md": "M01-L044-ai-usage-hygiene-security.md",
    "M01-L019-dynpro's-project-dna-ai-native-vision.md": "M01-L045-dynpro's-project-dna-ai-native-vision.md",
    
    # RAG Fundamentals (46-49)
    "M01-L052-rag-pattern-architecture.md": "M01-L046-rag-pattern-architecture.md",
    "M01-L064-vector-databases-for-semantic-search.md": "M01-L047-vector-databases-for-semantic-search.md",
    "M01-L018-document-processing-pipelines.md": "M01-L048-document-processing-pipelines.md",
    "M01-L057-retrieval-evaluation-optimization.md": "M01-L049-retrieval-evaluation-optimization.md",
    
    # Deployment Options - API-Hosted Models (50-53)
    "M01-L005-api-hosted-models.md": "M01-L050-api-hosted-models.md",
    "M01-L065-vendor-comparison-cost-optimization.md": "M01-L051-vendor-comparison-cost-optimization.md",
    "M01-L053-rate-limiting-request-management.md": "M01-L052-rate-limiting-request-management.md",
    "M01-L042-multi-provider-failover-architecture.md": "M01-L053-multi-provider-failover-architecture.md",
    
    # Self-Hosted & Open-Source (54-57)
    "M01-L047-open-source-llms.md": "M01-L054-open-source-llms.md",
    "M01-L041-model-selection-quantization.md": "M01-L055-model-selection-quantization.md",
    "M01-L036-inference-optimization.md": "M01-L056-inference-optimization.md",
    "M01-L033-hardware-requirements-scaling.md": "M01-L057-hardware-requirements-scaling.md",
    
    # Hybrid Deployment (58-60)
    "M01-L020-edge-vs.-cloud-decision-framework.md": "M01-L058-edge-vs.-cloud-decision-framework.md",
    "M01-L017-data-residency-compliance-requirements.md": "M01-L059-data-residency-compliance-requirements.md",
    "M01-L025-fallback-redundancy-patterns.md": "M01-L060-fallback-redundancy-patterns.md",
    
    # LLM Limitations - Current Constraints (61-63)
    "M01-L055-reasoning-constraints.md": "M01-L061-reasoning-constraints.md",
    "M01-L054-real-time-awareness-gaps.md": "M01-L062-real-time-awareness-gaps.md",
    "M01-L044-multimodal-capabilities.md": "M01-L063-multimodal-capabilities.md",
    
    # Advanced Paradigms (64-66)
    "M01-L002-agent-frameworks-orchestration.md": "M01-L064-agent-frameworks-orchestration.md",
    "M01-L023-emerging-capabilities.md": "M01-L065-emerging-capabilities.md",
    "M01-L008-build-vs.-buy-decision-framework.md": "M01-L066-build-vs.-buy-decision-framework.md",
}

# Module 2 Resequencing Map (Based on Content Structure.md)
MODULE_2_MAPPING = {
    # AI-Powered Planning & Analysis - Requirements & Design (1-4)
    "M02-L016-feature-decomposition-with-claude-cursor.md": "M02-L001-feature-decomposition-with-claude-cursor.md",
    "M02-L018-llm-assisted-user-story-mapping.md": "M02-L002-llm-assisted-user-story-mapping.md",
    "M02-L014-effort-estimation-with-ai-classification.md": "M02-L003-effort-estimation-with-ai-classification.md",
    "M02-L005-architecture-decision-records-adrs-with-ai.md": "M02-L004-architecture-decision-records-adrs-with-ai.md",
    
    # Technical Specification Generation (5-8)
    "M02-L004-api-design-openapi-specification.md": "M02-L005-api-design-openapi-specification.md",
    "M02-L012-database-schema-design-erd-generation.md": "M02-L006-database-schema-design-erd-generation.md",
    "M02-L029-system-architecture-diagram-creation.md": "M02-L007-system-architecture-diagram-creation.md",
    "M02-L021-non-functional-requirements-extraction.md": "M02-L008-non-functional-requirements-extraction.md",
    
    # AI-Augmented Coding & Testing - Development Acceleration (9-12)
    "M02-L011-context-aware-code-generation.md": "M02-L009-context-aware-code-generation.md",
    "M02-L003-ai-guided-test-case-generation.md": "M02-L010-ai-guided-test-case-generation.md",
    "M02-L002-ai-driven-debugging.md": "M02-L011-ai-driven-debugging.md",
    "M02-L026-refactoring-with-ai-pair-programming.md": "M02-L012-refactoring-with-ai-pair-programming.md",
    
    # Quality Assurance & Validation (13-15)
    "M02-L007-automated-test-generation.md": "M02-L013-automated-test-generation.md",
    "M02-L009-code-review-augmentation.md": "M02-L014-code-review-augmentation.md",
    "M02-L023-performance-profiling-optimization.md": "M02-L015-performance-profiling-optimization.md",
    
    # Documentation & Knowledge Capture - Code-Level Documentation (16-18)
    "M02-L006-auto-generated-api-documentation.md": "M02-L016-auto-generated-api-documentation.md",
    "M02-L017-intelligent-code-commenting.md": "M02-L017-intelligent-code-commenting.md",  # No change
    "M02-L025-readme-setup-documentation.md": "M02-L018-readme-setup-documentation.md",
    
    # Knowledge Transfer & Translation (19-21)
    "M02-L030-technical-translation-for-non-technical-audiences.md": "M02-L019-technical-translation-for-non-technical-audiences.md",
    "M02-L032-visual-summary-generation.md": "M02-L020-visual-summary-generation.md",
    "M02-L022-onboarding-documentation-automation.md": "M02-L021-onboarding-documentation-automation.md",
    
    # Living Documentation (22-25)
    "M02-L008-change-log-release-notes-generation.md": "M02-L022-change-log-release-notes-generation.md",
    "M02-L020-migration-guide-automation.md": "M02-L023-migration-guide-automation.md",
    "M02-L013-deprecation-notice-creation.md": "M02-L024-deprecation-notice-creation.md",
    "M02-L031-version-comparison-summaries.md": "M02-L025-version-comparison-summaries.md",
    
    # Collaboration Patterns - Meeting & Communication Efficiency (26-28)
    "M02-L019-meeting-communication-summarization.md": "M02-L026-meeting-communication-summarization.md",
    "M02-L027-requirements-to-prompts-translation.md": "M02-L027-requirements-to-prompts-translation.md",  # No change
    "M02-L001-ai-driven-action-item-extraction.md": "M02-L028-ai-driven-action-item-extraction.md",
    
    # Asynchronous Collaboration (29-32)
    "M02-L010-code-review-comment-generation.md": "M02-L029-code-review-comment-generation.md",
    "M02-L024-pull-request-description-templates.md": "M02-L030-pull-request-description-templates.md",
    "M02-L028-slack-teams-response-drafting.md": "M02-L031-slack-teams-response-drafting.md",
    "M02-L015-email-thread-summarization.md": "M02-L032-email-thread-summarization.md",
}

# Module 3 Resequencing Map (Based on Content Structure.md)
MODULE_3_MAPPING = {
    # Data Engineers & Analysts - Data Transformation & Orchestration (1-3)
    "M03-L003-ai-written-sql-transformations.md": "M03-L001-ai-written-sql-transformations.md",
    "M03-L044-snowflake-query-optimization.md": "M03-L002-snowflake-query-optimization.md",
    "M03-L015-dbt-model-generation-testing.md": "M03-L003-dbt-model-generation-testing.md",
    
    # Advanced Analytics & AI Integration (4-6)
    "M03-L030-natural-language-sql-agents.md": "M03-L004-natural-language-sql-agents.md",
    "M03-L036-rag-based-analytical-agents.md": "M03-L005-rag-based-analytical-agents.md",
    "M03-L035-python-data-pipeline-development.md": "M03-L006-python-data-pipeline-development.md",
    
    # Data Quality & Governance (7-10)
    "M03-L014-data-profiling-anomaly-detection.md": "M03-L007-data-profiling-anomaly-detection.md",
    "M03-L042-schema-evolution-management.md": "M03-L008-schema-evolution-management.md",
    "M03-L027-metadata-management-automation.md": "M03-L009-metadata-management-automation.md",
    "M03-L012-data-lineage-documentation.md": "M03-L010-data-lineage-documentation.md",
    
    # QA Engineers - Test Strategy & Planning (11-12)
    "M03-L039-requirements-based-test-generation.md": "M03-L011-requirements-based-test-generation.md",
    "M03-L048-test-plan-documentation.md": "M03-L012-test-plan-documentation.md",
    
    # Execution & Analysis (13-15)
    "M03-L026-log-analysis-root-cause-investigation.md": "M03-L013-log-analysis-root-cause-investigation.md",
    "M03-L021-hallucination-risk-testing.md": "M03-L014-hallucination-risk-testing.md",
    "M03-L033-performance-testing-scenarios.md": "M03-L015-performance-testing-scenarios.md",
    
    # Automation & CI/CD Integration (16-17)
    "M03-L007-ci-cd-test-integration.md": "M03-L016-ci-cd-test-integration.md",
    "M03-L047-test-data-management.md": "M03-L017-test-data-management.md",
    
    # AI-Specific QA Considerations (18-21)
    "M03-L025-llm-output-consistency-testing.md": "M03-L018-llm-output-consistency-testing.md",
    "M03-L034-prompt-regression-testing.md": "M03-L019-prompt-regression-testing.md",
    "M03-L029-model-drift-detection.md": "M03-L020-model-drift-detection.md",
    "M03-L017-ethical-ai-testing-frameworks.md": "M03-L021-ethical-ai-testing-frameworks.md",
    
    # Salesforce & SaaS Engineers - Development Acceleration (22-24)
    "M03-L004-apex-code-generation.md": "M03-L022-apex-code-generation.md",
    "M03-L024-lightning-web-component-development.md": "M03-L023-lightning-web-component-development.md",
    
    # CRM Intelligence & Automation (24-26)
    "M03-L010-crm-workflow-ai-integration.md": "M03-L024-crm-workflow-ai-integration.md",
    "M03-L016-declarative-ai-agents.md": "M03-L025-declarative-ai-agents.md",
    "M03-L020-genai-in-salesforce-reporting.md": "M03-L026-genai-in-salesforce-reporting.md",
    
    # Integration & Data Management (27-28)
    "M03-L005-api-integration-development.md": "M03-L027-api-integration-development.md",
    "M03-L013-data-migration-cleanup.md": "M03-L028-data-migration-cleanup.md",
    
    # Workato / Integration Specialists - Recipe Development & Optimization (29-32)
    "M03-L001-ai-generated-integration-recipes.md": "M03-L029-ai-generated-integration-recipes.md",
    "M03-L043-schema-mapping-automation.md": "M03-L030-schema-mapping-automation.md",
    "M03-L028-metadata-translation-validation.md": "M03-L031-metadata-translation-validation.md",
    
    # Enterprise Integration Patterns (32-35)
    "M03-L018-event-driven-architecture-design.md": "M03-L032-event-driven-architecture-design.md",
    "M03-L006-batch-processing-optimization.md": "M03-L033-batch-processing-optimization.md",
    "M03-L037-real-time-sync-configuration.md": "M03-L034-real-time-sync-configuration.md",
    "M03-L040-retry-circuit-breaker-patterns.md": "M03-L035-retry-circuit-breaker-patterns.md",
    
    # Monitoring & Troubleshooting (36-39)
    "M03-L038-recipe-performance-analysis.md": "M03-L036-recipe-performance-analysis.md",
    "M03-L009-connection-health-monitoring.md": "M03-L037-connection-health-monitoring.md",
    "M03-L023-job-failure-investigation.md": "M03-L038-job-failure-investigation.md",
    "M03-L022-integration-testing-automation.md": "M03-L039-integration-testing-automation.md",
    
    # PMs & Tech Leads - Project Planning & Execution (40-42)
    "M03-L002-ai-generated-project-artifacts.md": "M03-L040-ai-generated-project-artifacts.md",
    "M03-L046-task-decomposition-from-stories.md": "M03-L041-task-decomposition-from-stories.md",
    "M03-L045-sprint-ceremony-support.md": "M03-L042-sprint-ceremony-support.md",
    
    # Strategic Product Management (43-44)
    "M03-L032-outcome-based-pricing-agents.md": "M03-L043-outcome-based-pricing-agents.md",
    "M03-L008-competitive-analysis-automation.md": "M03-L044-competitive-analysis-automation.md",
    
    # Stakeholder Management (45-48)
    "M03-L019-executive-briefing-generation.md": "M03-L045-executive-briefing-generation.md",
    "M03-L011-customer-journey-mapping.md": "M03-L046-customer-journey-mapping.md",
    "M03-L031-okr-kpi-tracking-dashboards.md": "M03-L047-okr-kpi-tracking-dashboards.md",
    "M03-L041-roadmap-visualization.md": "M03-L048-roadmap-visualization.md",
}

# Module 4 Resequencing Map (Based on Content Structure.md)
MODULE_4_MAPPING = {
    # The Collaborative AI Workspace (Claude Teams) - Organizational Foundation (1-3)
    "M04-L030-organization-structure-governance.md": "M04-L001-organization-structure-governance.md",
    "M04-L033-projects-for-work-organization.md": "M04-L002-projects-for-work-organization.md",
    "M04-L023-knowledge-management.md": "M04-L003-knowledge-management.md",
    
    # Security & Compliance (4-7)
    "M04-L035-security-privacy.md": "M04-L004-security-privacy.md",
    "M04-L016-data-residency-encryption.md": "M04-L005-data-residency-encryption.md",
    "M04-L007-audit-trail-configuration.md": "M04-L006-audit-trail-configuration.md",
    "M04-L037-soc-2-compliance-alignment.md": "M04-L007-soc-2-compliance-alignment.md",
    
    # Advanced Collaboration Features (8-11)
    "M04-L036-shared-conversation-management.md": "M04-L008-shared-conversation-management.md",
    "M04-L039-team-prompt-libraries.md": "M04-L009-team-prompt-libraries.md",
    "M04-L015-custom-style-guides-instructions.md": "M04-L010-custom-style-guides-instructions.md",
    "M04-L022-integration-with-enterprise-systems.md": "M04-L011-integration-with-enterprise-systems.md",
    
    # The Command Line as Conversation (Claude Code) - Core Capabilities (12-13)
    "M04-L012-core-command-patterns.md": "M04-L012-core-command-patterns.md",  # No change
    "M04-L010-context-engineering.md": "M04-L013-context-engineering.md",
    
    # Agentic Workflows (14-15)
    "M04-L006-agentic-workflows.md": "M04-L014-agentic-workflows.md",
    "M04-L001-advanced-orchestration.md": "M04-L015-advanced-orchestration.md",
    
    # Practical Applications (16-19)
    "M04-L034-script-generation-execution.md": "M04-L016-script-generation-execution.md",
    "M04-L020-file-manipulation-batch-processing.md": "M04-L017-file-manipulation-batch-processing.md",
    "M04-L017-development-environment-setup.md": "M04-L018-development-environment-setup.md",
    "M04-L021-infrastructure-as-code-generation.md": "M04-L019-infrastructure-as-code-generation.md",
    
    # The AI-Native IDE (Cursor) - Core Development Features (20-21)
    "M04-L013-core-development-features.md": "M04-L020-core-development-features.md",
    "M04-L009-codebase-awareness.md": "M04-L021-codebase-awareness.md",
    
    # Advanced Workflows (22-25)
    "M04-L002-advanced-workflows.md": "M04-L022-advanced-workflows.md",
    "M04-L014-custom-rules-style-enforcement.md": "M04-L023-custom-rules-style-enforcement.md",
    "M04-L040-test-driven-development-support.md": "M04-L024-test-driven-development-support.md",
    
    # Configuration & Optimization (25-28)
    "M04-L026-model-selection-configuration.md": "M04-L025-model-selection-configuration.md",
    "M04-L011-context-window-management.md": "M04-L026-context-window-management.md",
    "M04-L031-performance-tuning.md": "M04-L027-performance-tuning.md",
    "M04-L019-extension-plugin-integration.md": "M04-L028-extension-plugin-integration.md",
    
    # Hybrid Workflow Integration - Tool Synergy Patterns (29-30)
    "M04-L008-claude-code-+-cursor-synergy.md": "M04-L029-claude-code-+-cursor-synergy.md",
    "M04-L024-macro-micro-task-distribution.md": "M04-L030-macro-micro-task-distribution.md",
    
    # Workflow Orchestration (31-34)
    "M04-L041-tool-selection-decision-trees.md": "M04-L031-tool-selection-decision-trees.md",
    "M04-L028-multi-tool-pipeline-design.md": "M04-L032-multi-tool-pipeline-design.md",
    "M04-L038-state-synchronization-strategies.md": "M04-L033-state-synchronization-strategies.md",
    "M04-L043-version-control-integration.md": "M04-L034-version-control-integration.md",
    
    # Agent Architecture & Orchestration - Foundational Agent Design (35-36)
    "M04-L003-agent-anatomy.md": "M04-L035-agent-anatomy.md",
    "M04-L004-agent-capabilities-spectrum.md": "M04-L036-agent-capabilities-spectrum.md",
    
    # Orchestration Patterns (37-38)
    "M04-L029-orchestrator-patterns.md": "M04-L037-orchestrator-patterns.md",
    "M04-L005-agent-development-lifecycle.md": "M04-L038-agent-development-lifecycle.md",
    
    # Enterprise Agent Platforms (39-41)
    "M04-L032-platform-patterns.md": "M04-L039-platform-patterns.md",
    "M04-L027-multi-agent-systems.md": "M04-L040-multi-agent-systems.md",
    
    # Advanced Agent Capabilities (41-43)
    "M04-L042-tool-use-function-calling.md": "M04-L041-tool-use-function-calling.md",
    "M04-L025-memory-state-management.md": "M04-L042-memory-state-management.md",
    "M04-L018-error-handling-resilience.md": "M04-L043-error-handling-resilience.md",
}

# Module 5 Resequencing Map (Based on Content Structure.md)
MODULE_5_MAPPING = {
    # AI Delivery Strategy - Outcome-Driven Design (1-2)
    "M05-L016-outcome-first-design.md": "M05-L001-outcome-first-design.md",
    "M05-L004-ai-roi-benchmarks.md": "M05-L002-ai-roi-benchmarks.md",
    
    # Implementation Frameworks (3-4)
    "M05-L017-project-dna-rollout-pattern.md": "M05-L003-project-dna-rollout-pattern.md",
    "M05-L015-outcome-based-pricing.md": "M05-L004-outcome-based-pricing.md",
    
    # Strategic Planning Tools (5-8)
    "M05-L003-ai-maturity-assessment.md": "M05-L005-ai-maturity-assessment.md",
    "M05-L008-capability-gap-analysis.md": "M05-L006-capability-gap-analysis.md",
    "M05-L023-technology-roadmap-development.md": "M05-L007-technology-roadmap-development.md",
    "M05-L007-build-vs.-buy-decision-framework.md": "M05-L008-build-vs.-buy-decision-framework.md",
    
    # Governance & Risk - Risk Management Frameworks (9-10)
    "M05-L014-llm-risk-classification.md": "M05-L009-llm-risk-classification.md",
    "M05-L021-responsible-ai-frameworks.md": "M05-L010-responsible-ai-frameworks.md",
    
    # Control Implementation (11-12)
    "M05-L013-human-in-the-loop-enforcement.md": "M05-L011-human-in-the-loop-enforcement.md",
    "M05-L005-ai-policy-development.md": "M05-L012-ai-policy-development.md",
    
    # Compliance & Monitoring (13-16)
    "M05-L020-regulatory-compliance-gdpr-ccpa-ai-act.md": "M05-L013-regulatory-compliance-gdpr-ccpa-ai-act.md",
    "M05-L018-model-performance-monitoring.md": "M05-L014-model-performance-monitoring.md",
    "M05-L006-bias-detection-remediation.md": "M05-L015-bias-detection-remediation.md",
    "M05-L012-incident-response-procedures.md": "M05-L016-incident-response-procedures.md",
    
    # Client Education & Adoption - Communication & Expectation Setting (17-18)
    "M05-L019-explaining-ai-to-clients.md": "M05-L017-explaining-ai-to-clients.md",
    "M05-L009-building-trust-in-ai-solutions.md": "M05-L018-building-trust-in-ai-solutions.md",
    
    # Collaborative Development (19-20)
    "M05-L010-co-creating-with-clients.md": "M05-L019-co-creating-with-clients.md",
    "M05-L015-managing-ai-maturity-curves.md": "M05-L020-managing-ai-maturity-curves.md",
    
    # Training & Enablement (21-23)
    "M05-L011-hands-on-training-programs.md": "M05-L021-hands-on-training-programs.md",
    "M05-L002-documentation-reference-materials.md": "M05-L022-documentation-reference-materials.md",
    "M05-L001-champion-network-development.md": "M05-L023-champion-network-development.md",
}

# Module 6 Resequencing Map (Based on Content Structure.md)
MODULE_6_MAPPING = {
    # Internal Learning Patterns - Structured Learning (1-3)
    "M06-L011-guild-based-study-groups.md": "M06-L001-guild-based-study-groups.md",
    "M06-L001-80-20-5-transformation-formula.md": "M06-L002-80-20-5-transformation-formula.md",
    "M06-L002-ai-hackathons.md": "M06-L003-ai-hackathons.md",
    
    # Knowledge Management (4-7)
    "M06-L008-documentation-blueprint-sharing.md": "M06-L004-documentation-blueprint-sharing.md",
    "M06-L017-pattern-library-development.md": "M06-L005-pattern-library-development.md",
    "M06-L022-solution-template-repository.md": "M06-L006-solution-template-repository.md",
    "M06-L006-best-practice-documentation.md": "M06-L007-best-practice-documentation.md",
    
    # Mentorship & Coaching (8-11)
    "M06-L003-ai-champion-program.md": "M06-L008-ai-champion-program.md",
    "M06-L018-peer-code-review-circles.md": "M06-L009-peer-code-review-circles.md",
    "M06-L015-office-hours-support-networks.md": "M06-L010-office-hours-support-networks.md",
    "M06-L007-career-development-pathways.md": "M06-L011-career-development-pathways.md",
    
    # Innovation Tools & Habits - Capability Tracking (12-13)
    "M06-L023-tracking-llm-capability-evolution.md": "M06-L012-tracking-llm-capability-evolution.md",
    "M06-L014-new-feature-evaluation.md": "M06-L013-new-feature-evaluation.md",
    
    # Design & Development Practices (14-15)
    "M06-L019-prompt-design-patterns.md": "M06-L014-prompt-design-patterns.md",
    "M06-L005-build-measure-learn-cycles.md": "M06-L015-build-measure-learn-cycles.md",
    
    # External Learning (16-19)
    "M06-L009-conference-meetup-participation.md": "M06-L016-conference-meetup-participation.md",
    "M06-L020-research-paper-review-sessions.md": "M06-L017-research-paper-review-sessions.md",
    "M06-L012-industry-trend-analysis.md": "M06-L018-industry-trend-analysis.md",
    "M06-L024-vendor-relationship-management.md": "M06-L019-vendor-relationship-management.md",
    
    # Organizational AI KPIs - Operational Metrics (20-21)
    "M06-L010-delivery-automation-metrics.md": "M06-L020-delivery-automation-metrics.md",
    "M06-L004-agent-portfolio-health.md": "M06-L021-agent-portfolio-health.md",
    
    # Adoption & Engagement (22-23)
    "M06-L021-tool-adoption-intensity.md": "M06-L022-tool-adoption-intensity.md",
    "M06-L016-skill-development-tracking.md": "M06-L023-skill-development-tracking.md",
    
    # Strategic Indicators (24)
    "M06-L013-ai-investment-roi.md": "M06-L024-ai-investment-roi.md",
}

def create_backup():
    """Create backup of current lessons before renaming"""
    if BACKUP_PATH.exists():
        shutil.rmtree(BACKUP_PATH)
    
    BACKUP_PATH.mkdir(parents=True)
    
    # Copy all lesson files to backup
    for module_dir in ["ai", "de"]:  # Add other modules as needed
        source_dir = Path(f"src/data/{module_dir}/lessons")
        if source_dir.exists():
            backup_module_dir = BACKUP_PATH / module_dir / "lessons"
            backup_module_dir.mkdir(parents=True)
            
            for lesson_file in source_dir.glob("*.md"):
                shutil.copy2(lesson_file, backup_module_dir / lesson_file.name)
    
    print(f"Backup created at: {BACKUP_PATH.absolute()}")

def resequence_module(module_mapping, module_name):
    """Resequence lessons for a specific module"""
    print(f"\nResequencing {module_name}...")
    
    # Use temporary directory to avoid conflicts
    temp_dir = LESSONS_PATH.parent / "temp_lessons"
    temp_dir.mkdir(exist_ok=True)
    
    renamed_count = 0
    
    for old_filename, new_filename in module_mapping.items():
        old_path = LESSONS_PATH / old_filename
        temp_path = temp_dir / new_filename
        
        if old_path.exists():
            shutil.move(str(old_path), str(temp_path))
            print(f"  Renamed: {old_filename} -> {new_filename}")
            renamed_count += 1
        else:
            print(f"  Missing: {old_filename}")
    
    # Move files back to original directory
    for temp_file in temp_dir.glob("*.md"):
        final_path = LESSONS_PATH / temp_file.name
        shutil.move(str(temp_file), str(final_path))
    
    # Clean up temp directory
    temp_dir.rmdir()
    
    print(f"{module_name} resequencing complete: {renamed_count} files renamed")

def verify_resequencing():
    """Verify that resequencing was successful"""
    print("\nVerifying resequencing...")
    
    modules = {
        "Module 1": list(MODULE_1_MAPPING.values()),
        "Module 2": list(MODULE_2_MAPPING.values()),
        "Module 3": list(MODULE_3_MAPPING.values()),
        "Module 4": list(MODULE_4_MAPPING.values()),
        "Module 5": list(MODULE_5_MAPPING.values()),
        "Module 6": list(MODULE_6_MAPPING.values()),
    }
    
    for module_name, expected_files in modules.items():
        missing_files = []
        for expected_file in expected_files:
            if not (LESSONS_PATH / expected_file).exists():
                missing_files.append(expected_file)
        
        if missing_files:
            print(f"ERROR {module_name}: Missing {len(missing_files)} files")
            for missing in missing_files[:5]:  # Show first 5
                print(f"    - {missing}")
        else:
            print(f"SUCCESS {module_name}: All {len(expected_files)} files present")

def main():
    """Main execution function"""
    print("AI Lesson Resequencing Script")
    print("=" * 50)
    
    # Create backup
    create_backup()
    
    # Resequence each module
    resequence_module(MODULE_1_MAPPING, "Module 1")
    resequence_module(MODULE_2_MAPPING, "Module 2")
    resequence_module(MODULE_3_MAPPING, "Module 3")
    resequence_module(MODULE_4_MAPPING, "Module 4")
    resequence_module(MODULE_5_MAPPING, "Module 5")
    resequence_module(MODULE_6_MAPPING, "Module 6")
    
    # Verify results
    verify_resequencing()
    
    print("\nLesson resequencing complete!")
    print(f"Backup available at: {BACKUP_PATH.absolute()}")
    print("\nSummary:")
    print(f"  - Module 1: {len(MODULE_1_MAPPING)} files resequenced")
    print(f"  - Module 2: {len(MODULE_2_MAPPING)} files resequenced")
    print(f"  - Module 3: {len(MODULE_3_MAPPING)} files resequenced")
    print(f"  - Module 4: {len(MODULE_4_MAPPING)} files resequenced")
    print(f"  - Module 5: {len(MODULE_5_MAPPING)} files resequenced")
    print(f"  - Module 6: {len(MODULE_6_MAPPING)} files resequenced")

if __name__ == "__main__":
    main()