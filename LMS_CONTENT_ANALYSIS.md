# LMS Platform Content Analysis & Structure Reference

## Overview

The LMS platform is a comprehensive professional development ecosystem containing 22 specialized learning tracks with well-structured curricula, targeting technical and business professionals across various domains.

## Directory Structure Analysis

### Location: `src/data/`

The platform organizes content into specialized tracks, each containing:
- Content structure documents (curriculum maps)
- Lesson files (Markdown format)
- Module descriptions (JSON configuration)
- Lesson generation configs

## Learning Tracks Catalog

### Technical Engineering Tracks

#### AI/Machine Learning
- **Directory:** `ai/`
- **Modules:** 6 modules (M1-M6)
- **Total Lessons:** 234 lessons
- **Duration:** 79 hours total
- **Focus:** AI fundamentals, SDLC integration, agent architecture, strategy

**Module Breakdown:**
- M1: AI Foundation & Tool Fluency (66 lessons, 22 hours)
- M2: AI in SDLC (32 lessons, 11 hours)
- M3: AI-Augmented Engineering (48 lessons, 16 hours)
- M4: AI Agent & Platform Architecture (43 lessons, 14 hours)
- M5: AI Strategy & Governance (23 lessons, 8 hours)
- M6: Continuous Learning & Innovation (24 lessons, 8 hours)

#### SaaS Development
- **Directory:** `saas/`
- **Modules:** 19 modules (M0-M18)
- **Structure:** Complete full-stack development lifecycle
- **Focus:** Production-ready SaaS architecture

**Module Progression:**
- M0: SaaS Architecture & System Design
- M1: Frontend Foundations (TypeScript/React/Next.js)
- M2: Backend & API Development
- M3: Data, Storage & ORM
- M4: Search, Retrieval & Recommendations
- M5: Identity, AuthN/Z & Privacy
- M6: Payments, Billing & Monetization
- M7: Files, Media & CDN
- M8: Jobs, Schedulers & Integrations
- M9: Testing & Quality Engineering
- M10: CI/CD, Release & Environments
- M11: Observability, SRE & Operations
- M12: Cloud, Infra & Platform Engineering
- M13: Analytics, Experimentation & Product
- M14: AI-Native Capabilities
- M15: Security, Compliance & Risk
- M16: Documentation, Process & Knowledge
- M17: Leadership & Career Skills
- M18: Capstone Projects

#### Data & Integration Tracks
- **`data_engineer/`** - Data Engineering fundamentals
- **`mdm/`** - Master Data Management & governance
- **`snowflake_tune/`** - Snowflake optimization & best practices
- **`rpa/`** - Robotic Process Automation
- **`workato/`** - Integration platform specialization
- **`sfdc/`** - Salesforce consulting & development
- **`data_gov/`** - Data governance frameworks

#### Infrastructure & Operations
- **`devops_engineer/`** - DevOps practices & tooling
- **`viz_engineer/`** - Data visualization engineering
- **`ta/`** - Technical architecture patterns
- **`qa/`** - Quality assurance & testing

### Business & Professional Tracks

#### Management & Strategy
- **`pm/`** - Product Management
- **`ba/`** - Business Analysis
- **`hr/`** - Human Resources
- **`sales/`** - Sales methodology & enablement
- **`marketing/`** - Marketing strategy & execution
- **`finance/`** - Financial analysis & management

#### Specialized Engineering Roles
- **`sfdc_engineer/`** - Salesforce engineering
- **`workato_engineer/`** - Integration engineering

## Content Structure & Standards

### Lesson Organization

#### Complexity Levels
- **[F] Foundational** - Basic concepts, prerequisites
- **[I] Intermediate** - Practical application, integration
- **[A] Advanced** - Complex problem-solving, experience-dependent
- **[E] Expert** - Specialized knowledge, architectural decisions

#### Lesson Format (Standardized)
```markdown
# Lesson M##-L###: [Title]

## Lesson Metadata
- Lesson Code: M##-L###
- Module: M## — [Module Title]
- Complexity Level: [F/I/A/E]
- Duration: [X] minutes
- Target Audience: [Specific role/experience level]
- Audience Relevance: [Why this matters for the target audience]

## Learning Objectives
1. Define [specific concept]
2. Identify [trade-offs/patterns]
3. Apply [framework/methodology]
4. Evaluate [real-world scenarios]
5. Design [implementation approach]

## Core Content
[Technical explanations with code examples]
[Practical implementations]
[Real-world applications]
```

### Module Configuration

#### JSON Structure (`modules-descriptions/module.json`)
```json
[
  {
    "id": "module-1",
    "title": "[Module Title]",
    "subtitle": "[Brief description]",
    "duration": "[X] hours",
    "lessons": [number],
    "labs": [number],
    "prerequisites": ["[Previous modules/knowledge]"],
    "keyTopics": ["[Core topic areas]"],
    "skillsGained": ["[Specific competencies developed]"]
  }
]
```

### Content Generation System

#### Lesson Generation Configuration
Files: `.lesson-gen-config.json`
- API provider: Claude (Haiku 4.5)
- Audience targeting
- Terminology customization
- Language preferences

#### Navigation System
File: `src/data/navigation.ts`
- TypeScript-based menu structure
- Hierarchical organization
- Lesson count tracking
- Status indicators (active/coming-soon)

## Content Quality & Depth Analysis

### AI Track Deep Dive
**Comprehensive Coverage:**
- Fundamental concepts (transformers, tokenization, embeddings)
- Practical tools (Claude Teams, Claude Code, Cursor, GitHub Copilot)
- Implementation patterns (RAG, agent architecture)
- Enterprise considerations (governance, strategy, risk)

**Code Examples Quality:**
- Production-ready patterns
- Real-world constraints (token limits, costs)
- Security considerations
- Performance optimization

### SaaS Track Architecture
**Production Focus:**
- Multi-tenancy patterns (row-level security, tenant isolation)
- Scalability considerations (caching, rate limiting)
- Security implementation (AuthN/Z, compliance)
- Operational excellence (observability, incident response)

**Business Integration:**
- Unit economics modeling
- Compliance frameworks (GDPR, SOC2, HIPAA)
- Go-to-market considerations
- Customer segmentation strategies

## Platform Features & Capabilities

### Learning Experience
- **Progressive Difficulty:** Foundation → Advanced → Expert
- **Hands-on Labs:** Practical application exercises
- **Real-world Context:** Industry-relevant scenarios
- **Capstone Projects:** Skill validation and portfolio building

### Technical Implementation
- **Modern Stack:** Next.js 14, TypeScript, Tailwind CSS
- **AI Integration:** Claude Haiku 4.5 for content generation
- **Responsive Design:** Mobile-first approach
- **Navigation:** Collapsible, hierarchical menu system

### Content Management
- **Version Control:** Git-based content management
- **Automated Generation:** AI-assisted lesson creation
- **Standardized Format:** Consistent structure across tracks
- **Metadata Rich:** Comprehensive tagging and organization

## Industry Applications

### Target Audiences
- **Technical Professionals:** Software engineers, data engineers, DevOps engineers
- **Business Professionals:** Product managers, business analysts, sales teams
- **Leadership:** Technical leads, engineering managers, executives
- **Specialists:** Integration engineers, compliance professionals, security experts

### Use Cases
- **Individual Learning:** Skill development and career advancement
- **Team Training:** Organizational capability building
- **Certification Preparation:** Structured learning paths
- **Knowledge Base:** Reference documentation and best practices

## Competitive Advantages

### Comprehensive Coverage
- Full-stack development lifecycle
- Business and technical integration
- Industry-specific specializations
- Emerging technology focus (AI-native development)

### Quality Standards
- Expert-level content depth
- Production-ready examples
- Real-world constraints and trade-offs
- Continuous updates and evolution

### Learning Design
- Outcome-focused objectives
- Practical application emphasis
- Progressive skill building
- Industry relevance validation

---

## Navigation Reference

### Main Menu Structure
```
Dashboard
Employee Onboarding (coming-soon)
Talent Development
├── AI Training (234 lessons)
│   ├── Module 1: AI Foundation & Tool Fluency (66 lessons)
│   ├── Module 2: AI in SDLC (32 lessons)
│   ├── Module 3: AI-Augmented Engineering (48 lessons)
│   ├── Module 4: AI Agent & Platform Architecture (42 lessons)
│   ├── Module 5: AI Strategy & Governance (23 lessons)
│   └── Module 6: Continuous Learning & Innovation (23 lessons)
├── Data Engineering (300 lessons)
│   ├── Module 1: Database Fundamentals
│   ├── Module 2: SQL & ELT Concepts
│   ├── Module 3: Data Warehousing Principles
│   ├── Module 4: Data Modeling
│   ├── Module 5: Snowflake Specific Knowledge
│   └── Module 20: Emerging Topics & Advanced Concepts
├── Integration (coming-soon)
├── SaaS App Build (coming-soon)
├── Salesforce (coming-soon)
└── MDM & Data Governance (coming-soon)
Compliance Training (coming-soon)
Sales Enablement (coming-soon)
Customer Education (coming-soon)
Calendar/Events (coming-soon)
Learning Catalog
Search
Profile/My Account
Support/Help
Administration (coming-soon)
```

This analysis represents the current state of the LMS platform content structure as of November 2, 2025.