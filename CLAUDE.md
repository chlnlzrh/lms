Python 3.11.9 can be found at C:\Users\bimal\AppData\Local\Programs\Python\Python311\python.exe
Machine: 10 core/20 thread/32GB RAM - use parallelism
GitHub: https://github.com/chlnlzrh/lms
Deploy: Vercel

Always use Claude Model claude-haiku-4-5-20251001 for API calls unless explicitly specified

API Configuration: Set ANTHROPIC_API_KEY environment variable for programmatic API calls not Claude Codes. Do not hardcode API keys in source files.

**Menu UX**: Desktop (collapsed icon-only, expands on hover/click) | Mobile (hamburger full-screen) | Auto-collapse | 300ms spring | Backdrop blur | Shadow gradients | Active states

## Golden Principle
"Always code as if someone else—who knows where you live—is going to maintain your code."

## 1. Context Management (MANDATORY)

BEFORE ANY CODE OPERATION:
- VERIFY complete context: README.md, requirements.txt/package.json, architecture docs
- REJECT if incomplete/stale
- VALIDATE scope boundaries - authorized directories only
- REFRESH on file modifications

**Context MUST include**: README.md, dependencies file, ARCHITECTURE.md, .env.example

**Scope**: ALLOW ./src/**, ./lib/**, ./tests/**, ./docs/**, ./scripts/** | FORBID ./node_modules/**, ./.env, ./secrets/**, ./.git/**

## 2. Prompt Structure (NON-NEGOTIABLE)

ACCEPT ONLY STRUCTURED REQUESTS: CONTEXT + GOAL + CONSTRAINTS + EXPECTED OUTPUT

REJECT: Vague ("make better"), missing context/constraints, unclear expectations

**Example**: CONTEXT: REST API FastAPI/PostgreSQL | GOAL: JWT auth | CONSTRAINTS: OAuth 2.0, bcrypt, refresh tokens | OUTPUT: Complete auth module with tests

## 3. UI/UX Enforcement

### 3.1 Tech Stack
**Framework**: Next.js 14+ App Router + TypeScript ONLY. Validate package.json/tsconfig.json. Reject non-compliance.

**Vercel-Native**: ALL backend via Vercel/Next.js services. Verify @vercel/* imports (postgres, blob, kv, edge-config). Block alternatives unless justified.

**Stack**: Auth.js v5+ | Postgres/KV | Blob Storage | Edge Config | API Routes/Server Actions/Edge Functions

**Checklist**: ✓ @vercel/* packages ✓ Env vars ✓ App Router patterns ✓ 'use server' directives ✓ Block Firebase/Supabase/AWS unless justified

**TypeScript**: NO JavaScript. Strict mode. .ts/.tsx only. Auto-convert JS to TS.

### 3.2 UI/UX Standards
**Self-Evidence**: Components immediately understandable. Conventional patterns > novel solutions.
**System Status**: ALL async ops = loading + empty + error states. Auto-generate missing states.
**Recognition**: Primary actions visible, not hidden. Require visible affordances.
**Design**: Shadcn UI + Tailwind tokens exclusively. Block hardcoded styles.
**Prevention**: Error prevention > error messages. Inline validation > post-submit errors.

## 4. Agent Platform UI/UX (MANDATORY)

### Typography & Menu
- ALL text: Inter `text-xs font-normal`. Headers: `font-bold` allowed, size stays `text-xs`
- Menus start collapsed: `expandedSections: []`. Click-to-expand only
- Menu items/headers: identical `text-xs font-normal`. Hierarchy via spacing/color, not weight
- Selected: `text-black dark:text-white` | Unselected: `text-gray-500` with hover
- Spacing: `py-1` to `py-1.5` max. Section: `space-y-0.5`. Compact vertical
- Content: Essential labels only. No descriptions/hover explanations
- Dashboard: Match menu typography. Headers `text-xs font-bold`, content `text-xs font-normal`

### Interactions & Behavior
- Animations: 300ms spring, staggered reveals, 60fps
- Hover: `text-gray-700 dark:text-gray-300`. Immediate feedback
- Headers: `text-xs`, chevron indicators, full-area click targets
- Nested: `ml-4`/`ml-6` indent, 3 levels max, parent-child indicators
- Icons: 16x16px max, Lucide React, descriptive labels (never standalone)
- Search: `text-xs` sizing, real-time filter, Cmd/Ctrl+K, clear "no results"
- Scroll: Smooth, scrollbar on hover, sticky headers, maintain selected visibility
- Mobile: Full-screen overlay/bottom sheet, `py-2` min, swipe dismiss
- A11y: Arrow/Enter/Escape nav, ARIA labels, focus trapping, WCAG compliance
- State: Persist expanded/collapsed (localStorage), maintain selection across refresh
- Loading: Skeleton loaders, progressive load, error states with retry
- Actions: Distinct treatment, confirm destructive, loading states
- Breadcrumbs: Reflect selection, `text-xs`, interactive segments

## 5. Layout & Responsive

Mobile-first (320px up). Test: 320/375/768/1024/1440px
Tailwind spacing tokens (4px/8px base). No hardcoded values
Handle notches/safe areas/folds. CSS safe area properties

## 6. Navigation

Clear persistent navigation. Max 5-7 top-level items
Breadcrumbs for hierarchical content. Current location indicators
EVERY modal/flow = clear exit (dismiss, back, Escape)

## 7. Forms

React Hook Form + Zod. Top-aligned labels. Progressive disclosure
Collect ONLY necessary data. Document purpose. Question every field
Inline validation. Actionable errors. Constraints > post-submit messages

## 8. Color & Theme

Semantic tokens only. NO hardcoded colors
WCAG AA: 4.5:1 normal, 3:1 large. Auto-validate contrast
Complete light + dark themes. Visual parity. Consider colorblindness

## 9. Motion & Interaction

Motion clarifies state/provides feedback. No gratuitous animation
Respect prefers-reduced-motion. Alternative feedback without motion
Optimistic UI: Immediate update, clear rollback on failure

## 10. Component Standards

Shadcn UI exclusively. Custom = extend, not replace
ALL components: default/hover/focus/loading/error/disabled states
ARIA attributes + keyboard support. Test with screen readers

## 11. Mobile

Touch targets: 44×44pt (iOS) / 48×48dp (Android) minimum
Follow platform patterns (iOS swipe-to-delete, Android long-press)
Mobile performance: Bundle size, image optimization, assume 3G

## 12. Development Workflow

### Planning (>50 lines)
Architecture diagram, component responsibilities, data flow, tech stack rationale

### Iterative
Max 200 lines per phase. Review after each. Map dependencies. Quality gates

### Review
Mandatory checklist: functionality, design, security, performance, maintainability

## 13. Architecture (ZERO TOLERANCE)

**Patterns**: Hexagonal, DI, Repository, CQRS, Event-driven

**SOLID**: Single Responsibility | Open/Closed | Liskov Substitution | Interface Segregation | Dependency Inversion

**Design Patterns**: Strategy, Factory, Observer, Adapter, Command. FORBID: God objects, singleton abuse, tight coupling

## 14. Testing (MANDATORY)

ALL CODE INCLUDES:
- Unit tests (every public method)
- Integration tests (external dependencies)
- Contract tests (API endpoints)
- Performance tests (critical paths)
- 80% coverage minimum

AAA pattern. Descriptive names. Edge cases + errors. Fixtures + mocks. Positive + negative cases

TDD when requested: Tests first → fail → implement → pass → refactor. BLOCK until tests complete

## 15. Quality Gates

**Pre-Gen**: Requirements complete, standards verified, security validated, architecture compliant
**Post-Gen**: Static analysis, vulnerability scan, performance check, docs complete, coverage validated
**Dimensions**: Functionality, Design, Security, Performance, Maintainability

## 16. Security (STRICT)

### Prevention
OWASP Top 10. Input validation/sanitization (ALL inputs). Parameterized queries. XSS prevention. Secret detection

### Data Protection
NO hardcoded secrets. Encrypt at rest/transit. Log access. Retention policies. GDPR for PII

### Auth
Proper patterns. Authorization checks (all protected resources). Secure sessions. Restrictive permissions default

## 17. Documentation

### Interface
Module docstrings, function docs (params/returns/exceptions), type annotations, usage examples, error handling

### Quality
Clear/concise. What + why. Update README. API docs. Migration guides

## 18. Debugging

### Problem Description
Current behavior, expected behavior, exact errors, relevant code, environment, recent changes

### Approach
Root cause analysis. Step-by-step methodology. Multiple fixes with pros/cons. Prevention strategies

### Performance
Current metrics, target metrics, profiling data. Analyze: algorithm efficiency, DB queries, memory, caching

## 19. AI Governance

### Prompt Safety
Sanitize inputs. Defend against injection. Filter high-risk patterns. Validate before processing

### Traceability
Model version metadata. Hidden tags: `@ai-gen: model=vX.Y, date`. Version logging

### Explainability
Human-readable rationale. Short "why" notes. Concise to avoid overload

### Context-Aware
Full project context required. Component library awareness. Maintain consistency with patterns

## 20. Collaboration

Align with ESLint/Prettier. Auto-format. Comply with VCS workflows (branch naming, PR requirements). Flag AI code for review with "AI-generated" labels

## 21. DevOps & CI/CD

ALL code passes gates (lint, test, security). Run pipeline simulations. Block unsafe merges
Enforce Vault/KMS/Secret Manager. Scan improper usage
Include structured logging, metrics, tracing. Validate framework usage

## 22. Data & API

Enforce schema compliance (OpenAPI, GraphQL). Compare vs contracts. Allow "draft" tag for experimental
Include rate limiting. Validate throttling. Suggest defaults
Align with GDPR/HIPAA/PCI-DSS. Validate storage/handling

## 23. Advanced Techniques

### Refactoring
Comprehensive strategy, phased approach, risk assessment, rollback plan, testing strategy, migration guide

### Code Gen
Follow project patterns, error handling, monitoring/logging, test suites, pattern consistency

### Migration
Detailed strategy, timeline, compatibility layer, data migration, risk mitigation, rollback

### Progressive Enhancement
Graceful degradation, a11y, performance, error boundaries. Baseline first, enhance progressively

### Integration
TypeScript compat, prop consistency, styling integration. Feel native to codebase

## 24. Governance

Log ALL outputs. Track acceptance/modification + timestamp. Preserve privacy
AI CANNOT self-approve. Require human reviewer
Map to ISO 27001, SOC 2. Generate compliance checklists

## 25. Override & Emergency

Humans override with justification. Explicit security approval. Audit trail
Immediate stop capability. Preserve safe state. Clear escalation. Auto-escalate security violations
Escalate: Security-critical, architectural, performance-critical, repeated violations, unusual patterns
Design system changes: Approval + documentation. Token tracking, pattern deviation analysis
Critical a11y/usability override normal flows. Severity assessment, impact validation
Performance degradation triggers optimization. Metric monitoring, threshold validation

## 26. Monitoring

Track: Compliance, quality trends, security violations (immediate alert), regular reports
Metrics: Quality scores, coverage, violations, docs completeness, architecture compliance
Feedback, analyze patterns, suggest optimizations, adapt rules
Pre-Gen: Validate completeness. Post-Gen: Test compliance. Continuous: Learn, improve

## 27. Command-Specific

**add-context**: Validate file completeness/scope, required docs, freshness
**implement**: Require testing, security scan, quality gates, docs
**review**: Complete checklist, specific recommendations, validate governance, require approval
**refactor**: Require strategy, risk assessment, backward compat, benefit justification

## 28. Enforcement Priorities

**CRITICAL (BLOCK)**: Security vulnerabilities, missing tests, incomplete context, architecture violations
**HIGH (WARN)**: Docs gaps, performance concerns, pattern misuse, quality issues
**ADVISORY (SUGGEST)**: Style consistency, optimizations, alternatives, best practices

## 29. Implementation Phases

**Phase 1**: Tech stack, component a11y, design tokens, responsive validation
**Phase 2**: Form UX, navigation, performance, security
**Phase 3**: Content quality, motion, progressive enhancement, cross-platform
**Phase 4**: AI UX suggestions, predictive a11y, performance automation, design system evolution

## 30. Typography (Legacy)

Tokenized scale. Proper hierarchy. Font size/line-height/spacing tokens
Clear, scannable, jargon-free. Readability analysis
System fonts with fallbacks. font-display: swap. Optimize delivery

## 31. Security Extended

### Data Minimization
Minimize collection. Explicit consent. PII analysis

### Secure Implementation
Secure auth/data patterns. Vulnerability assessment. Secure defaults

## 32. AI Agent Building (Anthropic/Claude Code)

### Core Principles
Autonomous operation: receive → plan → act → evaluate → iterate. Secure tool access, action logging, human-in-loop, comprehensive monitoring

### Tech Stack
Claude (Opus/Sonnet/Code) via API/desktop. Python for workflows/backend. TypeScript/React/Next.js for UI. Claude Code for multi-agent management

### Architecture
Iterative loop: Input → Plan/Reason → Tool Calls → Reflect → Repeat/Output. Lead orchestrator decomposes, distributes, synthesizes. Log every step/call/trace/error

### Tool Design
Explicit model-readable schemas. Sandboxed Docker/Xvfb for system access. Atomic reversible actions. Model-readable guides/errors

### Safety & Evaluation
Limits: Iterations, resources, tool access. High-risk = human review. Persist logs with timestamps. Employ subagents for validation

### UX
IDE panes: Logs, feedback, tool state, overrides. Clear, agent-friendly codebase. Fully documented prompts/APIs

### Memory
Vector DBs (Pinecone) for context/search. Memory scope management. Log/audit updates

### Integration
RESTful APIs, Computer Use, model-readable schemas. Backward compatibility. Migration paths

### Evaluators
Subagents for review/validation/auditing. Confidence thresholds. Feedback loops

### Structure
Separate: Orchestrator, tools, prompts. Version control configs/prompts. Document capabilities/limitations

## 33. Python
Python 3.11.9+. Validate path/env vars. pip + virtualenv

## 34. Golden Principle
"Always code as if someone else—who knows where you live—is going to maintain your code."

## 35. UI States

**Loading**: Skeleton (content), spinner (<3s), progress (longer), messages (>5s)
**Empty**: Why empty, next steps, illustration/icon, design consistency
**Error**: What/why/how to fix, fallback/retry

## 36. Forms

**Inputs**: Clear labels (not placeholders), helper text, real-time validation, proper types, length indicators
**Buttons**: Clear action text, loading state, disable during process, no layout shift
**Multi-Step**: Progress indicator, save draft, back without loss, clear titles

## 37. Tables

**Responsive**: Cards on mobile, horizontal scroll + fixed columns, column toggles, row actions accessible
**Features**: Sort, filter, pagination/infinite scroll, bulk actions, export

## 38. Modals & Dialogs

**Behavior**: Dim background, close on Escape, close on background click (unless destructive), return focus, prevent scroll
**Content**: Clear title, concise body, explicit actions, cancel available, loading states

## 39. Navigation Details

**Tabs**: Active indicator, keyboard nav (arrows), URL state, lazy load, ARIA
**Sidebar**: Consistent width, collapse/expand, clear location, group items, search for long lists
**Bottom (Mobile)**: 3-5 items max, labels + icons, indicate current, hide on scroll down, safe area

## 40. Performance

**Images**: WebP/AVIF, lazy load, width/height attrs, responsive srcsets, blur-up placeholders
**Bundles**: Initial <200KB, lazy routes/components, tree-shake, bundle analyzer, code-split routes
**Budgets**: FCP <1.5s, TTI <3s, CLS <0.1, FID <100ms, score >90
