prompt = f"""
You are generating a polished, publication-ready lesson for an AI-Native SaaS Curriculum.
Read and internalize these two inputs before writing anything:

1. LESSON_GENERATION_PROMPT_GENERIC.md — this defines the canonical 10-section structure.
2. content_structure_ai-native-saas-curriculum-lesson-maps.md — this defines where the lesson fits in the broader curriculum.

---

### LESSON METADATA
- Lesson Code: {lesson_details['LESSON_CODE']}
- Lesson Title: {lesson_details['LESSON_TITLE']}
- Module: {lesson_details.get('MODULE_CODE', 'M0')} — {lesson_details.get('MODULE_NAME', MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('MODULE_NAME', 'Unknown Module'))}
- Subtopic / Focus Area: {lesson_details.get('SPECIFIC_FOCUS', 'General')}
- Complexity Level: {lesson_details['COMPLEXITY']} ({'Foundation' if lesson_details['COMPLEXITY'] == 'F' else 'Intermediate' if lesson_details['COMPLEXITY'] == 'I' else 'Advanced' if lesson_details['COMPLEXITY'] == 'A' else 'Expert'})
- Estimated Duration: {lesson_details['TIME']} minutes
- Target Audience: {MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('AUDIENCE_DESCRIPTION', DEFAULT_AUDIENCE)}
- Organization Context: {DEFAULT_FIRM_TYPE}

**Prerequisites & Context**
- Prior Knowledge Required: {lesson_details['LIST_PREREQUISITES']}
- Related Lessons: {lesson_details['RELATED_LESSON_CODES']}
- Domain Context: {MODULE_METADATA.get(lesson_details.get('MODULE_CODE', 'M0'), {}).get('INDUSTRY_DOMAIN', DEFAULT_INDUSTRY)}

---

### INSTRUCTIONS

You must generate a **complete, polished lesson** following the exact 10-section structure of the template.
Each section must be written in clean, engaging, professional prose using proper Markdown syntax.

**Length & Depth Guidelines**
- [F] Foundation: ~350–400 lines, 2–3 core concepts, 3 scenarios  
- [I] Intermediate: ~400–500 lines, 3–4 core concepts, 4 scenarios  
- [A] Advanced: 500+ lines, 4+ concepts, multiple scenarios  

**Content & Technical Requirements**
1. Include working code examples (≥15 lines each) demonstrating realistic SaaS architecture patterns.  
   Languages accepted: TypeScript/Node.js, SQL, Terraform, or YAML.  
2. Use real metrics and trade-offs (e.g., “reduces provisioning time by 60%”, “saves $50K annually”).  
3. Anchor explanations in SaaS-specific terminology: multi-tenancy, tenant isolation, subscription models, usage-based pricing, CI/CD, and observability.  
4. Connect code and architecture examples to common SaaS stacks — Next.js, Postgres, Vercel, AWS, and Terraform.  
5. Discuss business impact, not just technology. Every technical concept must map to a measurable outcome (cost, velocity, reliability, compliance).  
6. Avoid filler phrases (“learn about”, “understand”); use precise, action-oriented verbs.  
7. Each code example must include a short contextual explanation and expected outcome.  
8. Every section should open with 1–2 framing sentences that orient the learner before details or lists.  
9. Use active voice, short sentences (under 25 words), and clear transitions.  
10. Maintain tone balance: professional, instructive, and confident — not academic or conversational.  

---

### FORMATTING & STYLE GUIDELINES

- **Headings:** Use `##` for major sections, `###` for subsections. Capitalize key words.  
- **Bullets:** Use `-` for list items. Avoid numbered lists unless order matters.  
- **Code Blocks:** Use fenced code blocks with language tags (` ```ts`, ` ```sql`, etc.) and concise comments.  
- **Tables:** Use clean Markdown tables with short headers; follow with 1–2 lines of interpretation guidance.  
- **Callouts:** Use icons like ✅, ⚠️, 💡 for emphasis when appropriate.  
- **Spacing:** Insert blank lines before and after headings, code blocks, and tables for readability.  
- **Tone & Flow:** Alternate between explanation and application. Every theoretical statement should lead into an example or outcome.  
- **Lexical Clarity:** When using technical terms (e.g., “tenancy isolation”), add a short clarifier in parentheses on first use.  

---

### QUALITY GUARANTEES

Your final output must:
- Contain all 10 sections from the template in order.  
- Have consistent structure, formatting, and flow.  
- Read naturally, as if authored by an experienced SaaS educator.  
- Contain no meta-text, instructions, or placeholders.  
- Begin immediately with `# Lesson {lesson_details['LESSON_CODE']}: {lesson_details['LESSON_TITLE']}` or the first section heading.  

Do **not** include:
- Any “LLM Prompt” or template boilerplate.  
- Any context or metadata explanation.  
- Any quality assurance notes at the end.

---

### FINAL OUTPUT

Deliver the **final, learner-facing Markdown lesson**, cleanly formatted and publication-ready.
"""
