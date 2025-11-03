"""
Generate only L008 lesson
"""
import os
import sys
import time
import re
from datetime import datetime
from openai import OpenAI

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

with open("openaiapikey.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)
MODEL_NAME = "gpt-5-nano"
FALLBACK_MODEL = "gpt-5-mini"

with open("LESSON_GENERATION_PROMPT_GENERIC.md", "r", encoding="utf-8") as f:
    template = f.read()

with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
    content_structure = f.read()

lesson_details = {
    "LESSON_CODE": "M00-L008",
    "LESSON_TITLE": "Migration Playbook: Single-Tenant → Multi-Tenant",
    "COMPLEXITY": "A",
    "TIME": "90",
    "LIST_PREREQUISITES": "M00-L001 (The SaaS Landscape), M00-L003 (Choosing a Tenancy Model), experience with database migrations",
    "RELATED_LESSON_CODES": "M00-L007 (Compliance Implications), M00-L009 (Monolith First)",
}

MODULE_CODE = "M0"
MODULE_NAME = "SaaS Architecture & System Design"
SPECIFIC_FOCUS = "SaaS Basics: Tenancy & Isolation"
AUDIENCE_DESCRIPTION = "Software engineers and architects building or evaluating SaaS platforms"
FIRM_TYPE = "Technology companies and engineering teams"
INDUSTRY_DOMAIN = "SaaS platform development, multi-tenant systems"

prompt = f"""You are generating a lesson for an AI-native SaaS curriculum. Please read and remember these two documents:

1. LESSON_GENERATION_PROMPT_GENERIC.md (the template structure):
{template}

2. Content Structure (curriculum map):
{content_structure}

Now, generate a complete lesson following the template structure with these specific parameters:

**Lesson Metadata:**
- Lesson Code: {lesson_details['LESSON_CODE']}
- Lesson Title: {lesson_details['LESSON_TITLE']}
- Module: {MODULE_CODE} — {MODULE_NAME}
- Subtopic/Focus Area: {SPECIFIC_FOCUS}
- Complexity Level: {lesson_details['COMPLEXITY']} (Advanced)
- Estimated Duration: {lesson_details['TIME']} minutes
- Target Audience: {AUDIENCE_DESCRIPTION}
- Organization: {FIRM_TYPE}

**Prerequisites & Context:**
- Prior knowledge required: {lesson_details['LIST_PREREQUISITES']}
- Related lessons: {lesson_details['RELATED_LESSON_CODES']}
- Domain context: {INDUSTRY_DOMAIN}

**Important Instructions:**
1. Follow the 10-section structure exactly as specified in the template
2. For Advanced [A] level: 500+ lines, 4+ concepts, multiple scenarios with architectural depth
3. Include working code examples (15+ lines) with realistic SaaS architecture patterns
4. Use concrete metrics and trade-offs (e.g., "reduces costs by 40%", "$50K annually")
5. Ground everything in real SaaS business outcomes and engineering responsibilities
6. Focus on SaaS-specific terminology: multi-tenancy, tenant isolation, subscription models, usage-based pricing
7. Include code examples in TypeScript/Node.js, SQL, or infrastructure-as-code (Terraform/YAML)
8. Address common SaaS misconceptions
9. Connect to Next.js/Vercel, Postgres, and modern SaaS tech stack patterns
10. Write in professional prose, avoiding placeholder content

Generate the complete lesson content now, following all quality standards from the template."""

print("="*80)
print("Generating M00-L008 - Migration Playbook: Single-Tenant to Multi-Tenant")
print("="*80)

model_to_use = MODEL_NAME
try:
    _ = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "test"}]
    )
    model_to_use = MODEL_NAME
    print(f"Using {MODEL_NAME}...")
except Exception:
    print(f"{MODEL_NAME} not available, using fallback {FALLBACK_MODEL}...")
    model_to_use = FALLBACK_MODEL

api_params = {
    "model": model_to_use,
    "messages": [{"role": "user", "content": prompt}],
    "stream": True
}

if model_to_use != "gpt-5-nano":
    api_params["max_tokens"] = 16384
    api_params["temperature"] = 0.7

print("Calling OpenAI API with streaming...")
stream = client.chat.completions.create(**api_params)

lesson_content_parts = []
for chunk in stream:
    if chunk.choices[0].delta.content:
        content = chunk.choices[0].delta.content
        lesson_content_parts.append(content)
        print(".", end="", flush=True)
print()

lesson_content = "".join(lesson_content_parts)

# Generate filename - handle Unicode arrow properly
date_str = datetime.now().strftime("%Y-%m-%d")
filename_slug = lesson_details['LESSON_TITLE'].lower()
filename_slug = filename_slug.replace('→', 'to').replace('(', '').replace(')', '')
filename_slug = filename_slug.replace('/', '-').replace(':', '').replace(',', '').replace('&', 'and')
filename_slug = re.sub(r'[-\s]+', '-', filename_slug)
filename_slug = re.sub(r'[^a-z0-9\-]', '', filename_slug)
filename = f"M00-L008-{filename_slug}--{date_str}.md"

output_path = f"src/data/saas/lessons/{filename}"
os.makedirs("src/data/saas/lessons", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(lesson_content)

lines = len(lesson_content.split('\n'))
chars = len(lesson_content)

print(f"\n[SUCCESS] Lesson generated!")
print(f"  File: {filename}")
print(f"  Lines: {lines}")
print(f"  Characters: {chars:,}")
print(f"  Saved to: {output_path}")

