"""
Generate only M00-L011 lesson using the LESSON_GENERATION_PROMPT_GENERIC.md prompt
"""
import os
import sys
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
    raise ValueError("OpenAI API key required in openaiapikey.txt")

client = AsyncOpenAI(api_key=api_key)
MODEL_NAME = "gpt-5-mini"
FALLBACK_MODEL = "gpt-5-nano"

# Note: Prompt template and content structure are reloaded fresh in generate_lesson() 
# to ensure we always use the latest version, even if the script is long-running

# Module metadata
MODULE_METADATA = {
    "M0": {
        "MODULE_NAME": "SaaS Architecture & System Design",
        "AUDIENCE_DESCRIPTION": "Software engineers and architects building or evaluating SaaS platforms",
        "INDUSTRY_DOMAIN": "SaaS platform development, multi-tenant systems"
    }
}

DEFAULT_AUDIENCE = "Software engineers building SaaS platforms"
DEFAULT_FIRM_TYPE = "Technology companies and engineering teams"
DEFAULT_INDUSTRY = "SaaS platform development"

# Lesson details for M00-L011
lesson_details = {
    "LESSON_CODE": "M00-L011",
    "LESSON_TITLE": "Microservices Without Regret: Preconditions and Anti-Patterns",
    "MODULE_CODE": "M0",
    "MODULE_NAME": "SaaS Architecture & System Design",
    "SPECIFIC_FOCUS": "Core Architectural Patterns",
    "COMPLEXITY": "A",
    "TIME": "75",
    "LIST_PREREQUISITES": "M00-L010 (Modular Monolith: Package Boundaries and Dependency Rules), understanding of microservices architecture basics",
    "RELATED_LESSON_CODES": "M00-L010, M00-L012"
}

async def generate_lesson():
    # Reload prompt template fresh to ensure we always use the latest version
    with open("LESSON_GENERATION_PROMPT_GENERIC.md", "r", encoding="utf-8") as f:
        current_prompt_template = f.read()
    
    # Prepare variables for the prompt template
    complexity_label = {'F': 'Foundation', 'I': 'Intermediate', 'A': 'Advanced', 'E': 'Expert'}.get(lesson_details['COMPLEXITY'], 'Foundation')
    module_code = lesson_details.get('MODULE_CODE', 'M0')
    module_name = lesson_details.get('MODULE_NAME', MODULE_METADATA.get(module_code, {}).get('MODULE_NAME', 'Unknown Module'))
    audience_desc = MODULE_METADATA.get(module_code, {}).get('AUDIENCE_DESCRIPTION', DEFAULT_AUDIENCE)
    industry_domain = MODULE_METADATA.get(module_code, {}).get('INDUSTRY_DOMAIN', DEFAULT_INDUSTRY)
    
    # Evaluate the prompt template as an f-string
    template_content = current_prompt_template.strip()
    if template_content.startswith('prompt = f"""'):
        template_content = template_content[13:]
    if template_content.endswith('"""'):
        template_content = template_content[:-3]
    template_content = template_content.strip()
    
    # Replace all the f-string expressions
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
    
    # Reload content structure fresh for each lesson
    with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
        current_content_structure = f.read()
    
    # Add the content structure reference
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

    print(f"\n[Generating] {lesson_details['LESSON_CODE']} - {lesson_details['LESSON_TITLE']}")
    print(f"  Complexity: {lesson_details['COMPLEXITY']} ({complexity_label}) | Duration: {lesson_details['TIME']} minutes")
    
    # Use gpt-5-mini as primary (it uses max_completion_tokens, not max_tokens)
    model_to_use = MODEL_NAME
    print(f"  Using model: {model_to_use}")
    
    # API params
    api_params = {
        "model": model_to_use,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    if model_to_use == "gpt-5-mini":
        api_params["max_completion_tokens"] = 16384  # gpt-5-mini uses max_completion_tokens, not max_tokens
        # gpt-5-mini only supports default temperature (1), don't set it
    
    try:
        # OpenAI async streaming
        print("  Streaming response...")
        stream = await client.chat.completions.create(**api_params)
        
        lesson_content_parts = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                lesson_content_parts.append(content)
                # Print progress dots
                if len(lesson_content_parts) % 100 == 0:
                    print(".", end="", flush=True)
        lesson_content = "".join(lesson_content_parts)
        print()  # New line after dots
        
    except Exception as e:
        print(f"  [WARNING] Streaming failed, trying non-streaming: {e}")
        non_stream_params = {
            "model": model_to_use,
            "messages": [{"role": "user", "content": prompt}]
        }
        if model_to_use == "gpt-5-mini":
            non_stream_params["max_completion_tokens"] = 16384  # gpt-5-mini uses max_completion_tokens, not max_tokens
            # gpt-5-mini only supports default temperature (1), don't set it
        
        response = await client.chat.completions.create(**non_stream_params)
        lesson_content = response.choices[0].message.content

    # Post-process to remove any prompt artifacts
    import re
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
        match = re.search(r'(^# Lesson .*|^# Section 1:.*|^## Section 1:.*|^Section 1:)', cleaned_content, re.MULTILINE)
        if match:
            cleaned_content = cleaned_content[match.start():]

    # Generate filename
    date_str = datetime.now().strftime("%Y-%m-%d")
    title_slug = lesson_details['LESSON_TITLE'].lower()
    title_slug = re.sub(r'[^\w\s-]', '', title_slug)
    title_slug = re.sub(r'[-\s]+', '-', title_slug)
    filename = f"{lesson_details['LESSON_CODE']}-{title_slug}--{date_str}.md"
    
    output_path = f"src/data/saas/lessons/{filename}"
    os.makedirs("src/data/saas/lessons", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)
    
    lines = len(cleaned_content.split('\n'))
    chars = len(cleaned_content)
    
    print(f"\n[SUCCESS] {lesson_details['LESSON_CODE']}: {lines} lines, {chars:,} chars")
    print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_lesson())

