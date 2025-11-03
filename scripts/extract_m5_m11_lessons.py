"""
Extract lesson details for M5-M11 from the curriculum map
"""
import re

# Read the curriculum map
with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
    content = f.read()

# Module metadata
MODULE_INFO = {
    "M5": {
        "MODULE_NAME": "Identity, AuthN/Z & Privacy",
        "AUDIENCE_DESCRIPTION": "Security engineers and developers building authentication and authorization systems",
        "INDUSTRY_DOMAIN": "Identity and access management, security, privacy compliance"
    },
    "M6": {
        "MODULE_NAME": "Payments, Billing & Monetization",
        "AUDIENCE_DESCRIPTION": "Payment engineers and developers building billing and subscription systems",
        "INDUSTRY_DOMAIN": "Payments, billing, subscriptions, revenue operations"
    },
    "M7": {
        "MODULE_NAME": "Files, Media & CDN",
        "AUDIENCE_DESCRIPTION": "Media engineers and developers building file storage and CDN systems",
        "INDUSTRY_DOMAIN": "File storage, media processing, CDN, object storage"
    },
    "M8": {
        "MODULE_NAME": "Jobs, Schedulers & Integrations",
        "AUDIENCE_DESCRIPTION": "Backend engineers building job processing and integration systems",
        "INDUSTRY_DOMAIN": "Job queues, schedulers, integrations, workflow automation"
    },
    "M9": {
        "MODULE_NAME": "Testing & Quality Engineering",
        "AUDIENCE_DESCRIPTION": "QA engineers and developers building testing infrastructure",
        "INDUSTRY_DOMAIN": "Software testing, quality assurance, test automation"
    },
    "M10": {
        "MODULE_NAME": "CI/CD, Release & Environments",
        "AUDIENCE_DESCRIPTION": "DevOps engineers and developers building CI/CD pipelines",
        "INDUSTRY_DOMAIN": "CI/CD, DevOps, deployment, infrastructure automation"
    },
    "M11": {
        "MODULE_NAME": "Observability, SRE & Operations",
        "AUDIENCE_DESCRIPTION": "SREs and operations engineers building observability systems",
        "INDUSTRY_DOMAIN": "Observability, monitoring, SRE, operations"
    }
}

lessons = []

# Find all modules M5-M11
for module_num in range(5, 12):
    module_code = f"M{module_num}"
    module_pattern = rf'# {module_code}: ([^\n]+) — Lesson Map'
    module_match = re.search(module_pattern, content)
    
    if not module_match:
        continue
    
    module_title = module_match.group(1)
    module_start = module_match.end()
    
    # Find the next module or end of file
    next_module_pattern = rf'# M{module_num + 1}:'
    next_module_match = re.search(next_module_pattern, content[module_start:])
    module_end = module_start + next_module_match.start() if next_module_match else len(content)
    
    module_content = content[module_start:module_end]
    
    # Extract lessons from this module
    lesson_pattern = r'## L(\d{3}): ([^\n]+) \[([FIAE])\]'
    for match in re.finditer(lesson_pattern, module_content):
        lesson_num = match.group(1)
        lesson_title = match.group(2).strip()
        complexity = match.group(3)
        
        lesson_code = f"{module_code}-L{lesson_num}"
        
        # Determine time based on complexity
        time_map = {"F": "45", "I": "60", "A": "75", "E": "90"}
        time = time_map.get(complexity, "60")
        
        # Build prerequisites (simplified - assume previous lesson)
        prev_lesson_num = int(lesson_num) - 1
        if prev_lesson_num > 0:
            prev_code = f"{module_code}-L{prev_lesson_num:03d}"
            prerequisites = f"{prev_code} (previous lesson in module)"
        else:
            prerequisites = f"Understanding of {MODULE_INFO[module_code]['INDUSTRY_DOMAIN']} basics"
        
        # Build related lessons
        next_lesson_num = int(lesson_num) + 1
        related_codes = [f"{module_code}-L{prev_lesson_num:03d}"] if prev_lesson_num > 0 else []
        if next_lesson_num <= 999:
            related_codes.append(f"{module_code}-L{next_lesson_num:03d}")
        
        lesson_dict = {
            "LESSON_CODE": lesson_code,
            "LESSON_TITLE": lesson_title,
            "COMPLEXITY": complexity,
            "TIME": time,
            "LIST_PREREQUISITES": prerequisites,
            "RELATED_LESSON_CODES": ", ".join(related_codes) if related_codes else lesson_code,
            "SPECIFIC_FOCUS": MODULE_INFO[module_code]['INDUSTRY_DOMAIN'],
            "MODULE_CODE": module_code,
            "MODULE_NAME": MODULE_INFO[module_code]['MODULE_NAME']
        }
        lessons.append(lesson_dict)

# Output as Python list
print("missing_lessons = [")
for i, lesson in enumerate(lessons):
    comma = "," if i < len(lessons) - 1 else ""
    print(f"    {lesson}{comma}")
print("]")

# Also save to file
with open("scripts/m5_m11_lessons.txt", "w", encoding="utf-8") as f:
    f.write("# All lessons for M5-M11\n")
    f.write("missing_lessons = [\n")
    for i, lesson in enumerate(lessons):
        comma = "," if i < len(lessons) - 1 else ""
        f.write(f"    {lesson}{comma}\n")
    f.write("]\n")

print(f"\nExtracted {len(lessons)} lessons for M5-M11")
print(f"Saved to scripts/m5_m11_lessons.txt")

