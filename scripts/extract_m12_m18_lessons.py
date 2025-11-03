"""
Extract lesson details for M12-M18 from the curriculum map
"""
import re

# Read the curriculum map
with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
    content = f.read()

# Module metadata (based on actual curriculum map)
MODULE_INFO = {
    "M12": {
        "MODULE_NAME": "Cloud, Infra & Platform Engineering",
        "AUDIENCE_DESCRIPTION": "Platform engineers and DevOps engineers building cloud infrastructure",
        "INDUSTRY_DOMAIN": "Cloud infrastructure, platform engineering, DevOps, IaC"
    },
    "M13": {
        "MODULE_NAME": "Analytics, Experimentation & Product",
        "AUDIENCE_DESCRIPTION": "Product engineers and data analysts building analytics and experimentation systems",
        "INDUSTRY_DOMAIN": "Product analytics, experimentation, metrics, data analysis"
    },
    "M14": {
        "MODULE_NAME": "AI-Native Capabilities",
        "AUDIENCE_DESCRIPTION": "AI/ML engineers and developers integrating AI capabilities into SaaS platforms",
        "INDUSTRY_DOMAIN": "AI/ML integration, LLMs, embeddings, vector search, AI-native features"
    },
    "M15": {
        "MODULE_NAME": "Security, Compliance & Risk",
        "AUDIENCE_DESCRIPTION": "Security engineers and compliance officers building secure and compliant SaaS platforms",
        "INDUSTRY_DOMAIN": "Security, compliance, risk management, audit"
    },
    "M16": {
        "MODULE_NAME": "Documentation, Process & Knowledge",
        "AUDIENCE_DESCRIPTION": "Technical writers and knowledge engineers building documentation systems",
        "INDUSTRY_DOMAIN": "Documentation, knowledge management, technical writing"
    },
    "M17": {
        "MODULE_NAME": "Leadership & Career Skills",
        "AUDIENCE_DESCRIPTION": "Engineering leaders and career-focused engineers",
        "INDUSTRY_DOMAIN": "Leadership, career development, management, soft skills"
    },
    "M18": {
        "MODULE_NAME": "Capstone",
        "AUDIENCE_DESCRIPTION": "Engineers completing capstone projects",
        "INDUSTRY_DOMAIN": "Capstone projects, portfolio building, real-world applications"
    }
}

lessons = []

# Find all modules M12-M18
for module_num in range(12, 19):
    module_code = f"M{module_num}"
    module_pattern = rf'# {module_code}: ([^\n]+) — Lesson Map'
    module_match = re.search(module_pattern, content)
    
    if not module_match:
        print(f"Warning: Module {module_code} not found in curriculum map")
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
with open("scripts/m12_m18_lessons.txt", "w", encoding="utf-8") as f:
    f.write("# All lessons for M12-M18\n")
    f.write("missing_lessons = [\n")
    for i, lesson in enumerate(lessons):
        comma = "," if i < len(lessons) - 1 else ""
        f.write(f"    {lesson}{comma}\n")
    f.write("]\n")

print(f"\nExtracted {len(lessons)} lessons for M12-M18")
print(f"Saved to scripts/m12_m18_lessons.txt")

