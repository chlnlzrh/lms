"""
Extract all lessons for M1, M2, M3, and M4
"""
import re
import json

with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
    content = f.read()

# Extract each module section
modules = {}

# M1 section
m1_match = re.search(r'# M1:.*?(?=# M2:|$)', content, re.DOTALL)
if m1_match:
    modules['M1'] = m1_match.group(0)

# M2 section
m2_match = re.search(r'# M2:.*?(?=# M3:|$)', content, re.DOTALL)
if m2_match:
    modules['M2'] = m2_match.group(0)

# M3 section
m3_match = re.search(r'# M3:.*?(?=# M4:|$)', content, re.DOTALL)
if m3_match:
    modules['M3'] = m3_match.group(0)

# M4 section
m4_match = re.search(r'# M4:.*?(?=# M5:|$)', content, re.DOTALL)
if m4_match:
    modules['M4'] = m4_match.group(0)

# Map complexity to time
time_map = {"F": "45", "I": "60", "A": "75", "E": "90"}

all_lessons = []

for mod_code in ['M1', 'M2', 'M3', 'M4']:
    if mod_code not in modules:
        continue
    
    module_section = modules[mod_code]
    lines = module_section.split('\n')
    
    lessons = []
    current_focus = "General"
    
    # Extract module name
    mod_name_match = re.search(rf'# {mod_code}:\s*(.+?)\s*—', module_section)
    mod_name = mod_name_match.group(1).strip() if mod_name_match else f"{mod_code} Module"
    
    for line in lines:
        # Check for section headers
        section_match = re.match(r'^## ([^L].+)$', line)
        if section_match and not section_match.group(1).startswith('L'):
            current_focus = section_match.group(1)
            continue
        
        # Check for lesson headers
        lesson_match = re.match(r'^## L(\d+): (.+?) \[([FIAE])\]', line)
        if lesson_match:
            num = int(lesson_match.group(1))
            title = lesson_match.group(2).strip()
            complexity = lesson_match.group(3)
            lessons.append((num, title, complexity, current_focus))
    
    # Build lesson data
    for num, title, comp, focus in lessons:
        prev_num = num - 1
        next_num = num + 1
        
        # Get previous lesson title
        prev_title = None
        for pnum, ptitle, _, _ in lessons:
            if pnum == prev_num:
                prev_title = ptitle
                break
        
        # Prerequisites
        if num == 1:
            if mod_code == 'M1':
                prereqs = "Basic JavaScript/ES6, understanding of frontend development"
            elif mod_code == 'M2':
                prereqs = "Basic Node.js knowledge, understanding of HTTP and REST"
            elif mod_code == 'M3':
                prereqs = "Basic SQL knowledge, understanding of relational databases"
            elif mod_code == 'M4':
                prereqs = "Understanding of search concepts, basic API knowledge"
            else:
                prereqs = "Basic knowledge relevant to this module"
        else:
            prereqs = f"{mod_code}-L{prev_num:03d} ({prev_title})" if prev_title else f"{mod_code}-L{prev_num:03d}"
        
        # Related lessons
        related = []
        if prev_num >= 1:
            related.append(f"{mod_code}-L{prev_num:03d}")
        # Check if next lesson exists
        for nnum, _, _, _ in lessons:
            if nnum == next_num:
                related.append(f"{mod_code}-L{next_num:03d}")
                break
        related_str = ", ".join(related)
        
        lesson_dict = {
            "LESSON_CODE": f"{mod_code}-L{num:03d}",
            "LESSON_TITLE": title,
            "COMPLEXITY": comp,
            "TIME": time_map[comp],
            "LIST_PREREQUISITES": prereqs,
            "RELATED_LESSON_CODES": related_str,
            "SPECIFIC_FOCUS": focus,
            "MODULE_CODE": mod_code,
            "MODULE_NAME": mod_name,
        }
        all_lessons.append(lesson_dict)

# Generate output
print("# All lessons for M1, M2, M3, M4")
print("missing_lessons = [")
for i, lesson in enumerate(all_lessons):
    comma = "," if i < len(all_lessons) - 1 else ""
    print(f"    {{")
    print(f'        "LESSON_CODE": {json.dumps(lesson["LESSON_CODE"])},')
    print(f'        "LESSON_TITLE": {json.dumps(lesson["LESSON_TITLE"])},')
    print(f'        "COMPLEXITY": {json.dumps(lesson["COMPLEXITY"])},')
    print(f'        "TIME": {json.dumps(lesson["TIME"])},')
    print(f'        "LIST_PREREQUISITES": {json.dumps(lesson["LIST_PREREQUISITES"])},')
    print(f'        "RELATED_LESSON_CODES": {json.dumps(lesson["RELATED_LESSON_CODES"])},')
    print(f'        "SPECIFIC_FOCUS": {json.dumps(lesson["SPECIFIC_FOCUS"])},')
    print(f'        "MODULE_CODE": {json.dumps(lesson["MODULE_CODE"])},')
    print(f'        "MODULE_NAME": {json.dumps(lesson["MODULE_NAME"])},')
    print(f"    }}{comma}")
print("]")

print(f"\n# Summary: {len(all_lessons)} lessons total")
for mod_code in ['M1', 'M2', 'M3', 'M4']:
    count = sum(1 for l in all_lessons if l['MODULE_CODE'] == mod_code)
    print(f"  {mod_code}: {count} lessons")

