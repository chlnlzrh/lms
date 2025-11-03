"""
Build lesson data structure for remaining M0 lessons (L017-L072)
"""
import re

with open("src/data/saas/content_structure_ai-native-saas-curriculum-lesson-maps.md", "r", encoding="utf-8") as f:
    content = f.read()

# Extract M0 section
m0_match = re.search(r'# M0:.*?(?=# M1:|$)', content, re.DOTALL)
m0_section = m0_match.group(0)
lines = m0_section.split('\n')

lessons = []
current_focus = "Core Architectural Patterns"

for line in lines:
    section_match = re.match(r'^## ([^L].+)$', line)
    if section_match and not section_match.group(1).startswith('L'):
        current_focus = section_match.group(1)
        continue
    
    lesson_match = re.match(r'^## L(\d+): (.+?) \[([FIAE])\]', line)
    if lesson_match:
        num = int(lesson_match.group(1))
        title = lesson_match.group(2).replace('', '-')  # Fix Unicode dash
        complexity = lesson_match.group(3)
        lessons.append((num, title, complexity, current_focus))

# Map complexity to time (minutes)
time_map = {"F": "45", "I": "60", "A": "75", "E": "90"}

# Filter lessons >= L017
remaining = [l for l in lessons if l[0] >= 17]

# Build lesson data structures
lesson_data = []
for num, title, comp, focus in remaining:
    prev_num = num - 1
    next_num = num + 1 if num < 72 else None
    
    # Determine prerequisites
    if num == 17:
        prereqs = "M00-L016 (Evolutionary Architecture), understanding of software metrics, continuous integration"
    elif num == 18:
        prereqs = "M00-L017 (Architecture Fitness Functions), understanding of domain-driven design basics"
    else:
        prereqs = f"M00-L{prev_num:03d} ({lessons[prev_num-1][1] if prev_num <= len(lessons) else 'Previous Lesson'})"
        if num >= 26 and num <= 33:
            prereqs += ", understanding of distributed systems, consistency models"
        elif num >= 34 and num <= 41:
            prereqs += ", understanding of HTTP protocols, caching strategies"
        elif num >= 42 and num <= 48:
            prereqs += ", understanding of rate limiting algorithms, traffic management"
        elif num >= 49 and num <= 56:
            prereqs += ", understanding of API design, REST principles"
        elif num >= 57 and num <= 64:
            prereqs += ", understanding of reliability engineering, SLOs"
        elif num >= 65:
            prereqs += ", understanding of platform engineering, organizational patterns"
    
    # Determine related lessons
    related = []
    if prev_num >= 1:
        related.append(f"M00-L{prev_num:03d}")
    if next_num and next_num <= 72:
        related.append(f"M00-L{next_num:03d}")
    related_str = ", ".join(related)
    
    lesson_dict = {
        "LESSON_CODE": f"M00-L{num:03d}",
        "LESSON_TITLE": title,
        "COMPLEXITY": comp,
        "TIME": time_map[comp],
        "LIST_PREREQUISITES": prereqs,
        "RELATED_LESSON_CODES": related_str,
        "SPECIFIC_FOCUS": focus,
    }
    lesson_data.append(lesson_dict)

# Print Python list format
print("missing_lessons = [")
for i, lesson in enumerate(lesson_data):
    comma = "," if i < len(lesson_data) - 1 else ""
    print(f"    {repr(lesson)}{comma}")
print("]")

