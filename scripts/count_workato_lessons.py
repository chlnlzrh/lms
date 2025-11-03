#!/usr/bin/env python3
"""
Count lessons per module in the Workato curriculum.
"""

import re
from pathlib import Path

def count_lessons_per_module(file_path):
    """Parse the curriculum file and count lessons per module."""
    content = Path(file_path).read_text(encoding='utf-8')
    
    # Find all module headers: # M0:, # M1:, etc.
    module_pattern = r'^# (M\d+):\s+(.+?)\s+—\s+Lesson Map'
    
    # Find all lesson headers: ## L001:, ## L002:, etc.
    lesson_pattern = r'^## (L\d{3}):'
    
    modules = {}
    current_module = None
    
    for line in content.split('\n'):
        # Check for module header
        module_match = re.match(module_pattern, line)
        if module_match:
            module_code = module_match.group(1)
            module_title = module_match.group(2)
            current_module = module_code
            modules[current_module] = {
                'title': module_title,
                'lessons': []
            }
        
        # Check for lesson header
        lesson_match = re.match(lesson_pattern, line)
        if lesson_match and current_module:
            lesson_code = lesson_match.group(1)
            modules[current_module]['lessons'].append(lesson_code)
    
    # Print results
    print("Lesson Count per Module - Workato Curriculum\n")
    print(f"{'Module':<8} {'Count':<8} {'Title'}")
    print("-" * 80)
    
    total_lessons = 0
    for module_code in sorted(modules.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999):
        module = modules[module_code]
        count = len(module['lessons'])
        total_lessons += count
        print(f"{module_code:<8} {count:<8} {module['title']}")
    
    print("-" * 80)
    print(f"{'TOTAL':<8} {total_lessons:<8}")
    print(f"\nTotal modules: {len(modules)}")
    
    return modules

if __name__ == '__main__':
    file_path = Path('src/data/workato/content_structure_workato-curriculum-lesson-maps.md')
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        exit(1)
    
    modules = count_lessons_per_module(file_path)





