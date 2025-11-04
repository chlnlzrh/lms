#!/usr/bin/env python3
"""
Lesson Validator
Validates generated lessons against quality standards and TSV specifications.
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import re

class LessonValidator:
    """Validates generated lesson files"""
    
    def __init__(self, base_path: str = "C:/ai/training/lms-platform"):
        self.base_path = Path(base_path)
        self.tsv_file = self.base_path / "final_layout" / "COMPLETE_LLM_LESSON_MANIFEST.tsv"
        
        # Validation criteria
        self.required_sections = [
            "Learning Objectives & Prerequisites",
            "Core Concepts",
            "Technical Implementation", 
            "Practical Examples & Exercises",
            "Step-by-Step Implementation Guide",
            "Best Practices & Considerations",
            "Common Pitfalls & Troubleshooting",
            "Assessment & Validation",
            "Real-World Applications",
            "Additional Resources & Next Steps"
        ]
        
        self.min_length = {
            'F': 300,  # Foundation
            'I': 400,  # Intermediate  
            'A': 500   # Advanced
        }
    
    def load_manifest(self) -> List[Dict]:
        """Load the TSV manifest"""
        lessons = []
        with open(self.tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                lessons.append(row)
        return lessons
    
    def validate_lesson_file(self, lesson_spec: Dict) -> Tuple[bool, List[str]]:
        """Validate a single lesson file"""
        errors = []
        
        # Check if file exists
        file_path = Path(lesson_spec['file_path']) / lesson_spec['filename']
        if not file_path.exists():
            errors.append(f"File not found: {file_path}")
            return False, errors
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
            return False, errors
        
        # Validate title format
        expected_title = f"# {lesson_spec['lesson_id']}: {lesson_spec['title']}"
        if not content.startswith(expected_title):
            errors.append(f"Title format incorrect. Expected: {expected_title}")
        
        # Validate required sections
        for section in self.required_sections:
            if f"## {section}" not in content:
                errors.append(f"Missing required section: {section}")
        
        # Validate minimum length
        line_count = len(content.split('\n'))
        min_lines = self.min_length.get(lesson_spec['difficulty'], 300)
        if line_count < min_lines:
            errors.append(f"Content too short: {line_count} lines (minimum: {min_lines})")
        
        # Validate code blocks for technical lessons
        if lesson_spec['code_complexity'] in ['Real-world scenarios', 'Production-ready']:
            code_blocks = re.findall(r'```[\w]*\n.*?\n```', content, re.DOTALL)
            if len(code_blocks) < 2:
                errors.append("Insufficient code examples for complexity level")
        
        return len(errors) == 0, errors
    
    def validate_all_lessons(self) -> Dict:
        """Validate all lessons in manifest"""
        lessons = self.load_manifest()
        
        results = {
            'total': len(lessons),
            'valid': 0,
            'invalid': 0,
            'missing': 0,
            'errors': []
        }
        
        for lesson in lessons:
            is_valid, errors = self.validate_lesson_file(lesson)
            
            if errors and "File not found" in errors[0]:
                results['missing'] += 1
            elif is_valid:
                results['valid'] += 1
                print(f"✅ {lesson['lesson_id']}: VALID")
            else:
                results['invalid'] += 1
                results['errors'].append({
                    'lesson_id': lesson['lesson_id'],
                    'errors': errors
                })
                print(f"❌ {lesson['lesson_id']}: INVALID")
                for error in errors:
                    print(f"   - {error}")
        
        return results
    
    def print_validation_summary(self, results: Dict) -> None:
        """Print validation summary"""
        print("\n" + "="*60)
        print("LESSON VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Lessons: {results['total']}")
        print(f"Valid: {results['valid']}")
        print(f"Invalid: {results['invalid']}")
        print(f"Missing: {results['missing']}")
        print(f"Validation Rate: {(results['valid']/results['total']*100):.1f}%")
        
        if results['errors']:
            print(f"\nInvalid Lessons ({len(results['errors'])}):")
            for error_info in results['errors'][:10]:
                print(f"  {error_info['lesson_id']}: {len(error_info['errors'])} issues")
        
        print("="*60)

def main():
    """Main entry point"""
    validator = LessonValidator()
    results = validator.validate_all_lessons()
    validator.print_validation_summary(results)
    
    # Exit with error code if validation failed
    if results['invalid'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()