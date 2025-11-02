#!/usr/bin/env python3
"""
General Purpose Module Description Builder
Analyzes lessons in any track folder and generates comprehensive module descriptions.

Usage:
    python build_module_descriptions.py <track_folder_name>

Example:
    python build_module_descriptions.py mdm
    python build_module_descriptions.py rpa
    python build_module_descriptions.py workato
"""

import os
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import frontmatter

class ModuleDescriptionBuilder:
    def __init__(self, track_name: str):
        self.track_name = track_name
        self.base_path = Path(__file__).parent.parent / "src" / "data" / track_name
        self.lessons_path = self.base_path / "lessons"
        self.output_path = self.base_path / "modules-descriptions"
        
        # Track-specific configurations
        self.track_configs = {
            'mdm': {
                'theme': 'Master Data Management',
                'color': 'indigo',
                'icon': 'Database',
                'focus': 'data integration, governance, and quality'
            },
            'rpa': {
                'theme': 'Robotic Process Automation',
                'color': 'orange',
                'icon': 'Bot',
                'focus': 'automation, workflows, and efficiency'
            },
            'workato': {
                'theme': 'Integration Platform',
                'color': 'teal',
                'icon': 'Link',
                'focus': 'API integration, workflows, and connectivity'
            },
            'snowflake_tune': {
                'theme': 'Snowflake Performance',
                'color': 'cyan',
                'icon': 'Zap',
                'focus': 'performance optimization, tuning, and scaling'
            },
            'sfdc': {
                'theme': 'Salesforce Development',
                'color': 'blue',
                'icon': 'Cloud',
                'focus': 'CRM, platform development, and customization'
            },
            'data_gov': {
                'theme': 'Data Governance',
                'color': 'emerald',
                'icon': 'Shield',
                'focus': 'governance, compliance, and data stewardship'
            },
            'finance': {
                'theme': 'Financial Technology',
                'color': 'green',
                'icon': 'DollarSign',
                'focus': 'financial systems, compliance, and analytics'
            },
            'saas': {
                'theme': 'Software as a Service',
                'color': 'purple',
                'icon': 'Globe',
                'focus': 'cloud applications, scalability, and service delivery'
            },
            'ai': {
                'theme': 'Artificial Intelligence',
                'color': 'violet',
                'icon': 'Brain',
                'focus': 'machine learning, automation, and intelligent systems'
            },
            'de': {
                'theme': 'Data Engineering',
                'color': 'slate',
                'icon': 'Database',
                'focus': 'data pipelines, processing, and infrastructure'
            }
        }
    
    def validate_track(self) -> bool:
        """Validate that the track exists and has lessons."""
        if not self.lessons_path.exists():
            print(f"[ERROR] Lessons directory not found at {self.lessons_path}")
            return False
        
        lesson_files = list(self.lessons_path.glob("*.md"))
        if not lesson_files:
            print(f"[ERROR] No lesson files found in {self.lessons_path}")
            return False
        
        print(f"[SUCCESS] Found {len(lesson_files)} lesson files in {self.track_name} track")
        return True
    
    def extract_lesson_info(self, lesson_file: Path) -> Optional[Dict]:
        """Extract lesson information from markdown file."""
        try:
            with open(lesson_file, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            # Extract lesson and module numbers from filename
            # Patterns: l001.md -> (None, 1) or M0-L001.md -> (0, 1)
            module_lesson_match = re.search(r'(?:M(\d+)-)?[Ll](\d+)', lesson_file.stem)
            if not module_lesson_match:
                return None
            
            module_number = int(module_lesson_match.group(1)) if module_lesson_match.group(1) else None
            lesson_number = int(module_lesson_match.group(2))
            
            return {
                'number': lesson_number,
                'module_number': module_number,
                'title': post.metadata.get('title', 'Untitled Lesson'),
                'complexity': post.metadata.get('complexity', 'F'),
                'topics': post.metadata.get('topics', []),
                'content': post.content,
                'filename': lesson_file.name
            }
        except Exception as e:
            print(f"[WARNING] Could not parse {lesson_file.name}: {e}")
            return None
    
    def group_lessons_by_module(self, lessons: List[Dict]) -> Dict[int, List[Dict]]:
        """Group lessons into modules based on lesson numbers or explicit module numbers."""
        modules = {}
        
        # Check if lessons have explicit module numbers (e.g., M0-L001.md format)
        has_explicit_modules = any(lesson.get('module_number') is not None for lesson in lessons)
        
        if has_explicit_modules:
            # Use explicit module numbers from filenames
            for lesson in lessons:
                module_number = lesson.get('module_number', 0)
                if module_number not in modules:
                    modules[module_number] = []
                modules[module_number].append(lesson)
        else:
            # Fallback to grouping by lesson numbers (24 lessons per module)
            lessons_per_module = 24
            for lesson in lessons:
                module_number = lesson['number'] // lessons_per_module
                if module_number not in modules:
                    modules[module_number] = []
                modules[module_number].append(lesson)
        
        return modules
    
    def generate_module_title(self, module_number: int, lessons: List[Dict]) -> Tuple[str, str]:
        """Generate module title and subtitle based on lesson content."""
        config = self.track_configs.get(self.track_name, {})
        theme = config.get('theme', self.track_name.title())
        
        # Common module progression patterns
        module_titles = {
            0: ("Fundamentals & Foundation", f"Core concepts and introduction to {theme}"),
            1: ("Architecture & Design", f"System architecture and design patterns for {theme}"),
            2: ("Implementation & Development", f"Hands-on development and implementation techniques"),
            3: ("Integration & Connectivity", f"Integration patterns and connectivity solutions"),
            4: ("Advanced Features", f"Advanced capabilities and specialized features"),
            5: ("Performance & Optimization", f"Performance tuning and optimization strategies"),
            6: ("Security & Compliance", f"Security best practices and compliance requirements"),
            7: ("Monitoring & Observability", f"Monitoring, logging, and observability practices"),
            8: ("Automation & Workflows", f"Automation strategies and workflow optimization"),
            9: ("Data Management", f"Data handling, quality, and lifecycle management"),
            10: ("Governance & Standards", f"Governance frameworks and industry standards"),
            11: ("Analytics & Reporting", f"Analytics implementation and reporting solutions"),
            12: ("AI & Machine Learning", f"AI/ML integration and intelligent automation"),
            13: ("Enterprise Patterns", f"Enterprise-scale patterns and best practices"),
            14: ("Platform Operations", f"Operations, maintenance, and platform management"),
            15: ("Innovation & Future", f"Innovation strategies and future roadmap"),
            16: ("Troubleshooting & Support", f"Diagnostic techniques and support processes"),
            17: ("Migration & Transformation", f"Migration strategies and digital transformation"),
            18: ("Leadership & Strategy", f"Leadership skills and strategic planning"),
            19: ("Capstone & Assessment", f"Comprehensive project work and skill assessment")
        }
        
        if module_number in module_titles:
            return module_titles[module_number]
        
        # Fallback for modules beyond predefined range
        return (f"Module {module_number}", f"Advanced {theme} topics and specialized applications")
    
    def extract_key_topics(self, lessons: List[Dict]) -> List[str]:
        """Extract key topics from lesson content and metadata."""
        topics = set()
        
        for lesson in lessons:
            # Add topics from frontmatter
            lesson_topics = lesson.get('topics', [])
            if isinstance(lesson_topics, list):
                topics.update(lesson_topics)
            
            # Extract topics from lesson titles (common patterns)
            title = lesson['title'].lower()
            
            # Common technical terms extraction
            technical_terms = [
                'api', 'authentication', 'authorization', 'security', 'integration',
                'workflow', 'automation', 'database', 'performance', 'monitoring',
                'analytics', 'reporting', 'governance', 'compliance', 'architecture',
                'design patterns', 'best practices', 'optimization', 'scaling',
                'cloud', 'deployment', 'testing', 'debugging', 'troubleshooting'
            ]
            
            for term in technical_terms:
                if term in title:
                    topics.add(term.title())
        
        return sorted(list(topics))[:8]  # Limit to most relevant topics
    
    def generate_skills_gained(self, module_number: int, lessons: List[Dict]) -> List[str]:
        """Generate skills gained based on module content and progression."""
        config = self.track_configs.get(self.track_name, {})
        focus_areas = config.get('focus', 'technical skills').split(', ')
        
        # Base skills that evolve with module progression
        skill_templates = {
            'foundation': [
                f"Understanding of {self.track_name.upper()} fundamentals",
                "Technical terminology and concepts",
                "Platform navigation and basic operations"
            ],
            'intermediate': [
                f"Implementation of {focus_areas[0] if focus_areas else 'core features'}",
                "Configuration and customization techniques",
                "Integration with external systems"
            ],
            'advanced': [
                f"Advanced {focus_areas[1] if len(focus_areas) > 1 else 'optimization'} strategies",
                "Complex problem-solving approaches",
                "Leadership in technical decision-making"
            ]
        }
        
        # Determine skill level based on module number
        if module_number <= 5:
            base_skills = skill_templates['foundation']
        elif module_number <= 12:
            base_skills = skill_templates['intermediate']
        else:
            base_skills = skill_templates['advanced']
        
        # Add module-specific skills based on lesson content
        lesson_titles = [lesson['title'] for lesson in lessons]
        specific_skills = []
        
        # Extract specific skills from lesson patterns
        for title in lesson_titles[:3]:  # Use first 3 lessons as representative
            if 'build' in title.lower() or 'create' in title.lower():
                specific_skills.append(f"Building and creating {title.split()[-1].lower()} solutions")
            elif 'manage' in title.lower() or 'configure' in title.lower():
                specific_skills.append(f"Managing and configuring {title.split()[-1].lower()}")
            elif 'optimize' in title.lower() or 'improve' in title.lower():
                specific_skills.append(f"Optimizing and improving {title.split()[-1].lower()}")
        
        # Combine base and specific skills
        all_skills = base_skills + specific_skills[:2]  # Limit specific skills
        return all_skills[:5]  # Return top 5 skills
    
    def calculate_duration_and_labs(self, lessons: List[Dict]) -> Tuple[str, int]:
        """Calculate estimated duration and number of labs."""
        lesson_count = len(lessons)
        
        # Estimate duration based on lesson count and complexity
        complexity_weights = {'F': 1, 'I': 1.5, 'A': 2}
        total_weight = sum(complexity_weights.get(lesson.get('complexity', 'F'), 1) for lesson in lessons)
        
        # Base time per lesson: 30 minutes for foundation, 45 for intermediate, 60 for advanced
        estimated_hours = (total_weight * 0.75)  # 45 minutes average
        weeks = max(1, round(estimated_hours / 8))  # Assume 8 hours per week
        
        # Estimate labs (typically 20-30% of lessons are hands-on)
        lab_count = max(1, round(lesson_count * 0.25))
        
        duration = f"{weeks} week{'s' if weeks > 1 else ''}"
        return duration, lab_count
    
    def generate_prerequisites(self, module_number: int) -> List[str]:
        """Generate prerequisites based on module progression."""
        if module_number == 0:
            return ["Basic technical knowledge", "Familiarity with web technologies"]
        elif module_number == 1:
            return [f"Completion of Module 0: Fundamentals"]
        else:
            return [f"Completion of Module {module_number - 1}", "Understanding of previous module concepts"]
    
    def build_module_descriptions(self) -> List[Dict]:
        """Build comprehensive module descriptions."""
        print(f"[INFO] Analyzing lessons in {self.track_name} track...")
        
        # Load all lessons
        lessons = []
        for lesson_file in sorted(self.lessons_path.glob("*.md")):
            lesson_info = self.extract_lesson_info(lesson_file)
            if lesson_info:
                lessons.append(lesson_info)
        
        print(f"[INFO] Loaded {len(lessons)} lessons")
        
        # Group lessons by module
        modules = self.group_lessons_by_module(lessons)
        print(f"[INFO] Organized into {len(modules)} modules")
        
        # Generate module descriptions
        module_descriptions = []
        for module_number in sorted(modules.keys()):
            module_lessons = modules[module_number]
            
            title, subtitle = self.generate_module_title(module_number, module_lessons)
            key_topics = self.extract_key_topics(module_lessons)
            skills_gained = self.generate_skills_gained(module_number, module_lessons)
            duration, lab_count = self.calculate_duration_and_labs(module_lessons)
            prerequisites = self.generate_prerequisites(module_number)
            
            module_desc = {
                "id": f"{self.track_name}-module-{module_number}",
                "title": title,
                "subtitle": subtitle,
                "duration": duration,
                "lessons": len(module_lessons),
                "labs": lab_count,
                "prerequisites": prerequisites,
                "keyTopics": key_topics,
                "skillsGained": skills_gained
            }
            
            module_descriptions.append(module_desc)
            print(f"[SUCCESS] Module {module_number}: {title} ({len(module_lessons)} lessons)")
        
        return module_descriptions
    
    def save_module_descriptions(self, descriptions: List[Dict]) -> bool:
        """Save module descriptions to JSON file."""
        try:
            # Create output directory if it doesn't exist
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Save JSON file
            output_file = self.output_path / "module.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(descriptions, f, indent=2, ensure_ascii=False)
            
            print(f"[SUCCESS] Saved module descriptions to: {output_file}")
            print(f"[INFO] Total modules created: {len(descriptions)}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error saving module descriptions: {e}")
            return False
    
    def run(self) -> bool:
        """Run the complete module description building process."""
        print(f"[START] Module Description Builder for '{self.track_name}' track")
        print("=" * 60)
        
        # Validate track
        if not self.validate_track():
            return False
        
        # Build descriptions
        descriptions = self.build_module_descriptions()
        if not descriptions:
            print("[ERROR] No module descriptions generated")
            return False
        
        # Save to file
        if not self.save_module_descriptions(descriptions):
            return False
        
        print("=" * 60)
        print(f"[COMPLETE] Successfully created module descriptions for {self.track_name} track!")
        
        # Display summary
        config = self.track_configs.get(self.track_name, {})
        print(f"[INFO] Track: {config.get('theme', self.track_name.title())}")
        print(f"[INFO] Focus: {config.get('focus', 'Technical skills development')}")
        print(f"[INFO] Modules: {len(descriptions)}")
        print(f"[INFO] Output: {self.output_path / 'module.json'}")
        
        return True

def get_available_tracks() -> List[str]:
    """Dynamically discover available tracks from the data directory."""
    data_dir = Path(__file__).parent.parent / "src" / "data"
    tracks = []
    
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != 'src':
                # Check if the directory has lessons
                lessons_dir = item / "lessons"
                if lessons_dir.exists() and any(lessons_dir.glob("*.md")):
                    tracks.append(item.name)
    
    return sorted(tracks)

def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        available_tracks = get_available_tracks()
        print("Usage: python build_module_descriptions.py <track_folder_name>")
        print("\nAvailable tracks:")
        for track in available_tracks:
            builder = ModuleDescriptionBuilder(track)
            config = builder.track_configs.get(track, {'theme': track.title()})
            print(f"  {track:<16} - {config.get('theme', track.title())}")
        sys.exit(1)
    
    track_name = sys.argv[1].lower()
    
    # Validate track name
    valid_tracks = get_available_tracks()
    if track_name not in valid_tracks:
        print(f"[ERROR] '{track_name}' is not a valid track name")
        print(f"Valid tracks: {', '.join(valid_tracks)}")
        sys.exit(1)
    
    # Build module descriptions
    builder = ModuleDescriptionBuilder(track_name)
    success = builder.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()