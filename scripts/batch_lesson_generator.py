#!/usr/bin/env python3
"""
Batch Lesson Generator for Claude Haiku 4.5
Generates all 563 missing lessons using the TSV manifest and places them in correct directories.
"""

import os
import sys
import json
import time
import csv
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lesson_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LessonSpec:
    """Data class for lesson specifications from TSV"""
    lesson_id: str
    title: str
    category: str
    module_code: str
    module_name: str
    sequence: str
    difficulty: str
    description: str
    target_audience: str
    prerequisites: str
    learning_objectives: str
    duration: str
    lesson_type: str
    content_format: str
    practical_components: str
    assessment_type: str
    tools_required: str
    industry_context: str
    skill_level: str
    cognitive_load: str
    interaction_style: str
    output_artifacts: str
    file_path: str
    filename: str
    tone: str
    code_complexity: str
    explanation_depth: str

class ClaudeBatchLessonGenerator:
    """Batch lesson generator using Claude Haiku 4.5 API"""
    
    def __init__(self, api_key: str, base_path: str = "C:/ai/training/lms-platform"):
        self.api_key = api_key
        self.base_path = Path(base_path)
        self.tsv_file = self.base_path / "final_layout" / "COMPLETE_LLM_LESSON_MANIFEST.tsv"
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-haiku-4-5-20251001"
        self.max_concurrent = 5  # Rate limiting
        self.retry_count = 3
        self.retry_delay = 2  # seconds
        
        # Statistics
        self.stats = {
            'total_lessons': 0,
            'generated': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
    
    def load_lesson_manifest(self) -> List[LessonSpec]:
        """Load lesson specifications from TSV file"""
        lessons = []
        
        try:
            with open(self.tsv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    lesson = LessonSpec(**row)
                    lessons.append(lesson)
                    
            logger.info(f"Loaded {len(lessons)} lesson specifications from TSV")
            self.stats['total_lessons'] = len(lessons)
            return lessons
            
        except Exception as e:
            logger.error(f"Failed to load TSV manifest: {e}")
            raise
    
    def create_lesson_prompt(self, lesson: LessonSpec) -> str:
        """Generate the lesson prompt for Claude"""
        
        prompt = f"""You are an expert curriculum designer creating a lesson for an enterprise AI training platform.

**Lesson Details:**
- ID: {lesson.lesson_id}
- Title: {lesson.title}
- Target Audience: {lesson.target_audience}
- Duration: {lesson.duration}
- Difficulty: {lesson.difficulty} ({'Foundational' if lesson.difficulty == 'F' else 'Intermediate' if lesson.difficulty == 'I' else 'Advanced' if lesson.difficulty == 'A' else 'Expert' if lesson.difficulty == 'E' else 'Leadership' if lesson.difficulty == 'L' else 'Practical'})

**Learning Objectives:**
{lesson.learning_objectives}

**Content Requirements:**
- Format: {lesson.content_format}
- Tone: {lesson.tone}
- Depth: {lesson.explanation_depth}
- Code Complexity: {lesson.code_complexity}
- Practical Components: {lesson.practical_components}
- Industry Context: {lesson.industry_context}
- Tools Required: {lesson.tools_required}

**Prerequisites:**
{lesson.prerequisites}

**Expected Deliverables:**
{lesson.output_artifacts}

---

### INSTRUCTIONS

Generate a **complete, polished lesson** following this structure:

## Learning Objectives & Prerequisites
## Core Concepts  
## Technical Implementation
## Practical Examples & Exercises
## Step-by-Step Implementation Guide
## Best Practices & Considerations
## Common Pitfalls & Troubleshooting
## Assessment & Validation
## Real-World Applications
## Additional Resources & Next Steps

**Content Guidelines:**
- **Length:** {'~300-400 lines' if lesson.difficulty == 'F' else '~400-500 lines' if lesson.difficulty == 'I' else '500+ lines'}
- **Technical Depth:** Include working code examples demonstrating {lesson.code_complexity} patterns
- **Business Context:** Connect technical concepts to measurable business outcomes
- **Practical Focus:** Every concept should include hands-on application
- **Industry Relevance:** Anchor examples in {lesson.industry_context} scenarios

**Formatting Requirements:**
- Use `##` for major sections, `###` for subsections
- Include fenced code blocks with language tags
- Use tables for comparisons and structured data
- Include callouts (✅, ⚠️, 💡) for emphasis
- Maintain {lesson.tone} tone throughout

**Quality Standards:**
- Begin with `# {lesson.lesson_id}: {lesson.title}`
- Include all 10 sections in order
- Provide {lesson.explanation_depth} coverage of topics
- Enable learners to complete {lesson.assessment_type}
- Support {lesson.interaction_style} learning approach

Generate the complete lesson content as clean Markdown, ready for enterprise learning platform deployment."""

        return prompt
    
    async def generate_lesson_content(self, session: aiohttp.ClientSession, lesson: LessonSpec) -> Tuple[bool, str, Optional[str]]:
        """Generate lesson content using Claude API"""
        
        prompt = self.create_lesson_prompt(lesson)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        for attempt in range(self.retry_count):
            try:
                async with session.post(self.api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('content', [{}])[0].get('text', '')
                        return True, content, None
                    elif response.status == 429:  # Rate limit
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited for {lesson.lesson_id}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        return False, "", f"API error {response.status}: {error_text}"
                        
            except Exception as e:
                if attempt == self.retry_count - 1:
                    return False, "", f"Request failed after {self.retry_count} attempts: {str(e)}"
                await asyncio.sleep(self.retry_delay)
        
        return False, "", "Max retry attempts exceeded"
    
    def ensure_directory_exists(self, file_path: str) -> bool:
        """Ensure the target directory exists"""
        try:
            directory = Path(file_path).parent
            directory.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory for {file_path}: {e}")
            return False
    
    def save_lesson_file(self, lesson: LessonSpec, content: str) -> bool:
        """Save the generated lesson content to the correct file location"""
        
        try:
            # Ensure directory exists
            if not self.ensure_directory_exists(lesson.file_path):
                return False
            
            # Construct full file path
            full_path = Path(lesson.file_path) / lesson.filename
            
            # Save content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Saved lesson: {lesson.lesson_id} -> {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save lesson {lesson.lesson_id}: {e}")
            return False
    
    def check_existing_lesson(self, lesson: LessonSpec) -> bool:
        """Check if lesson file already exists"""
        full_path = Path(lesson.file_path) / lesson.filename
        return full_path.exists()
    
    async def process_lesson_batch(self, session: aiohttp.ClientSession, lessons: List[LessonSpec], semaphore: asyncio.Semaphore) -> None:
        """Process a batch of lessons with rate limiting"""
        
        tasks = []
        for lesson in lessons:
            task = self.process_single_lesson(session, lesson, semaphore)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_single_lesson(self, session: aiohttp.ClientSession, lesson: LessonSpec, semaphore: asyncio.Semaphore) -> None:
        """Process a single lesson with semaphore for rate limiting"""
        
        async with semaphore:
            # Check if lesson already exists
            if self.check_existing_lesson(lesson):
                logger.info(f"Skipping existing lesson: {lesson.lesson_id}")
                self.stats['skipped'] += 1
                return
            
            # Generate content
            logger.info(f"Generating lesson: {lesson.lesson_id}")
            success, content, error = await self.generate_lesson_content(session, lesson)
            
            if success and content:
                # Save the lesson
                if self.save_lesson_file(lesson, content):
                    self.stats['generated'] += 1
                    logger.info(f"✅ Successfully generated: {lesson.lesson_id}")
                else:
                    self.stats['failed'] += 1
                    self.stats['errors'].append(f"Save failed: {lesson.lesson_id}")
            else:
                self.stats['failed'] += 1
                error_msg = f"Generation failed: {lesson.lesson_id} - {error}"
                self.stats['errors'].append(error_msg)
                logger.error(error_msg)
            
            # Rate limiting delay
            await asyncio.sleep(0.5)
    
    async def generate_all_lessons(self, skip_existing: bool = True) -> Dict:
        """Generate all lessons from the manifest"""
        
        logger.info("Starting batch lesson generation...")
        
        # Load lesson specifications
        lessons = self.load_lesson_manifest()
        
        if not lessons:
            logger.error("No lessons loaded from manifest")
            return self.stats
        
        # Filter existing lessons if requested
        if skip_existing:
            original_count = len(lessons)
            lessons = [lesson for lesson in lessons if not self.check_existing_lesson(lesson)]
            skipped_count = original_count - len(lessons)
            logger.info(f"Skipping {skipped_count} existing lessons, processing {len(lessons)} new lessons")
        
        # Process lessons in batches with rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process in smaller batches to manage memory
            batch_size = 10
            for i in range(0, len(lessons), batch_size):
                batch = lessons[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(lessons) + batch_size - 1)//batch_size}")
                
                await self.process_lesson_batch(session, batch, semaphore)
                
                # Progress update
                progress = ((i + batch_size) / len(lessons)) * 100
                logger.info(f"Progress: {progress:.1f}% - Generated: {self.stats['generated']}, Failed: {self.stats['failed']}")
        
        return self.stats
    
    def print_summary(self) -> None:
        """Print generation summary"""
        
        print("\n" + "="*60)
        print("LESSON GENERATION SUMMARY")
        print("="*60)
        print(f"Total Lessons in Manifest: {self.stats['total_lessons']}")
        print(f"Successfully Generated: {self.stats['generated']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped (existing): {self.stats['skipped']}")
        print(f"Success Rate: {(self.stats['generated']/(self.stats['generated']+self.stats['failed'])*100):.1f}%" if (self.stats['generated']+self.stats['failed']) > 0 else "N/A")
        
        if self.stats['errors']:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more errors")
        
        print("="*60)

def main():
    """Main entry point"""
    
    if len(sys.argv) != 2:
        print("Usage: python batch_lesson_generator.py <CLAUDE_API_KEY>")
        print("\nThis script will:")
        print("1. Load lesson specifications from COMPLETE_LLM_LESSON_MANIFEST.tsv")
        print("2. Generate lessons using Claude Haiku 4.5")
        print("3. Save lessons to correct directories with proper filenames")
        print("4. Skip existing lessons automatically")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    if not api_key or api_key.startswith("sk-"):
        print("ERROR: Please provide a valid Claude API key")
        print("The key should start with 'sk-ant-' for Anthropic Claude API")
        sys.exit(1)
    
    # Initialize generator
    generator = ClaudeBatchLessonGenerator(api_key)
    
    try:
        # Run the async generation process
        stats = asyncio.run(generator.generate_all_lessons(skip_existing=True))
        
        # Print summary
        generator.print_summary()
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            logger.warning("Some lessons failed to generate - check logs for details")
            sys.exit(1)
        else:
            logger.info("All lessons generated successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        generator.print_summary()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()