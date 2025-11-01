#!/usr/bin/env python3
"""
Correct DE Lesson Resequencing Script
Reorders DE lesson files according to the actual Content Structure.md pedagogical flow (M01-M07)
"""

import os
import shutil
from pathlib import Path

# Base path for lessons
LESSONS_PATH = Path("src/data/de/lessons")
BACKUP_PATH = Path("backup-de-lessons-correct")

def create_backup():
    """Create backup of current DE lessons before renaming"""
    if BACKUP_PATH.exists():
        shutil.rmtree(BACKUP_PATH)
    
    BACKUP_PATH.mkdir(parents=True)
    
    # Copy all DE lesson files to backup
    source_dir = LESSONS_PATH
    if source_dir.exists():
        for lesson_file in source_dir.glob("*.md"):
            shutil.copy2(lesson_file, BACKUP_PATH / lesson_file.name)
    
    print(f"Backup created at: {BACKUP_PATH.absolute()}")

def categorize_lessons():
    """Categorize all lessons by their topic patterns"""
    lessons_by_category = {}
    
    for lesson_file in LESSONS_PATH.glob("*.md"):
        filename = lesson_file.name
        
        # Extract category from filename
        if filename.startswith("Additional-Technical-Skills"):
            category = "Additional-Technical-Skills"
        elif filename.startswith("CICD-Deployment-Practices"):
            category = "CICD-Deployment-Practices"
        elif filename.startswith("Data-and-Database"):
            category = "Data-and-Database"
        elif filename.startswith("Data-Database-Fundamentals"):
            category = "Data-Database-Fundamentals"
        elif filename.startswith("Data-Governance-Quality"):
            category = "Data-Governance-Quality"
        elif filename.startswith("Data-Modeling"):
            category = "Data-Modeling"
        elif filename.startswith("Data-Transformation-with"):
            category = "Data-Transformation-with"
        elif filename.startswith("Data-Warehousing-Principles"):
            category = "Data-Warehousing-Principles"
        elif filename.startswith("ETLELT-Design-Best"):
            category = "ETLELT-Design-Best"
        elif filename.startswith("Emerging-Topics-Advanced"):
            category = "Emerging-Topics-Advanced"
        elif filename.startswith("Monitoring-Observability"):
            category = "Monitoring-Observability"
        elif filename.startswith("Orchestration-Scheduling-Tools"):
            category = "Orchestration-Scheduling-Tools"
        elif filename.startswith("Performance-Optimization-Troubleshooting"):
            category = "Performance-Optimization-Troubleshooting"
        elif filename.startswith("Reporting-BI-Concepts"):
            category = "Reporting-BI-Concepts"
        elif filename.startswith("Snowflake-Security-Access"):
            category = "Snowflake-Security-Access"
        elif filename.startswith("Snowflake-Specific-Knowledge"):
            category = "Snowflake-Specific-Knowledge"
        elif filename.startswith("Soft-Skills-Professional"):
            category = "Soft-Skills-Professional"
        elif filename.startswith("SQL-ELT-Concepts"):
            category = "SQL-ELT-Concepts"
        elif filename.startswith("UnixLinux-File-Handling"):
            category = "UnixLinux-File-Handling"
        elif filename.startswith("Version-Control-Team"):
            category = "Version-Control-Team"
        elif filename.startswith("Business-Domain-Knowledge"):
            category = "Business-Domain-Knowledge"
        else:
            category = "Uncategorized"
        
        if category not in lessons_by_category:
            lessons_by_category[category] = []
        lessons_by_category[category].append(filename)
    
    return lessons_by_category

def resequence_lessons():
    """Resequence all DE lessons according to actual module structure (M01-M07)"""
    print("Resequencing DE lessons according to actual Content Structure...")
    
    lessons_by_category = categorize_lessons()
    
    # Create temp directory
    temp_dir = LESSONS_PATH.parent / "temp_de_lessons_correct"
    temp_dir.mkdir(exist_ok=True)
    
    lesson_counter = {}
    total_renamed = 0
    
    # Module 1: Data Engineering Foundations
    # - What Does a Data Engineer Do?
    # - Core Concepts: The Data Lifecycle, Data Formats (JSON, CSV, Parquet)  
    # - Essential Toolkit: Linux/Shell Scripting, Git & Version Control
    module = "M01"
    lesson_counter[module] = 1
    categories = ["Data-and-Database", "Data-Database-Fundamentals", "UnixLinux-File-Handling", "Version-Control-Team", "Additional-Technical-Skills"]
    
    for category in categories:
        if category in lessons_by_category:
            for old_filename in sorted(lessons_by_category[category]):
                # Clean up the filename - extract the core topic after the double dash
                topic_part = old_filename.split('--', 1)[1] if '--' in old_filename else old_filename.replace(category + '-', '')
                topic_part = topic_part.replace('.md', '')
                new_filename = f"{module}-L{lesson_counter[module]:03d}-{topic_part}.md"
                
                old_path = LESSONS_PATH / old_filename
                temp_path = temp_dir / new_filename
                
                if old_path.exists():
                    shutil.move(str(old_path), str(temp_path))
                    print(f"  {old_filename} -> {new_filename}")
                    lesson_counter[module] += 1
                    total_renamed += 1
    
    # Module 2: SQL & Data Modeling
    # - SQL Fundamentals, Advanced SQL, Database Deep Dive, Data Modeling
    module = "M02"
    lesson_counter[module] = 1
    categories = ["SQL-ELT-Concepts", "Data-Modeling", "Data-Warehousing-Principles"]
    
    for category in categories:
        if category in lessons_by_category:
            for old_filename in sorted(lessons_by_category[category]):
                topic_part = old_filename.split('--', 1)[1] if '--' in old_filename else old_filename.replace(category + '-', '')
                topic_part = topic_part.replace('.md', '')
                new_filename = f"{module}-L{lesson_counter[module]:03d}-{topic_part}.md"
                
                old_path = LESSONS_PATH / old_filename
                temp_path = temp_dir / new_filename
                
                if old_path.exists():
                    shutil.move(str(old_path), str(temp_path))
                    print(f"  {old_filename} -> {new_filename}")
                    lesson_counter[module] += 1
                    total_renamed += 1
    
    # Module 3: Python for Data Engineering
    # - Python Essentials, Key Libraries, Connecting to Databases, Building Data Pipelines
    module = "M03"
    lesson_counter[module] = 1
    # Python-related lessons would be in Additional-Technical-Skills or could be separate
    # For now, we'll skip this module as the current lessons don't have clear Python categorization
    
    # Module 4: The Modern Warehouse & Transformation
    # - Intro to Data Warehousing (BigQuery, Snowflake, Redshift)
    # - The ELT Paradigm, Data Transformation with dbt
    module = "M04"
    lesson_counter[module] = 1
    categories = ["Snowflake-Specific-Knowledge", "Snowflake-Security-Access", "Data-Transformation-with", "ETLELT-Design-Best"]
    
    for category in categories:
        if category in lessons_by_category:
            for old_filename in sorted(lessons_by_category[category]):
                topic_part = old_filename.split('--', 1)[1] if '--' in old_filename else old_filename.replace(category + '-', '')
                topic_part = topic_part.replace('.md', '')
                new_filename = f"{module}-L{lesson_counter[module]:03d}-{topic_part}.md"
                
                old_path = LESSONS_PATH / old_filename
                temp_path = temp_dir / new_filename
                
                if old_path.exists():
                    shutil.move(str(old_path), str(temp_path))
                    print(f"  {old_filename} -> {new_filename}")
                    lesson_counter[module] += 1
                    total_renamed += 1
    
    # Module 5: Batch Processing & Cloud Platforms
    # - Cloud Fundamentals, Big Data Concepts, Spark, Data Lakes
    module = "M05"
    lesson_counter[module] = 1
    categories = ["Performance-Optimization-Troubleshooting"]
    
    for category in categories:
        if category in lessons_by_category:
            for old_filename in sorted(lessons_by_category[category]):
                topic_part = old_filename.split('--', 1)[1] if '--' in old_filename else old_filename.replace(category + '-', '')
                topic_part = topic_part.replace('.md', '')
                new_filename = f"{module}-L{lesson_counter[module]:03d}-{topic_part}.md"
                
                old_path = LESSONS_PATH / old_filename
                temp_path = temp_dir / new_filename
                
                if old_path.exists():
                    shutil.move(str(old_path), str(temp_path))
                    print(f"  {old_filename} -> {new_filename}")
                    lesson_counter[module] += 1
                    total_renamed += 1
    
    # Module 6: Orchestration & DataOps
    # - Workflow Orchestration, Data Quality, CI/CD, Monitoring & Observability
    module = "M06"
    lesson_counter[module] = 1
    categories = ["Orchestration-Scheduling-Tools", "CICD-Deployment-Practices", "Monitoring-Observability", "Data-Governance-Quality"]
    
    for category in categories:
        if category in lessons_by_category:
            for old_filename in sorted(lessons_by_category[category]):
                topic_part = old_filename.split('--', 1)[1] if '--' in old_filename else old_filename.replace(category + '-', '')
                topic_part = topic_part.replace('.md', '')
                new_filename = f"{module}-L{lesson_counter[module]:03d}-{topic_part}.md"
                
                old_path = LESSONS_PATH / old_filename
                temp_path = temp_dir / new_filename
                
                if old_path.exists():
                    shutil.move(str(old_path), str(temp_path))
                    print(f"  {old_filename} -> {new_filename}")
                    lesson_counter[module] += 1
                    total_renamed += 1
    
    # Module 7: Real-Time Data & Streaming
    # - Batch vs. Streaming, Messaging Queues, Kafka, Stream Processing
    module = "M07"
    lesson_counter[module] = 1
    categories = ["Emerging-Topics-Advanced", "Reporting-BI-Concepts", "Business-Domain-Knowledge", "Soft-Skills-Professional"]
    
    for category in categories:
        if category in lessons_by_category:
            for old_filename in sorted(lessons_by_category[category]):
                topic_part = old_filename.split('--', 1)[1] if '--' in old_filename else old_filename.replace(category + '-', '')
                topic_part = topic_part.replace('.md', '')
                new_filename = f"{module}-L{lesson_counter[module]:03d}-{topic_part}.md"
                
                old_path = LESSONS_PATH / old_filename
                temp_path = temp_dir / new_filename
                
                if old_path.exists():
                    shutil.move(str(old_path), str(temp_path))
                    print(f"  {old_filename} -> {new_filename}")
                    lesson_counter[module] += 1
                    total_renamed += 1
    
    # Move files back to original directory
    for temp_file in temp_dir.glob("*.md"):
        final_path = LESSONS_PATH / temp_file.name
        shutil.move(str(temp_file), str(final_path))
    
    # Clean up temp directory
    temp_dir.rmdir()
    
    print(f"DE resequencing complete: {total_renamed} files renamed")
    
    # Print summary by module
    print("\nModule Summary:")
    for module in ["M01", "M02", "M03", "M04", "M05", "M06", "M07"]:
        if module in lesson_counter:
            count = lesson_counter[module] - 1
            print(f"  {module}: {count} lessons")

def main():
    """Main execution function"""
    print("Correct DE Lesson Resequencing Script (M01-M07)")
    print("=" * 60)
    
    # Create backup
    create_backup()
    
    # Resequence lessons
    resequence_lessons()
    
    print(f"\nBackup available at: {BACKUP_PATH.absolute()}")

if __name__ == "__main__":
    main()