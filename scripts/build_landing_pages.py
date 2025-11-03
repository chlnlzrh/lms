#!/usr/bin/env python3
"""
Landing Page Data Aggregator
Generates comprehensive JSON data for Book of Knowledge and Learning Path landing pages
by aggregating data from all track module.json files.

Usage:
    python build_landing_pages.py
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

class LandingPageAggregator:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent / "src" / "data"
        self.output_path = Path(__file__).parent.parent / "src" / "data"
        
        # Track configurations (from build_module_descriptions.py)
        self.track_configs = {
            # Book of Knowledge tracks
            'ai': {
                'theme': 'Artificial Intelligence',
                'color': 'violet',
                'icon': 'Bot',
                'focus': 'machine learning, automation, and intelligent systems',
                'difficulty': 'Intermediate',
                'featured': 'popular'
            },
            'de': {
                'theme': 'Data Engineering',
                'color': 'slate',
                'icon': 'Database',
                'focus': 'data pipelines, processing, and infrastructure',
                'difficulty': 'Advanced',
                'featured': 'technical'
            },
            'saas': {
                'theme': 'SaaS Development',
                'color': 'purple',
                'icon': 'Cloud',
                'focus': 'cloud applications, scalability, and service delivery',
                'difficulty': 'Intermediate',
                'featured': 'newest'
            },
            'sfdc': {
                'theme': 'Salesforce',
                'color': 'blue',
                'icon': 'Zap',
                'focus': 'CRM, platform development, and customization',
                'difficulty': 'Beginner',
                'featured': 'quickstart'
            },
            'snowflake_tune': {
                'theme': 'Snowflake Tuning',
                'color': 'cyan',
                'icon': 'Snowflake',
                'focus': 'performance optimization, tuning, and scaling',
                'difficulty': 'Advanced',
                'featured': 'specialist'
            },
            'workato': {
                'theme': 'Workato',
                'color': 'teal',
                'icon': 'Link',
                'focus': 'API integration, workflows, and connectivity',
                'difficulty': 'Intermediate',
                'featured': 'integration'
            },
            # Learning Path tracks
            'ba': {
                'theme': 'Business Analyst',
                'color': 'orange',
                'icon': 'BarChart3',
                'focus': 'business analysis skills and requirements gathering expertise',
                'difficulty': 'Beginner',
                'careerLevel': 'mid'
            },
            'data_engineer': {
                'theme': 'Data Engineer',
                'color': 'green',
                'icon': 'Database',
                'focus': 'data engineering principles, pipelines, and infrastructure',
                'difficulty': 'Advanced',
                'careerLevel': 'senior'
            },
            'data_gov': {
                'theme': 'Data Governance',
                'color': 'emerald',
                'icon': 'Shield',
                'focus': 'data governance frameworks and data quality compliance',
                'difficulty': 'Intermediate',
                'careerLevel': 'mid'
            },
            'devops_engineer': {
                'theme': 'DevOps Engineer',
                'color': 'slate',
                'icon': 'Settings',
                'focus': 'DevOps practices, CI/CD, and infrastructure automation',
                'difficulty': 'Advanced',
                'careerLevel': 'senior'
            },
            'finance': {
                'theme': 'Finance',
                'color': 'green',
                'icon': 'DollarSign',
                'focus': 'technology solutions for financial processes and systems',
                'difficulty': 'Beginner',
                'careerLevel': 'entry'
            },
            'hr': {
                'theme': 'Human Resources',
                'color': 'pink',
                'icon': 'Users',
                'focus': 'technology for human resources management and operations',
                'difficulty': 'Beginner',
                'careerLevel': 'entry'
            },
            'marketing': {
                'theme': 'Marketing',
                'color': 'red',
                'icon': 'Megaphone',
                'focus': 'marketing technology and digital marketing strategies',
                'difficulty': 'Beginner',
                'careerLevel': 'entry'
            },
            'mdm': {
                'theme': 'Master Data Management',
                'color': 'indigo',
                'icon': 'Archive',
                'focus': 'data management strategies, governance, and quality assurance',
                'difficulty': 'Advanced',
                'careerLevel': 'senior'
            },
            'pm': {
                'theme': 'Project Manager',
                'color': 'yellow',
                'icon': 'Briefcase',
                'focus': 'project management skills with modern methodologies and tools',
                'difficulty': 'Intermediate',
                'careerLevel': 'mid'
            },
            'qa': {
                'theme': 'Quality Assurance',
                'color': 'green',
                'icon': 'CheckCircle',
                'focus': 'quality assurance practices, testing frameworks, and automation',
                'difficulty': 'Intermediate',
                'careerLevel': 'mid'
            },
            'rpa': {
                'theme': 'Robotic Process Automation',
                'color': 'orange',
                'icon': 'Bot',
                'focus': 'robotic process automation solutions for business efficiency',
                'difficulty': 'Intermediate',
                'careerLevel': 'mid'
            },
            'sales': {
                'theme': 'Sales',
                'color': 'red',
                'icon': 'TrendingUp',
                'focus': 'technology and analytics for sales processes and customer management',
                'difficulty': 'Beginner',
                'careerLevel': 'entry'
            },
            'sfdc_engineer': {
                'theme': 'Salesforce Engineer',
                'color': 'blue',
                'icon': 'Zap',
                'focus': 'advanced Salesforce solutions and platform integrations',
                'difficulty': 'Advanced',
                'careerLevel': 'senior'
            },
            'ta': {
                'theme': 'Talent Acquisition',
                'color': 'gray',
                'icon': 'FileSearch',
                'focus': 'talent acquisition strategies, recruitment technologies, and candidate assessment',
                'difficulty': 'Beginner',
                'careerLevel': 'entry'
            },
            'viz_engineer': {
                'theme': 'Visualization Engineer',
                'color': 'purple',
                'icon': 'PieChart',
                'focus': 'compelling data visualizations and business intelligence solutions',
                'difficulty': 'Advanced',
                'careerLevel': 'senior'
            },
            'workato_engineer': {
                'theme': 'Workato Engineer',
                'color': 'teal',
                'icon': 'Link',
                'focus': 'complex integration solutions using Workato platform',
                'difficulty': 'Advanced',
                'careerLevel': 'senior'
            }
        }
    
    def load_track_data(self, track_id: str) -> Optional[Dict]:
        """Load module.json data for a specific track."""
        module_file = self.base_path / track_id / "modules-descriptions" / "module.json"
        
        if not module_file.exists():
            print(f"[WARNING] Module file not found for track: {track_id}")
            return None
            
        try:
            with open(module_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"[ERROR] Failed to load data for track {track_id}: {e}")
            return None
    
    def get_all_tracks_by_type(self, track_type: str) -> List[Dict]:
        """Get all tracks of a specific type (Book of Knowledge or Learning Path)."""
        tracks = []
        
        for track_id in self.track_configs.keys():
            track_data = self.load_track_data(track_id)
            if track_data and track_data.get('track', {}).get('type') == track_type:
                # Merge with configuration data
                config = self.track_configs.get(track_id, {})
                track_info = track_data['track'].copy()
                track_info.update({
                    'config': config,
                    'href': f"/{track_type.lower().replace(' ', '-')}/{track_id}",
                    'modules_data': track_data.get('modules', [])
                })
                tracks.append(track_info)
        
        return tracks
    
    def calculate_aggregated_stats(self, tracks: List[Dict]) -> Dict:
        """Calculate aggregated statistics across multiple tracks."""
        total_lessons = sum(track.get('lessons', 0) for track in tracks)
        total_modules = sum(track.get('modules', 0) for track in tracks)
        total_duration = sum(track.get('estimatedDuration', 0) for track in tracks)
        total_reading_time = sum(track.get('readingTime', 0) for track in tracks)
        total_hands_on = sum(track.get('handsOnLessons', 0) for track in tracks)
        total_labs = sum(track.get('labs', 0) for track in tracks)
        
        # Calculate averages
        practical_ratios = [track.get('statistics', {}).get('practicalRatio', 0) for track in tracks if track.get('statistics')]
        avg_practical_ratio = statistics.mean(practical_ratios) if practical_ratios else 0
        
        durations = [track.get('statistics', {}).get('averageLessonDuration', 0) for track in tracks if track.get('statistics')]
        avg_lesson_duration = statistics.mean(durations) if durations else 0
        
        # Collect all lesson types
        all_lesson_types = {}
        for track in tracks:
            lesson_types = track.get('statistics', {}).get('lessonTypes', {})
            for lesson_type, count in lesson_types.items():
                all_lesson_types[lesson_type] = all_lesson_types.get(lesson_type, 0) + count
        
        # Extract unique technologies and topics
        all_topics = set()
        for track in tracks:
            for module in track.get('modules_data', []):
                topics = module.get('keyTopics', [])
                all_topics.update(topics)
        
        return {
            'totals': {
                'tracks': len(tracks),
                'modules': total_modules,
                'lessons': total_lessons,
                'estimatedHours': round(total_duration / 60, 1),
                'readingHours': round(total_reading_time / 60, 1),
                'handsOnLessons': total_hands_on,
                'labs': total_labs
            },
            'averages': {
                'lessonsPerTrack': round(total_lessons / len(tracks), 1) if tracks else 0,
                'hoursPerTrack': round((total_duration / 60) / len(tracks), 1) if tracks else 0,
                'practicalRatio': round(avg_practical_ratio, 1),
                'lessonDuration': round(avg_lesson_duration, 1)
            },
            'distributions': {
                'lessonTypes': all_lesson_types,
                'technologies': sorted(list(all_topics))[:15]  # Top 15 topics
            }
        }
    
    def extract_featured_content(self, tracks: List[Dict]) -> Dict:
        """Extract featured content based on track configurations."""
        featured = {
            'mostPopular': None,
            'newest': None,
            'quickStart': None,
            'advanced': None
        }
        
        for track in tracks:
            config = track.get('config', {})
            track_featured = config.get('featured', '')
            
            if track_featured == 'popular':
                featured['mostPopular'] = track['id']
            elif track_featured == 'newest':
                featured['newest'] = track['id']
            elif track_featured == 'quickstart':
                featured['quickStart'] = track['id']
            elif config.get('difficulty') == 'Advanced' and not featured['advanced']:
                featured['advanced'] = track['id']
        
        return featured
    
    def build_track_cards(self, tracks: List[Dict]) -> List[Dict]:
        """Build track card data for the landing page."""
        cards = []
        
        for track in tracks:
            config = track.get('config', {})
            
            # Extract key topics from modules
            all_topics = set()
            for module in track.get('modules_data', []):
                topics = module.get('keyTopics', [])
                all_topics.update(topics)
            
            key_topics = sorted(list(all_topics))[:4]  # Top 4 topics
            
            card = {
                'id': track['id'],
                'title': track['title'],
                'description': track['description'],
                'icon': config.get('icon', 'BookOpen'),
                'color': config.get('color', 'blue'),
                'href': track['href'],
                'difficulty': config.get('difficulty', 'Intermediate'),
                'stats': {
                    'modules': track.get('modules', 0),
                    'lessons': track.get('lessons', 0),
                    'duration': track.get('duration', '0 weeks'),
                    'estimatedHours': round(track.get('estimatedDuration', 0) / 60, 1),
                    'practicalRatio': track.get('statistics', {}).get('practicalRatio', 0),
                    'handsOnLessons': track.get('handsOnLessons', 0)
                },
                'keyTopics': key_topics,
                'focus': config.get('focus', ''),
                'status': 'active'
            }
            
            # Add career level for Learning Paths
            if 'careerLevel' in config:
                card['careerLevel'] = config['careerLevel']
            
            cards.append(card)
        
        return cards
    
    def generate_filters(self, tracks: List[Dict]) -> Dict:
        """Generate filter options based on track data."""
        difficulties = set()
        technologies = set()
        durations = set()
        career_levels = set()
        
        for track in tracks:
            config = track.get('config', {})
            
            # Difficulty levels
            difficulties.add(config.get('difficulty', 'Intermediate'))
            
            # Duration categories
            estimated_hours = track.get('estimatedDuration', 0) / 60
            if estimated_hours < 50:
                durations.add('< 50 hours')
            elif estimated_hours < 100:
                durations.add('50-100 hours')
            else:
                durations.add('100+ hours')
            
            # Career levels (Learning Paths only)
            if 'careerLevel' in config:
                career_levels.add(config['careerLevel'])
            
            # Technologies from key topics
            for module in track.get('modules_data', []):
                topics = module.get('keyTopics', [])
                technologies.update(topics)
        
        filters = {
            'byDifficulty': sorted(list(difficulties)),
            'byDuration': sorted(list(durations)),
            'byTechnology': sorted(list(technologies))[:12]  # Top 12 technologies
        }
        
        if career_levels:
            filters['byCareerLevel'] = sorted(list(career_levels))
        
        return filters
    
    def build_book_of_knowledge_landing(self) -> Dict:
        """Build comprehensive Book of Knowledge landing page data."""
        print("[INFO] Building Book of Knowledge landing page data...")
        
        tracks = self.get_all_tracks_by_type("Book of Knowledge")
        
        if not tracks:
            print("[ERROR] No Book of Knowledge tracks found!")
            return {}
        
        aggregated_stats = self.calculate_aggregated_stats(tracks)
        track_cards = self.build_track_cards(tracks)
        featured_content = self.extract_featured_content(tracks)
        filters = self.generate_filters(tracks)
        
        landing_data = {
            'pageInfo': {
                'title': 'Book of Knowledge',
                'description': 'Master foundational technologies and cutting-edge tools',
                'subtitle': 'Comprehensive technical knowledge across key technology domains',
                'category': 'foundation',
                'lastUpdated': self._get_last_updated()
            },
            'overview': aggregated_stats,
            'tracks': track_cards,
            'featuredContent': featured_content,
            'filters': filters,
            'metadata': {
                'totalTracks': len(tracks),
                'dataSource': 'aggregated from module.json files',
                'generatedAt': self._get_timestamp()
            }
        }
        
        print(f"[SUCCESS] Built Book of Knowledge data: {len(tracks)} tracks, {aggregated_stats['totals']['lessons']} lessons")
        return landing_data
    
    def build_learning_path_landing(self) -> Dict:
        """Build comprehensive Learning Path landing page data."""
        print("[INFO] Building Learning Path landing page data...")
        
        tracks = self.get_all_tracks_by_type("Learning Path")
        
        if not tracks:
            print("[ERROR] No Learning Path tracks found!")
            return {}
        
        aggregated_stats = self.calculate_aggregated_stats(tracks)
        track_cards = self.build_track_cards(tracks)
        filters = self.generate_filters(tracks)
        
        # Group tracks by career level
        career_groups = {}
        for track in track_cards:
            career_level = track.get('careerLevel', 'mid')
            if career_level not in career_groups:
                career_groups[career_level] = []
            career_groups[career_level].append(track)
        
        landing_data = {
            'pageInfo': {
                'title': 'Learning Paths',
                'description': 'Role-specific career development and skill building',
                'subtitle': 'Structured learning journeys for professional growth',
                'category': 'role-specific',
                'lastUpdated': self._get_last_updated()
            },
            'overview': aggregated_stats,
            'tracks': track_cards,
            'careerGroups': career_groups,
            'filters': filters,
            'metadata': {
                'totalTracks': len(tracks),
                'dataSource': 'aggregated from module.json files',
                'generatedAt': self._get_timestamp()
            }
        }
        
        print(f"[SUCCESS] Built Learning Path data: {len(tracks)} tracks, {aggregated_stats['totals']['lessons']} lessons")
        return landing_data
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_last_updated(self) -> str:
        """Get last updated date in readable format."""
        from datetime import datetime
        return datetime.now().strftime("%B %d, %Y")
    
    def save_landing_data(self, book_data: Dict, learning_data: Dict) -> bool:
        """Save both landing page data files."""
        try:
            # Save Book of Knowledge data
            book_file = self.output_path / "book-of-knowledge-landing.json"
            with open(book_file, 'w', encoding='utf-8') as f:
                json.dump(book_data, f, indent=2, ensure_ascii=False)
            print(f"[SUCCESS] Saved Book of Knowledge landing data: {book_file}")
            
            # Save Learning Path data
            learning_file = self.output_path / "learning-path-landing.json"
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)
            print(f"[SUCCESS] Saved Learning Path landing data: {learning_file}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save landing page data: {e}")
            return False
    
    def run(self) -> bool:
        """Run the complete landing page data generation process."""
        print("=" * 70)
        print("LMS Landing Page Data Aggregator")
        print("=" * 70)
        
        try:
            # Build both landing page datasets
            book_data = self.build_book_of_knowledge_landing()
            learning_data = self.build_learning_path_landing()
            
            if not book_data or not learning_data:
                print("[ERROR] Failed to generate landing page data")
                return False
            
            # Save the data
            if not self.save_landing_data(book_data, learning_data):
                return False
            
            print("=" * 70)
            print("Landing Page Data Generation Complete!")
            print(f"Book of Knowledge: {book_data['metadata']['totalTracks']} tracks")
            print(f"Learning Paths: {learning_data['metadata']['totalTracks']} tracks")
            print("=" * 70)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Landing page generation failed: {e}")
            return False

def main():
    """Main entry point for the script."""
    aggregator = LandingPageAggregator()
    success = aggregator.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()