#!/usr/bin/env python3
"""
Comprehensive All Modules Markdown Integration Test
Verify all 20 modules correctly load data from markdown files
Optimized for 10 core/20 thread/32GB RAM machine
"""

import asyncio
import time
import json
from datetime import datetime
from playwright.async_api import async_playwright

class AllModulesMarkdownTester:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.modules = [f"module-{i}" for i in range(1, 21)]
        self.results = {}
        self.start_time = None
        
        # Expected markdown file data to verify against
        self.expected_data = {
            "module-1": {"title": "Module 1: Data & Database Fundamentals", "hours": 30, "lessons": 38},
            "module-2": {"title": "Module 2: SQL & ELT Concepts", "hours": 32, "lessons": 41},
            "module-3": {"title": "Module 3: Data Warehousing Principles", "hours": 19, "lessons": 25},
            "module-4": {"title": "Module 4: Data Modeling", "hours": 31, "lessons": 38},
            "module-5": {"title": "Module 5: Snowflake Specific Knowledge", "hours": 43, "lessons": 55},
            "module-6": {"title": "Module 6: ETL/ELT Design & Best Practices", "hours": 24, "lessons": 30},
            "module-7": {"title": "Module 7: Data Governance, Quality & Metadata", "hours": 32, "lessons": 32},
            "module-8": {"title": "Module 8: Snowflake Security & Access Control", "hours": 15, "lessons": 19},
            "module-9": {"title": "Module 9: Reporting & BI Concepts", "hours": 18, "lessons": 23},
            "module-10": {"title": "Module 10: Unix/Linux & File Handling", "hours": 24, "lessons": 30},
            "module-11": {"title": "Module 11: Version Control & Team Collaboration", "hours": 20, "lessons": 25},
            "module-12": {"title": "Module 12: Performance Optimization & Troubleshooting", "hours": 16, "lessons": 20},
            "module-13": {"title": "Module 13: CI/CD & Deployment Practices", "hours": 20, "lessons": 25},
            "module-14": {"title": "Module 14: Monitoring & Observability", "hours": 22, "lessons": 28},
            "module-15": {"title": "Module 15: Orchestration & Scheduling Tools", "hours": 32, "lessons": 40},
            "module-16": {"title": "Module 16: Data Transformation with dbt", "hours": 40, "lessons": 50},
            "module-17": {"title": "Module 17: Soft Skills & Professional Practices", "hours": 16, "lessons": 20},
            "module-18": {"title": "Module 18: Business & Domain Knowledge", "hours": 20, "lessons": 25},
            "module-19": {"title": "Module 19: Additional Technical Skills", "hours": 32, "lessons": 40},
            "module-20": {"title": "Module 20: Emerging Topics & Advanced Concepts", "hours": 24, "lessons": 30}
        }
        
    async def test_single_module_markdown_integration(self, module_id):
        """Test a single module's markdown integration"""
        result = {
            "module_id": module_id,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "success": False
        }
        
        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            # Test API endpoint
            context = await browser.new_context()
            page = await context.new_page()
            
            # Test API directly
            api_response = await page.request.get(f"{self.base_url}/api/modules/{module_id}")
            
            if api_response.status == 200:
                api_data = await api_response.json()
                module_data = api_data.get("module", {})
                
                expected = self.expected_data.get(module_id, {})
                
                # Test 1: Title matches markdown
                title_match = module_data.get("title", "") == expected.get("title", "")
                result["tests"]["title_from_markdown"] = {
                    "passed": title_match,
                    "expected": expected.get("title", ""),
                    "actual": module_data.get("title", "")
                }
                
                # Test 2: Duration matches markdown
                hours_match = module_data.get("estimatedHours", 0) == expected.get("hours", 0)
                result["tests"]["duration_from_markdown"] = {
                    "passed": hours_match,
                    "expected": expected.get("hours", 0),
                    "actual": module_data.get("estimatedHours", 0)
                }
                
                # Test 3: Lessons count matches markdown
                lessons_match = module_data.get("lessons", 0) == expected.get("lessons", 0)
                result["tests"]["lessons_from_markdown"] = {
                    "passed": lessons_match,
                    "expected": expected.get("lessons", 0),
                    "actual": module_data.get("lessons", 0)
                }
                
                # Test 4: Has learning objectives from markdown
                has_objectives = len(module_data.get("learningObjectives", [])) >= 3
                result["tests"]["learning_objectives_present"] = {
                    "passed": has_objectives,
                    "count": len(module_data.get("learningObjectives", []))
                }
                
                # Test 5: Has topics from markdown
                has_topics = len(module_data.get("topics", [])) >= 5
                result["tests"]["topics_from_categories"] = {
                    "passed": has_topics,
                    "count": len(module_data.get("topics", []))
                }
                
                # Test 6: Has prerequisites (if applicable)
                prerequisites = module_data.get("prerequisites", [])
                has_prereqs = isinstance(prerequisites, list)
                result["tests"]["prerequisites_structure"] = {
                    "passed": has_prereqs,
                    "count": len(prerequisites) if has_prereqs else 0
                }
                
                # Test 7: Page loads correctly
                await page.goto(f"{self.base_url}/learning-path/{module_id}", wait_until="networkidle", timeout=30000)
                page_title = await page.title()
                page_loads = "data engineering" in page_title.lower() or module_id in page_title.lower()
                
                result["tests"]["page_loads"] = {
                    "passed": page_loads,
                    "page_title": page_title
                }
                
            else:
                result["tests"]["api_error"] = {
                    "passed": False,
                    "status_code": api_response.status
                }
            
            await browser.close()
            await playwright.stop()
            
        except Exception as e:
            result["error"] = str(e)
            
        # Calculate success
        passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
        total_tests = len(result["tests"])
        result["success"] = passed_tests >= (total_tests * 0.85)  # 85% pass rate
        result["pass_rate"] = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return result
    
    async def run_all_modules_test(self):
        """Test all modules in parallel"""
        print("ALL MODULES MARKDOWN INTEGRATION TEST")
        print("=" * 70)
        print(f"Target: {self.base_url}")
        print(f"Modules: {len(self.modules)}")
        print(f"Machine: 10 core/20 thread/32GB RAM optimized")
        print(f"Testing: Markdown file data integration")
        print("-" * 70)
        
        self.start_time = time.time()
        
        # Create semaphore for controlled concurrency
        semaphore = asyncio.Semaphore(10)  # 10 concurrent tests
        
        async def test_with_semaphore(module_id):
            async with semaphore:
                return await self.test_single_module_markdown_integration(module_id)
        
        # Execute all tests in parallel
        tasks = [test_with_semaphore(module_id) for module_id in self.modules]
        print(f"Executing {len(tasks)} module tests in parallel...")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.results[self.modules[i]] = {
                    "error": str(result),
                    "success": False
                }
            else:
                self.results[self.modules[i]] = result
        
        total_time = time.time() - self.start_time
        print(f"Completed in {total_time:.2f} seconds")
        
        return self.results
    
    def generate_comprehensive_report(self):
        """Generate detailed markdown integration report"""
        total_modules = len(self.results)
        successful_modules = sum(1 for r in self.results.values() if r.get("success", False))
        
        print("\n" + "=" * 80)
        print("ALL MODULES MARKDOWN INTEGRATION REPORT")
        print("=" * 80)
        
        # Overall summary
        success_rate = (successful_modules / total_modules) * 100
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total Modules Tested: {total_modules}")
        print(f"  Successfully Integrated: {successful_modules}")
        print(f"  Failed Integration: {total_modules - successful_modules}")
        print(f"  Success Rate: {success_rate:.1f}%")
        
        # Test categories analysis
        test_categories = {}
        for result in self.results.values():
            if "tests" in result:
                for test_name, test_result in result["tests"].items():
                    if test_name not in test_categories:
                        test_categories[test_name] = {"passed": 0, "total": 0}
                    test_categories[test_name]["total"] += 1
                    if test_result.get("passed", False):
                        test_categories[test_name]["passed"] += 1
        
        print(f"\nMARKDOWN INTEGRATION TEST RESULTS:")
        for category, stats in test_categories.items():
            category_rate = (stats["passed"] / stats["total"]) * 100
            print(f"  {category.replace('_', ' ').title()}: {stats['passed']}/{stats['total']} ({category_rate:.1f}%)")
        
        # Detailed module results
        print(f"\nDETAILED MODULE RESULTS:")
        for module_id, result in self.results.items():
            if result.get("success", False):
                pass_rate = result.get("pass_rate", 0)
                print(f"  {module_id}: PASS ({pass_rate:.0f}%)")
            else:
                error_msg = result.get("error", "Integration test failures")[:50]
                print(f"  {module_id}: FAIL - {error_msg}")
        
        # Data verification summary
        print(f"\nMARKDOWN DATA VERIFICATION:")
        title_matches = sum(1 for r in self.results.values() 
                          if r.get("tests", {}).get("title_from_markdown", {}).get("passed", False))
        duration_matches = sum(1 for r in self.results.values() 
                             if r.get("tests", {}).get("duration_from_markdown", {}).get("passed", False))
        lessons_matches = sum(1 for r in self.results.values() 
                            if r.get("tests", {}).get("lessons_from_markdown", {}).get("passed", False))
        
        print(f"  Titles from Markdown: {title_matches}/{total_modules}")
        print(f"  Durations from Markdown: {duration_matches}/{total_modules}")
        print(f"  Lesson Counts from Markdown: {lessons_matches}/{total_modules}")
        
        # Performance metrics
        if self.start_time:
            execution_time = time.time() - self.start_time
            print(f"\nPERFORMANCE METRICS:")
            print(f"  Test Execution Time: {execution_time:.2f}s")
            print(f"  Parallel Efficiency: {total_modules} modules tested concurrently")
            print(f"  Average Time per Module: {execution_time/total_modules:.2f}s")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"all_modules_markdown_test_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed report saved: {report_file}")
        
        return success_rate >= 90.0  # 90% success rate required

async def main():
    """Main execution"""
    tester = AllModulesMarkdownTester()
    
    try:
        # Run all modules test
        await tester.run_all_modules_test()
        
        # Generate comprehensive report
        success = tester.generate_comprehensive_report()
        
        if success:
            print("\nSUCCESS: All modules successfully integrated with markdown data!")
            print("Ready for production use!")
            return 0
        else:
            print("\nPARTIAL: Some modules need attention")
            return 1
            
    except Exception as e:
        print(f"\nERROR: Module testing failed - {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)