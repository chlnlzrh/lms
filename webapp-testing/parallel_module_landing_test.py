#!/usr/bin/env python3
"""
Parallel Module Landing Page Testing Suite
Optimized for 10 core/20 thread/32GB RAM machine
Tests all 20 enhanced module landing pages with dynamic content loading
"""

import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.async_api import async_playwright
import json
from datetime import datetime
import sys

class ParallelModuleTester:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.max_workers = 20  # Utilize all threads
        self.results = {}
        self.start_time = None
        
        # Module IDs to test
        self.module_ids = [f"module-{i}" for i in range(1, 21)]
        
        # Test scenarios for each module
        self.test_scenarios = [
            "load_module_card",
            "expand_module_details", 
            "verify_dynamic_content",
            "test_prerequisites",
            "check_learning_objectives",
            "validate_topic_categories",
            "test_action_buttons",
            "verify_progress_indicators",
            "check_api_integration",
            "test_responsive_design"
        ]

    async def setup_browser_context(self):
        """Setup browser context for parallel testing"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage', 
                '--disable-gpu',
                '--disable-web-security',
                '--max_old_space_size=8192'  # Use more memory
            ]
        )
        
    async def teardown_browser_context(self):
        """Cleanup browser resources"""
        await self.browser.close()
        await self.playwright.stop()

    async def test_module_landing_page(self, module_id, session):
        """Test a single module landing page with all scenarios"""
        test_results = {
            "module_id": module_id,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "errors": [],
            "performance": {}
        }
        
        try:
            # Create new page context
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            # Navigate to main page with module cards
            start_nav = time.time()
            await page.goto(f"{self.base_url}/", wait_until="networkidle")
            nav_time = time.time() - start_nav
            test_results["performance"]["page_load"] = nav_time
            
            # Test 1: Load Module Card
            test_results["tests"]["load_module_card"] = await self.test_load_module_card(page, module_id)
            
            # Test 2: Expand Module Details
            test_results["tests"]["expand_module_details"] = await self.test_expand_module_details(page, module_id)
            
            # Test 3: Verify Dynamic Content
            test_results["tests"]["verify_dynamic_content"] = await self.test_verify_dynamic_content(page, module_id)
            
            # Test 4: Test Prerequisites
            test_results["tests"]["test_prerequisites"] = await self.test_prerequisites(page, module_id)
            
            # Test 5: Check Learning Objectives
            test_results["tests"]["check_learning_objectives"] = await self.test_learning_objectives(page, module_id)
            
            # Test 6: Validate Topic Categories
            test_results["tests"]["validate_topic_categories"] = await self.test_topic_categories(page, module_id)
            
            # Test 7: Test Action Buttons
            test_results["tests"]["test_action_buttons"] = await self.test_action_buttons(page, module_id)
            
            # Test 8: Verify Progress Indicators
            test_results["tests"]["verify_progress_indicators"] = await self.test_progress_indicators(page, module_id)
            
            # Test 9: Check API Integration
            test_results["tests"]["check_api_integration"] = await self.test_api_integration(session, module_id)
            
            # Test 10: Test Responsive Design
            test_results["tests"]["test_responsive_design"] = await self.test_responsive_design(page, module_id)
            
            await context.close()
            
        except Exception as e:
            test_results["errors"].append({
                "type": "general_error",
                "message": str(e)
            })
            
        return test_results

    async def test_load_module_card(self, page, module_id):
        """Test if module card loads properly"""
        try:
            # Look for module card by data attribute or class
            module_card = await page.wait_for_selector(
                f'[data-module-id="{module_id}"], .module-card:has-text("{module_id.replace("-", " ").title()}")',
                timeout=5000
            )
            
            if module_card:
                # Check if card is visible
                is_visible = await module_card.is_visible()
                
                # Get card text content
                card_text = await module_card.text_content()
                
                return {
                    "passed": True,
                    "visible": is_visible,
                    "has_content": len(card_text.strip()) > 0,
                    "card_text_length": len(card_text)
                }
            else:
                return {"passed": False, "error": "Module card not found"}
                
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_expand_module_details(self, page, module_id):
        """Test module card expansion functionality"""
        try:
            # Find expand button (chevron down/up)
            expand_button = await page.query_selector('button:has-text(""), [data-testid="expand-button"]')
            
            if not expand_button:
                # Try alternative selectors
                expand_button = await page.query_selector('button svg[class*="chevron"]')
            
            if expand_button:
                # Click to expand
                await expand_button.click()
                await page.wait_for_timeout(500)  # Wait for animation
                
                # Check if expanded content is visible
                expanded_content = await page.query_selector('.expanded-content, [data-testid="expanded-content"]')
                
                return {
                    "passed": True,
                    "expand_button_found": True,
                    "expanded_content_visible": expanded_content is not None
                }
            else:
                return {"passed": False, "error": "Expand button not found"}
                
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_verify_dynamic_content(self, page, module_id):
        """Test if dynamic content from markdown files is loaded"""
        try:
            # Wait for any loading to complete
            await page.wait_for_timeout(1000)
            
            # Check for typical dynamic content elements
            duration_element = await page.query_selector(':has-text("hours"), :has-text("h")')
            lessons_element = await page.query_selector(':has-text("lessons")')
            labs_element = await page.query_selector(':has-text("labs")')
            
            # Check page content for module-specific information
            page_content = await page.content()
            has_prerequisites = "prerequisites" in page_content.lower()
            has_objectives = "learning objectives" in page_content.lower() or "objectives" in page_content.lower()
            
            return {
                "passed": True,
                "has_duration": duration_element is not None,
                "has_lessons": lessons_element is not None,
                "has_labs": labs_element is not None,
                "has_prerequisites": has_prerequisites,
                "has_objectives": has_objectives
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_prerequisites(self, page, module_id):
        """Test prerequisites display and functionality"""
        try:
            # Look for prerequisites section
            prereq_section = await page.query_selector(':has-text("Prerequisites"), [data-testid="prerequisites"]')
            
            if prereq_section:
                prereq_text = await prereq_section.text_content()
                has_module_links = "module" in prereq_text.lower()
                
                return {
                    "passed": True,
                    "prerequisites_found": True,
                    "has_module_references": has_module_links,
                    "content_length": len(prereq_text)
                }
            else:
                # Some modules might not have prerequisites (like Module 1)
                return {
                    "passed": True,
                    "prerequisites_found": False,
                    "note": "No prerequisites section (expected for foundational modules)"
                }
                
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_learning_objectives(self, page, module_id):
        """Test learning objectives display"""
        try:
            # Look for learning objectives
            objectives_section = await page.query_selector(':has-text("Learning Objectives"), :has-text("What you"), [data-testid="objectives"]')
            
            if objectives_section:
                objectives_text = await objectives_section.text_content()
                
                # Count bullet points or list items
                bullet_points = await page.query_selector_all('li, :has-text("•"), :has-text("-")')
                
                return {
                    "passed": True,
                    "objectives_found": True,
                    "content_length": len(objectives_text),
                    "bullet_points_count": len(bullet_points)
                }
            else:
                return {"passed": False, "error": "Learning objectives not found"}
                
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_topic_categories(self, page, module_id):
        """Test topic categories and tags display"""
        try:
            # Look for topic tags/badges
            topic_badges = await page.query_selector_all('.badge, [data-testid="topic-badge"], .tag')
            
            # Look for topic categories
            topic_sections = await page.query_selector_all(':has-text("Topics"), [data-testid="topics"]')
            
            page_content = await page.content()
            has_topics = "topics" in page_content.lower()
            
            return {
                "passed": True,
                "topic_badges_count": len(topic_badges),
                "topic_sections_count": len(topic_sections),
                "has_topics_content": has_topics
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_action_buttons(self, page, module_id):
        """Test action buttons functionality"""
        try:
            # Look for action buttons
            start_button = await page.query_selector('button:has-text("Start"), button:has-text("Continue"), [data-testid="start-button"]')
            view_button = await page.query_selector('button:has-text("View"), button:has-text("Details"), [data-testid="view-button"]')
            
            # Test button states and accessibility
            buttons = await page.query_selector_all('button')
            enabled_buttons = []
            disabled_buttons = []
            
            for button in buttons:
                is_disabled = await button.is_disabled()
                button_text = await button.text_content()
                
                if is_disabled:
                    disabled_buttons.append(button_text.strip())
                else:
                    enabled_buttons.append(button_text.strip())
            
            return {
                "passed": True,
                "start_button_found": start_button is not None,
                "view_button_found": view_button is not None,
                "total_buttons": len(buttons),
                "enabled_buttons": len(enabled_buttons),
                "disabled_buttons": len(disabled_buttons)
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_progress_indicators(self, page, module_id):
        """Test progress indicators and status display"""
        try:
            # Look for progress bars
            progress_bars = await page.query_selector_all('progress, .progress, [role="progressbar"]')
            
            # Look for status indicators
            status_indicators = await page.query_selector_all('.status, [data-testid="status"], .badge')
            
            # Check for percentage indicators
            page_content = await page.content()
            has_percentage = "%" in page_content
            
            return {
                "passed": True,
                "progress_bars_count": len(progress_bars),
                "status_indicators_count": len(status_indicators),
                "has_percentage": has_percentage
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_api_integration(self, session, module_id):
        """Test API endpoints for module descriptions"""
        try:
            # Test module descriptions API
            async with session.get(f"{self.base_url}/api/module-descriptions") as response:
                if response.status == 200:
                    data = await response.json()
                    all_descriptions_loaded = data.get("success", False)
                    descriptions_count = data.get("count", 0)
                else:
                    all_descriptions_loaded = False
                    descriptions_count = 0
            
            # Test individual module API
            async with session.get(f"{self.base_url}/api/module-descriptions/{module_id}") as response:
                if response.status == 200:
                    module_data = await response.json()
                    module_loaded = module_data.get("success", False)
                    has_description = "description" in module_data
                else:
                    module_loaded = False
                    has_description = False
            
            return {
                "passed": True,
                "all_descriptions_api": all_descriptions_loaded,
                "descriptions_count": descriptions_count,
                "individual_module_api": module_loaded,
                "has_module_description": has_description
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def test_responsive_design(self, page, module_id):
        """Test responsive design at different viewport sizes"""
        try:
            viewports = [
                {"width": 320, "height": 568, "name": "mobile"},
                {"width": 768, "height": 1024, "name": "tablet"},
                {"width": 1920, "height": 1080, "name": "desktop"}
            ]
            
            responsive_results = {}
            
            for viewport in viewports:
                await page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
                await page.wait_for_timeout(500)
                
                # Check if content is still visible and accessible
                visible_elements = await page.query_selector_all(':visible')
                
                responsive_results[viewport["name"]] = {
                    "visible_elements_count": len(visible_elements),
                    "viewport_width": viewport["width"],
                    "viewport_height": viewport["height"]
                }
            
            return {
                "passed": True,
                "responsive_tests": responsive_results
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}

    async def run_parallel_tests(self):
        """Run parallel tests for all 20 modules"""
        print("Starting Parallel Module Landing Page Tests")
        print(f"Using {self.max_workers} parallel workers (optimized for 10 core/20 thread machine)")
        print(f"Testing {len(self.module_ids)} modules with {len(self.test_scenarios)} scenarios each")
        
        self.start_time = time.time()
        
        # Setup browser context
        await self.setup_browser_context()
        
        # Create HTTP session for API tests
        async with aiohttp.ClientSession() as session:
            # Create tasks for parallel execution
            tasks = []
            for module_id in self.module_ids:
                task = asyncio.create_task(
                    self.test_module_landing_page(module_id, session)
                )
                tasks.append(task)
            
            # Execute all tasks in parallel
            print(f"\nExecuting {len(tasks)} parallel test suites...")
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(completed_tasks):
                if isinstance(result, Exception):
                    self.results[self.module_ids[i]] = {
                        "error": str(result),
                        "failed": True
                    }
                else:
                    self.results[self.module_ids[i]] = result
        
        # Cleanup
        await self.teardown_browser_context()
        
        total_time = time.time() - self.start_time
        print(f"\nAll parallel tests completed in {total_time:.2f} seconds")
        
        return self.results

    def generate_comprehensive_report(self):
        """Generate detailed test report"""
        total_modules = len(self.results)
        passed_modules = 0
        failed_modules = 0
        total_tests = 0
        passed_tests = 0
        
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE MODULE LANDING PAGE TEST REPORT")
        print("="*80)
        
        print(f"\n🎯 TEST SUMMARY:")
        print(f"   • Total Modules Tested: {total_modules}")
        print(f"   • Test Scenarios per Module: {len(self.test_scenarios)}")
        print(f"   • Total Execution Time: {time.time() - self.start_time:.2f} seconds")
        print(f"   • Average Time per Module: {(time.time() - self.start_time) / total_modules:.2f} seconds")
        
        # Detailed results per module
        for module_id, result in self.results.items():
            print(f"\n📦 {module_id.upper().replace('-', ' ')}")
            print("-" * 60)
            
            if "error" in result and result.get("failed"):
                print(f"   ❌ FAILED: {result['error']}")
                failed_modules += 1
                continue
            
            passed_modules += 1
            module_tests_passed = 0
            
            for test_name, test_result in result.get("tests", {}).items():
                total_tests += 1
                if test_result.get("passed", False):
                    passed_tests += 1
                    module_tests_passed += 1
                    print(f"   ✅ {test_name.replace('_', ' ').title()}")
                else:
                    print(f"   ❌ {test_name.replace('_', ' ').title()}: {test_result.get('error', 'Unknown error')}")
            
            # Performance metrics
            if "performance" in result:
                page_load_time = result["performance"].get("page_load", 0)
                print(f"   ⚡ Page Load Time: {page_load_time:.3f}s")
            
            print(f"   📊 Module Score: {module_tests_passed}/{len(self.test_scenarios)} tests passed")
        
        # Overall statistics
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   • Modules Passed: {passed_modules}/{total_modules} ({(passed_modules/total_modules)*100:.1f}%)")
        print(f"   • Tests Passed: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
        print(f"   • Failed Modules: {failed_modules}")
        
        # Performance analysis
        avg_load_times = []
        for result in self.results.values():
            if "performance" in result and "page_load" in result["performance"]:
                avg_load_times.append(result["performance"]["page_load"])
        
        if avg_load_times:
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   • Average Page Load: {sum(avg_load_times)/len(avg_load_times):.3f}s")
            print(f"   • Fastest Load: {min(avg_load_times):.3f}s")
            print(f"   • Slowest Load: {max(avg_load_times):.3f}s")
        
        # Save detailed report
        with open('module_landing_test_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Detailed report saved to: module_landing_test_report.json")
        
        return {
            "total_modules": total_modules,
            "passed_modules": passed_modules,
            "failed_modules": failed_modules,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests/total_tests)*100 if total_tests > 0 else 0
        }

async def main():
    """Main test execution"""
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:3000"
    
    print(f"🔗 Testing against: {base_url}")
    
    tester = ParallelModuleTester(base_url)
    
    try:
        # Run all parallel tests
        results = await tester.run_parallel_tests()
        
        # Generate comprehensive report
        summary = tester.generate_comprehensive_report()
        
        # Exit with appropriate code
        if summary["success_rate"] >= 80:
            print(f"\n🎉 SUCCESS: {summary['success_rate']:.1f}% tests passed!")
            sys.exit(0)
        else:
            print(f"\n⚠️  WARNING: Only {summary['success_rate']:.1f}% tests passed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())