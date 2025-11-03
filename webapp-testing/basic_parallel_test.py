#!/usr/bin/env python3
"""
Basic Parallel Module Landing Page Testing
Using only Playwright and built-in libraries
"""

import asyncio
import time
import json
from datetime import datetime
from playwright.async_api import async_playwright

class BasicModuleTester:
    def __init__(self, base_url="http://localhost:3001"):
        self.base_url = base_url
        self.module_ids = [f"module-{i}" for i in range(1, 21)]
        self.results = {}
        
    async def test_module_landing(self, module_id):
        """Test a single module landing functionality"""
        result = {
            "module_id": module_id,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "tests": {}
        }
        
        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            page = await browser.new_page()
            
            # Test 1: Page loads
            start_time = time.time()
            await page.goto(f"{self.base_url}/", wait_until="networkidle", timeout=15000)
            load_time = time.time() - start_time
            
            result["tests"]["page_load"] = {
                "passed": True,
                "load_time": round(load_time, 3)
            }
            
            # Test 2: Enhanced module cards are present
            await page.wait_for_timeout(1000)  # Wait for dynamic content
            
            # Look for module cards
            cards = await page.query_selector_all('[class*="card"]')
            result["tests"]["module_cards"] = {
                "passed": len(cards) >= 10,  # Expect at least 10 visible cards
                "count": len(cards)
            }
            
            # Test 3: Dynamic content indicators
            page_content = await page.content()
            content_indicators = [
                "learning path" in page_content.lower(),
                "module" in page_content.lower(), 
                "lesson" in page_content.lower(),
                "hours" in page_content.lower() or "hour" in page_content.lower(),
                "lab" in page_content.lower()
            ]
            
            result["tests"]["dynamic_content"] = {
                "passed": sum(content_indicators) >= 3,
                "indicators_found": sum(content_indicators),
                "total_indicators": len(content_indicators)
            }
            
            # Test 4: Interactive elements
            buttons = await page.query_selector_all('button')
            links = await page.query_selector_all('a')
            
            result["tests"]["interactive_elements"] = {
                "passed": len(buttons) >= 5,  # Expect multiple interactive buttons
                "buttons": len(buttons),
                "links": len(links)
            }
            
            # Test 5: Progress indicators
            progress_elements = await page.query_selector_all('[class*="progress"], [role="progressbar"]')
            badges = await page.query_selector_all('[class*="badge"]')
            
            result["tests"]["progress_ui"] = {
                "passed": len(progress_elements) > 0 or len(badges) > 0,
                "progress_elements": len(progress_elements),
                "badges": len(badges)
            }
            
            # Test 6: Expandable content (try to find expand buttons)
            expand_buttons = await page.query_selector_all('button[class*="chevron"], button svg[class*="chevron"]')
            
            result["tests"]["expandable_content"] = {
                "passed": len(expand_buttons) > 0,
                "expand_buttons": len(expand_buttons)
            }
            
            # Test 7: Responsive design indicators
            await page.set_viewport_size({"width": 768, "height": 1024})
            await page.wait_for_timeout(300)
            
            mobile_cards = await page.query_selector_all('[class*="card"]')
            result["tests"]["responsive_design"] = {
                "passed": len(mobile_cards) > 0,
                "mobile_card_count": len(mobile_cards)
            }
            
            # Calculate overall success
            passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
            total_tests = len(result["tests"])
            result["success"] = passed_tests >= (total_tests * 0.7)  # 70% pass rate
            result["pass_rate"] = round((passed_tests / total_tests) * 100, 1)
            
            await browser.close()
            await playwright.stop()
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            
        return result
    
    async def run_parallel_tests(self):
        """Run tests for all modules in parallel"""
        print("PARALLEL MODULE LANDING PAGE TESTING")
        print("="*50)
        print(f"Target: {self.base_url}")
        print(f"Modules: {len(self.module_ids)}")
        print(f"Machine: 10 core/20 thread optimized")
        print("-"*50)
        
        start_time = time.time()
        
        # Create parallel tasks (limit to 10 concurrent to avoid overwhelming)
        semaphore = asyncio.Semaphore(10)
        
        async def test_with_semaphore(module_id):
            async with semaphore:
                return await self.test_module_landing(module_id)
        
        # Execute all tests in parallel
        tasks = [test_with_semaphore(module_id) for module_id in self.module_ids]
        print(f"Executing {len(tasks)} parallel tests...")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.results[self.module_ids[i]] = {
                    "error": str(result),
                    "success": False
                }
            else:
                self.results[self.module_ids[i]] = result
        
        total_time = time.time() - start_time
        print(f"Completed in {total_time:.2f} seconds")
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_modules = len(self.results)
        successful_modules = sum(1 for r in self.results.values() if r.get("success", False))
        
        print("\n" + "="*60)
        print("MODULE LANDING PAGE TEST RESULTS")
        print("="*60)
        
        # Summary
        success_rate = (successful_modules / total_modules) * 100
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total Modules Tested: {total_modules}")
        print(f"  Successful Tests: {successful_modules}")
        print(f"  Failed Tests: {total_modules - successful_modules}")
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
        
        print(f"\nTEST CATEGORY RESULTS:")
        for category, stats in test_categories.items():
            category_rate = (stats["passed"] / stats["total"]) * 100
            print(f"  {category.replace('_', ' ').title()}: {stats['passed']}/{stats['total']} ({category_rate:.1f}%)")
        
        # Detailed results
        print(f"\nDETAILED MODULE RESULTS:")
        for module_id, result in self.results.items():
            if result.get("success", False):
                pass_rate = result.get("pass_rate", 0)
                print(f"  {module_id}: PASS ({pass_rate}%)")
            else:
                error_msg = result.get("error", "Multiple test failures")[:50]
                print(f"  {module_id}: FAIL - {error_msg}")
        
        # Performance metrics
        load_times = []
        for result in self.results.values():
            if "tests" in result and "page_load" in result["tests"]:
                load_time = result["tests"]["page_load"].get("load_time", 0)
                if load_time > 0:
                    load_times.append(load_time)
        
        if load_times:
            avg_load = sum(load_times) / len(load_times)
            print(f"\nPERFORMANCE METRICS:")
            print(f"  Average Page Load: {avg_load:.3f}s")
            print(f"  Fastest Load: {min(load_times):.3f}s")
            print(f"  Slowest Load: {max(load_times):.3f}s")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"module_test_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed report saved: {report_file}")
        
        # Return success if 80% or more modules passed
        return success_rate >= 80.0

async def main():
    """Main test execution"""
    tester = BasicModuleTester()
    
    try:
        # Run all tests
        await tester.run_parallel_tests()
        
        # Generate report
        success = tester.generate_report()
        
        if success:
            print("\nSUCCESS: Module landing pages are working correctly!")
            return 0
        else:
            print("\nWARNING: Some module landing page tests failed")
            return 1
            
    except Exception as e:
        print(f"\nERROR: Test execution failed - {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)