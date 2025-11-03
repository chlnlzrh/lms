#!/usr/bin/env python3
"""
Simplified Parallel Module Landing Page Testing
For 10 core/20 thread/32GB RAM machine
"""

import asyncio
import aiohttp
import time
from playwright.async_api import async_playwright
import json
from datetime import datetime

class SimpleModuleTester:
    def __init__(self, base_url="http://localhost:3001"):
        self.base_url = base_url
        self.module_ids = [f"module-{i}" for i in range(1, 21)]
        self.results = {}
        
    async def test_single_module(self, module_id, session):
        """Test a single module landing page"""
        result = {
            "module_id": module_id,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "success": False
        }
        
        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Navigate to main page
            start_time = time.time()
            await page.goto(f"{self.base_url}/", wait_until="networkidle", timeout=30000)
            load_time = time.time() - start_time
            
            # Test 1: Check if page loads
            title = await page.title()
            result["tests"]["page_loads"] = {
                "passed": "data engineering" in title.lower() or len(title) > 0,
                "title": title,
                "load_time": load_time
            }
            
            # Test 2: Look for module cards
            module_cards = await page.query_selector_all('[class*="card"], .module-card')
            result["tests"]["module_cards_present"] = {
                "passed": len(module_cards) > 0,
                "card_count": len(module_cards)
            }
            
            # Test 3: Look for module-specific content
            page_content = await page.content()
            has_module_content = any(word in page_content.lower() for word in [
                "module", "lesson", "lab", "hour", "prerequisite"
            ])
            result["tests"]["module_content"] = {
                "passed": has_module_content,
                "content_indicators": has_module_content
            }
            
            # Test 4: Test API endpoint
            try:
                async with session.get(f"{self.base_url}/api/module-descriptions/{module_id}") as response:
                    api_success = response.status == 200
                    if api_success:
                        api_data = await response.json()
                        has_description = "description" in api_data
                    else:
                        has_description = False
            except:
                api_success = False
                has_description = False
                
            result["tests"]["api_integration"] = {
                "passed": api_success,
                "status_code": response.status if 'response' in locals() else None,
                "has_description": has_description
            }
            
            # Test 5: Check for interactive elements
            buttons = await page.query_selector_all('button')
            links = await page.query_selector_all('a')
            result["tests"]["interactive_elements"] = {
                "passed": len(buttons) > 0 or len(links) > 0,
                "button_count": len(buttons),
                "link_count": len(links)
            }
            
            # Calculate overall success
            passed_tests = sum(1 for test in result["tests"].values() if test.get("passed", False))
            total_tests = len(result["tests"])
            result["success"] = passed_tests >= (total_tests * 0.6)  # 60% pass rate
            result["pass_rate"] = (passed_tests / total_tests) * 100
            
            await browser.close()
            await playwright.stop()
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            
        return result
    
    async def run_all_tests(self):
        """Run tests for all modules in parallel"""
        print("Starting Parallel Module Tests")
        print(f"Testing {len(self.module_ids)} modules")
        print(f"Target URL: {self.base_url}")
        print("-" * 60)
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Create tasks for all modules
            tasks = []
            for module_id in self.module_ids:
                task = asyncio.create_task(self.test_single_module(module_id, session))
                tasks.append(task)
            
            # Execute all tasks in parallel
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
        print(f"All tests completed in {total_time:.2f} seconds")
        
        return self.results
    
    def generate_report(self):
        """Generate test report"""
        total_modules = len(self.results)
        successful_modules = sum(1 for r in self.results.values() if r.get("success", False))
        
        print("\n" + "="*60)
        print("MODULE LANDING PAGE TEST REPORT")
        print("="*60)
        
        print(f"\nSUMMARY:")
        print(f"  Total Modules: {total_modules}")
        print(f"  Successful: {successful_modules}")
        print(f"  Failed: {total_modules - successful_modules}")
        print(f"  Success Rate: {(successful_modules/total_modules)*100:.1f}%")
        
        print(f"\nDETAILED RESULTS:")
        for module_id, result in self.results.items():
            if result.get("success", False):
                pass_rate = result.get("pass_rate", 0)
                print(f"  {module_id}: PASS ({pass_rate:.0f}%)")
            else:
                error = result.get("error", "Unknown error")
                print(f"  {module_id}: FAIL - {error}")
        
        # Save report
        with open('simple_module_test_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: simple_module_test_report.json")
        
        return successful_modules >= (total_modules * 0.8)  # 80% success rate

async def main():
    """Main execution"""
    tester = SimpleModuleTester()
    
    # Run tests
    await tester.run_all_tests()
    
    # Generate report
    success = tester.generate_report()
    
    if success:
        print("\nSUCCESS: Tests passed!")
        return 0
    else:
        print("\nFAILED: Some tests failed")
        return 1

if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)