#!/usr/bin/env python3
"""
Parallel Comprehensive Test Suite for Data Engineering Learning Platform
Runs multiple test suites in parallel for faster execution
Uses the 10 core/20 thread machine capabilities
"""

from playwright.sync_api import sync_playwright
import time
import json
import os
import threading
import concurrent.futures
from datetime import datetime
from queue import Queue
import tempfile


class ParallelWebAppTester:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "parallel_execution": True,
            "machine_specs": "10 core/20 thread/32GB RAM",
            "test_suites": {},
            "global_issues": [],
            "performance_summary": {},
            "screenshots": [],
            "execution_time": 0
        }
        
        # Thread-safe queues for collecting results
        self.results_queue = Queue()
        self.issues_queue = Queue()
        self.screenshots_queue = Queue()
        
        # Test configurations for parallel execution
        self.test_suites = {
            "performance_suite": {
                "pages": ["/", "/learning-path", "/lessons", "/projects", "/community"],
                "metrics": ["load_time", "first_paint", "largest_contentful_paint"]
            },
            "responsive_suite": {
                "viewports": [
                    {"name": "mobile_320", "width": 320, "height": 568},
                    {"name": "mobile_375", "width": 375, "height": 667},
                    {"name": "tablet_768", "width": 768, "height": 1024},
                    {"name": "desktop_1024", "width": 1024, "height": 768},
                    {"name": "desktop_1440", "width": 1440, "height": 900},
                    {"name": "ultrawide_1920", "width": 1920, "height": 1080}
                ]
            },
            "accessibility_suite": {
                "tests": ["keyboard_nav", "aria_labels", "color_contrast", "screen_reader"],
                "browsers": ["chromium", "firefox", "webkit"]
            },
            "functionality_suite": {
                "components": ["navigation", "dark_mode", "search", "filters", "cards", "buttons"],
                "user_flows": ["browse_lessons", "search_lessons", "view_lesson_details"]
            },
            "cross_browser_suite": {
                "browsers": ["chromium", "firefox", "webkit"],
                "pages": ["/", "/learning-path", "/lessons"]
            }
        }

    def log_result(self, suite_name, test_name, status, details=None, thread_id=None):
        """Thread-safe result logging"""
        result = {
            "suite": suite_name,
            "test": test_name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id
        }
        self.results_queue.put(result)
        print(f"[{suite_name}][Thread-{thread_id}] {test_name}: {status}")

    def log_issue(self, severity, description, location=None, thread_id=None):
        """Thread-safe issue logging"""
        issue = {
            "severity": severity,
            "description": description,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id
        }
        self.issues_queue.put(issue)
        print(f"[{severity}][Thread-{thread_id}] {description}")

    def take_screenshot(self, page, name, suite_name, thread_id=None):
        """Thread-safe screenshot capture"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir = tempfile.gettempdir()
        filename = f"{temp_dir}\\{suite_name}_{name}_{timestamp}_t{thread_id}.png"
        try:
            page.screenshot(path=filename, full_page=True)
            self.screenshots_queue.put(filename)
            return filename
        except Exception as e:
            self.log_issue("LOW", f"Screenshot failed: {str(e)}", thread_id=thread_id)
            return None

    def performance_test_suite(self, thread_id):
        """Run performance tests on multiple pages"""
        suite_name = "performance_suite"
        print(f"[{suite_name}][Thread-{thread_id}] Starting performance testing...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            performance_results = {}
            
            for test_page in self.test_suites[suite_name]["pages"]:
                url = f"{self.base_url}{test_page}"
                print(f"[{suite_name}][Thread-{thread_id}] Testing {url}")
                
                try:
                    # Measure load time
                    start_time = time.time()
                    page.goto(url, timeout=30000)
                    page.wait_for_load_state('networkidle', timeout=30000)
                    load_time = time.time() - start_time
                    
                    # Get performance metrics
                    perf_data = page.evaluate("""
                        () => {
                            const navigation = performance.getEntriesByType('navigation')[0];
                            const paint = performance.getEntriesByName('first-contentful-paint')[0];
                            return {
                                loadTime: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
                                domContentLoaded: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : 0,
                                firstContentfulPaint: paint ? paint.startTime : 0,
                                resourceCount: performance.getEntriesByType('resource').length
                            }
                        }
                    """)
                    
                    # Take screenshot
                    screenshot = self.take_screenshot(page, f"perf_{test_page.replace('/', 'home')}", suite_name, thread_id)
                    
                    performance_results[test_page] = {
                        "total_load_time": load_time,
                        "metrics": perf_data,
                        "screenshot": screenshot
                    }
                    
                    # Performance thresholds
                    if load_time > 3:
                        self.log_issue("HIGH", f"Slow page load: {url} took {load_time:.2f}s", url, thread_id)
                    elif load_time > 1.5:
                        self.log_issue("MEDIUM", f"Suboptimal load time: {url} took {load_time:.2f}s", url, thread_id)
                    
                    self.log_result(suite_name, f"Performance-{test_page}", "PASS", 
                                  {"load_time": f"{load_time:.2f}s", "metrics": perf_data}, thread_id)
                    
                except Exception as e:
                    self.log_issue("HIGH", f"Performance test failed for {url}: {str(e)}", url, thread_id)
                    self.log_result(suite_name, f"Performance-{test_page}", "FAIL", {"error": str(e)}, thread_id)
            
            browser.close()
            return performance_results

    def responsive_test_suite(self, thread_id):
        """Run responsive design tests across viewports"""
        suite_name = "responsive_suite"
        print(f"[{suite_name}][Thread-{thread_id}] Starting responsive testing...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            responsive_results = {}
            
            # Load the main page first
            page.goto(self.base_url, timeout=30000)
            page.wait_for_load_state('networkidle', timeout=30000)
            
            for viewport in self.test_suites[suite_name]["viewports"]:
                print(f"[{suite_name}][Thread-{thread_id}] Testing {viewport['name']}")
                
                try:
                    # Set viewport
                    page.set_viewport_size({"width": viewport['width'], "height": viewport['height']})
                    page.wait_for_timeout(1000)  # Allow responsive adjustments
                    
                    # Take screenshot
                    screenshot = self.take_screenshot(page, f"responsive_{viewport['name']}", suite_name, thread_id)
                    
                    # Check for responsive issues
                    issues = []
                    
                    # Check for horizontal overflow
                    body_width = page.evaluate("document.body.scrollWidth")
                    if body_width > viewport['width']:
                        issues.append(f"Horizontal overflow: {body_width}px > {viewport['width']}px")
                        self.log_issue("MEDIUM", f"Horizontal overflow in {viewport['name']}", thread_id=thread_id)
                    
                    # Check mobile menu on small screens
                    if viewport['width'] <= 768:
                        mobile_menu_count = page.locator("button[aria-label*='menu'], .hamburger, [data-testid*='menu']").count()
                        if mobile_menu_count == 0:
                            issues.append("No mobile menu trigger found")
                            self.log_issue("HIGH", f"No mobile menu trigger in {viewport['name']}", thread_id=thread_id)
                    
                    # Check for readable text size
                    small_text_count = page.evaluate("""
                        () => {
                            const elements = document.querySelectorAll('*');
                            let count = 0;
                            elements.forEach(el => {
                                const style = window.getComputedStyle(el);
                                const fontSize = parseFloat(style.fontSize);
                                if (fontSize < 14 && el.innerText && el.innerText.trim().length > 0) {
                                    count++;
                                }
                            });
                            return count;
                        }
                    """)
                    
                    if small_text_count > 5:
                        issues.append(f"{small_text_count} elements with small text (<14px)")
                        self.log_issue("MEDIUM", f"Small text issues in {viewport['name']}: {small_text_count} elements", thread_id=thread_id)
                    
                    responsive_results[viewport['name']] = {
                        "viewport": viewport,
                        "issues": issues,
                        "screenshot": screenshot,
                        "body_width": body_width
                    }
                    
                    status = "FAIL" if issues else "PASS"
                    self.log_result(suite_name, f"Responsive-{viewport['name']}", status, 
                                  {"issues_count": len(issues), "issues": issues}, thread_id)
                    
                except Exception as e:
                    self.log_issue("HIGH", f"Responsive test failed for {viewport['name']}: {str(e)}", thread_id=thread_id)
                    self.log_result(suite_name, f"Responsive-{viewport['name']}", "FAIL", {"error": str(e)}, thread_id)
            
            browser.close()
            return responsive_results

    def accessibility_test_suite(self, thread_id):
        """Run accessibility tests"""
        suite_name = "accessibility_suite"
        print(f"[{suite_name}][Thread-{thread_id}] Starting accessibility testing...")
        
        accessibility_results = {}
        
        for browser_name in self.test_suites[suite_name]["browsers"]:
            print(f"[{suite_name}][Thread-{thread_id}] Testing accessibility in {browser_name}")
            
            try:
                with sync_playwright() as p:
                    browser_type = getattr(p, browser_name)
                    browser = browser_type.launch(headless=True)
                    page = browser.new_page()
                    
                    page.goto(self.base_url, timeout=30000)
                    page.wait_for_load_state('networkidle', timeout=30000)
                    
                    # Test keyboard navigation
                    keyboard_results = {}
                    try:
                        page.keyboard.press('Tab')
                        focused_element = page.evaluate("document.activeElement.tagName")
                        keyboard_results["first_tab"] = focused_element
                        
                        # Test multiple tabs
                        for i in range(10):
                            page.keyboard.press('Tab')
                        
                        final_focused = page.evaluate("document.activeElement.tagName")
                        keyboard_results["tab_navigation"] = "functional" if final_focused else "limited"
                        
                    except Exception as e:
                        keyboard_results["error"] = str(e)
                        self.log_issue("HIGH", f"Keyboard navigation error in {browser_name}: {str(e)}", thread_id=thread_id)
                    
                    # Test ARIA elements
                    aria_results = {
                        "buttons_with_aria": page.locator("button[aria-label], button[aria-describedby]").count(),
                        "headings": page.locator("h1, h2, h3, h4, h5, h6").count(),
                        "landmarks": page.locator("[role='main'], [role='navigation'], [role='banner']").count(),
                        "images_with_alt": page.locator("img[alt]").count(),
                        "images_without_alt": page.locator("img:not([alt])").count()
                    }
                    
                    # Check for accessibility issues
                    if aria_results["images_without_alt"] > 0:
                        self.log_issue("MEDIUM", f"{aria_results['images_without_alt']} images missing alt text in {browser_name}", thread_id=thread_id)
                    
                    if aria_results["headings"] == 0:
                        self.log_issue("HIGH", f"No heading elements found in {browser_name}", thread_id=thread_id)
                    
                    # Take accessibility screenshot
                    screenshot = self.take_screenshot(page, f"accessibility_{browser_name}", suite_name, thread_id)
                    
                    accessibility_results[browser_name] = {
                        "keyboard_navigation": keyboard_results,
                        "aria_elements": aria_results,
                        "screenshot": screenshot
                    }
                    
                    self.log_result(suite_name, f"Accessibility-{browser_name}", "PASS", aria_results, thread_id)
                    
                    browser.close()
                    
            except Exception as e:
                self.log_issue("HIGH", f"Accessibility test failed for {browser_name}: {str(e)}", thread_id=thread_id)
                self.log_result(suite_name, f"Accessibility-{browser_name}", "FAIL", {"error": str(e)}, thread_id)
        
        return accessibility_results

    def functionality_test_suite(self, thread_id):
        """Run functionality tests"""
        suite_name = "functionality_suite"
        print(f"[{suite_name}][Thread-{thread_id}] Starting functionality testing...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            functionality_results = {}
            
            try:
                page.goto(self.base_url, timeout=30000)
                page.wait_for_load_state('networkidle', timeout=30000)
                
                # Test navigation functionality
                print(f"[{suite_name}][Thread-{thread_id}] Testing navigation")
                nav_tests = {
                    "nav_links_count": page.locator("nav a, .nav-link").count(),
                    "buttons_count": page.locator("button").count(),
                    "interactive_elements": page.locator("button, a, input, select, textarea").count()
                }
                
                # Test dark mode toggle
                print(f"[{suite_name}][Thread-{thread_id}] Testing dark mode")
                dark_mode_results = {"found": False, "functional": False}
                
                dark_mode_selectors = [
                    "button[aria-label*='theme']",
                    "button[aria-label*='dark']",
                    ".theme-toggle",
                    "[data-testid*='theme']"
                ]
                
                for selector in dark_mode_selectors:
                    try:
                        toggle = page.locator(selector).first
                        if toggle.is_visible():
                            dark_mode_results["found"] = True
                            
                            # Test toggle functionality
                            initial_class = page.evaluate("document.documentElement.className")
                            toggle.click()
                            page.wait_for_timeout(500)
                            after_class = page.evaluate("document.documentElement.className")
                            
                            if initial_class != after_class:
                                dark_mode_results["functional"] = True
                                # Toggle back
                                toggle.click()
                                page.wait_for_timeout(500)
                            break
                    except:
                        continue
                
                if not dark_mode_results["found"]:
                    self.log_issue("MEDIUM", "Dark mode toggle not found", thread_id=thread_id)
                elif not dark_mode_results["functional"]:
                    self.log_issue("HIGH", "Dark mode toggle not functional", thread_id=thread_id)
                
                # Test lessons page if it exists
                print(f"[{suite_name}][Thread-{thread_id}] Testing lessons page")
                lessons_test = {"accessible": False, "search_functional": False}
                
                try:
                    # Try to navigate to lessons page
                    lessons_link = page.locator("text='Lessons', a[href*='/lessons']").first
                    if lessons_link.is_visible():
                        lessons_link.click()
                        page.wait_for_load_state('networkidle', timeout=10000)
                        lessons_test["accessible"] = True
                        
                        # Test search functionality if present
                        search_input = page.locator("input[placeholder*='search'], input[type='search']").first
                        if search_input.is_visible():
                            search_input.fill("test query")
                            page.wait_for_timeout(1000)
                            lessons_test["search_functional"] = True
                            
                        # Navigate back
                        page.go_back()
                        page.wait_for_load_state('networkidle', timeout=10000)
                except Exception as e:
                    self.log_issue("MEDIUM", f"Lessons page test failed: {str(e)}", thread_id=thread_id)
                
                # Take functionality screenshot
                screenshot = self.take_screenshot(page, "functionality_overview", suite_name, thread_id)
                
                functionality_results = {
                    "navigation": nav_tests,
                    "dark_mode": dark_mode_results,
                    "lessons_page": lessons_test,
                    "screenshot": screenshot
                }
                
                self.log_result(suite_name, "Navigation", "PASS", nav_tests, thread_id)
                self.log_result(suite_name, "Dark Mode", "PASS" if dark_mode_results["functional"] else "FAIL", dark_mode_results, thread_id)
                self.log_result(suite_name, "Lessons Page", "PASS" if lessons_test["accessible"] else "FAIL", lessons_test, thread_id)
                
            except Exception as e:
                self.log_issue("HIGH", f"Functionality test failed: {str(e)}", thread_id=thread_id)
                self.log_result(suite_name, "Functionality", "FAIL", {"error": str(e)}, thread_id)
            
            browser.close()
            return functionality_results

    def cross_browser_test_suite(self, thread_id):
        """Run cross-browser compatibility tests"""
        suite_name = "cross_browser_suite"
        print(f"[{suite_name}][Thread-{thread_id}] Starting cross-browser testing...")
        
        browser_results = {}
        
        for browser_name in self.test_suites[suite_name]["browsers"]:
            print(f"[{suite_name}][Thread-{thread_id}] Testing {browser_name}")
            
            try:
                with sync_playwright() as p:
                    browser_type = getattr(p, browser_name)
                    browser = browser_type.launch(headless=True)
                    page = browser.new_page()
                    
                    # Test main pages
                    browser_page_results = {}
                    
                    for test_page in self.test_suites[suite_name]["pages"]:
                        url = f"{self.base_url}{test_page}"
                        
                        try:
                            start_time = time.time()
                            page.goto(url, timeout=30000)
                            page.wait_for_load_state('networkidle', timeout=30000)
                            load_time = time.time() - start_time
                            
                            # Get basic page metrics
                            metrics = {
                                "load_time": load_time,
                                "title": page.title(),
                                "buttons": page.locator("button").count(),
                                "links": page.locator("a").count(),
                                "images": page.locator("img").count()
                            }
                            
                            browser_page_results[test_page] = metrics
                            
                            if load_time > 5:
                                self.log_issue("MEDIUM", f"Slow load in {browser_name} for {test_page}: {load_time:.2f}s", thread_id=thread_id)
                            
                        except Exception as e:
                            browser_page_results[test_page] = {"error": str(e)}
                            self.log_issue("HIGH", f"Page load failed in {browser_name} for {test_page}: {str(e)}", thread_id=thread_id)
                    
                    # Take browser-specific screenshot
                    screenshot = self.take_screenshot(page, f"browser_{browser_name}", suite_name, thread_id)
                    
                    browser_results[browser_name] = {
                        "pages": browser_page_results,
                        "screenshot": screenshot,
                        "status": "PASS"
                    }
                    
                    self.log_result(suite_name, f"Browser-{browser_name}", "PASS", browser_page_results, thread_id)
                    
                    browser.close()
                    
            except Exception as e:
                browser_results[browser_name] = {"status": "FAIL", "error": str(e)}
                self.log_issue("HIGH", f"Browser {browser_name} test failed: {str(e)}", thread_id=thread_id)
                self.log_result(suite_name, f"Browser-{browser_name}", "FAIL", {"error": str(e)}, thread_id)
        
        return browser_results

    def run_parallel_tests(self):
        """Run all test suites in parallel"""
        print("Starting Parallel Comprehensive Web App Testing")
        print(f"Target URL: {self.base_url}")
        print(f"Machine: {self.test_results['machine_specs']}")
        print("Using 5 parallel test threads (10 core machine)")
        print("=" * 70)
        
        start_time = time.time()
        
        # Define test suite functions
        test_functions = [
            self.performance_test_suite,
            self.responsive_test_suite,
            self.accessibility_test_suite,
            self.functionality_test_suite,
            self.cross_browser_test_suite
        ]
        
        # Run tests in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all test suites
            future_to_suite = {
                executor.submit(test_func, i+1): test_func.__name__ 
                for i, test_func in enumerate(test_functions)
            }
            
            # Collect results as they complete
            suite_results = {}
            for future in concurrent.futures.as_completed(future_to_suite):
                suite_name = future_to_suite[future]
                try:
                    result = future.result()
                    suite_results[suite_name] = result
                    print(f"[COMPLETED] {suite_name} finished successfully")
                except Exception as e:
                    print(f"[FAILED] {suite_name} failed: {str(e)}")
                    self.log_issue("CRITICAL", f"Test suite {suite_name} failed: {str(e)}")
        
        execution_time = time.time() - start_time
        self.test_results["execution_time"] = execution_time
        
        # Collect all results from queues
        self.collect_queue_results()
        
        # Store suite results
        self.test_results["suite_results"] = suite_results
        
        # Generate summary and save results
        self.generate_performance_summary()
        self.save_results()
        self.print_summary()
        
        print(f"\n[INFO] Total execution time: {execution_time:.2f} seconds")
        print(f"[INFO] Parallel efficiency: ~{execution_time/5:.2f}s per suite (vs ~{execution_time:.2f}s sequential)")

    def collect_queue_results(self):
        """Collect results from thread-safe queues"""
        # Collect test results
        while not self.results_queue.empty():
            result = self.results_queue.get()
            suite_name = result["suite"]
            if suite_name not in self.test_results["test_suites"]:
                self.test_results["test_suites"][suite_name] = {}
            self.test_results["test_suites"][suite_name][result["test"]] = result
        
        # Collect issues
        while not self.issues_queue.empty():
            issue = self.issues_queue.get()
            self.test_results["global_issues"].append(issue)
        
        # Collect screenshots
        while not self.screenshots_queue.empty():
            screenshot = self.screenshots_queue.get()
            self.test_results["screenshots"].append(screenshot)

    def generate_performance_summary(self):
        """Generate performance summary"""
        total_tests = sum(len(suite) for suite in self.test_results["test_suites"].values())
        passed_tests = sum(
            1 for suite in self.test_results["test_suites"].values() 
            for test in suite.values() 
            if test["status"] == "PASS"
        )
        
        issues_by_severity = {}
        for issue in self.test_results["global_issues"]:
            severity = issue["severity"]
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        
        self.test_results["performance_summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_issues": len(self.test_results["global_issues"]),
            "issues_by_severity": issues_by_severity,
            "total_screenshots": len(self.test_results["screenshots"])
        }

    def save_results(self):
        """Save comprehensive test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        filename = f"{temp_dir}\\parallel_webapp_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n[INFO] Comprehensive test results saved: {filename}")
        return filename

    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 70)
        print("PARALLEL COMPREHENSIVE TEST SUMMARY")
        print("=" * 70)
        
        summary = self.test_results["performance_summary"]
        
        print(f"Execution Mode: Parallel (5 concurrent test suites)")
        print(f"Total Execution Time: {self.test_results['execution_time']:.2f} seconds")
        print(f"Average Time per Suite: {self.test_results['execution_time']/5:.2f} seconds")
        
        print(f"\nTest Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed_tests']}")
        print(f"  Failed: {summary['failed_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\nIssues Found: {summary['total_issues']}")
        for severity, count in summary['issues_by_severity'].items():
            print(f"  {severity}: {count}")
        
        print(f"\nArtifacts Generated:")
        print(f"  Screenshots: {summary['total_screenshots']}")
        print(f"  Test Suites: {len(self.test_results['test_suites'])}")
        
        print(f"\nTest Suite Breakdown:")
        for suite_name, tests in self.test_results["test_suites"].items():
            passed = sum(1 for test in tests.values() if test["status"] == "PASS")
            total = len(tests)
            print(f"  {suite_name}: {passed}/{total} passed")
        
        if self.test_results["screenshots"]:
            print(f"\nScreenshot Locations (first 5):")
            for screenshot in self.test_results["screenshots"][:5]:
                print(f"  {screenshot}")
            if len(self.test_results["screenshots"]) > 5:
                print(f"  ... and {len(self.test_results['screenshots']) - 5} more")


def main():
    """Main function to run parallel comprehensive tests"""
    import sys
    
    # Default to localhost:3000, but allow override
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
    
    tester = ParallelWebAppTester(base_url)
    tester.run_parallel_tests()


if __name__ == "__main__":
    main()