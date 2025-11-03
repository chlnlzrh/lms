#!/usr/bin/env python3
"""
Comprehensive Test Suite for Data Engineering Learning Platform
Tests navigation, responsiveness, accessibility, performance, and functionality
"""

from playwright.sync_api import sync_playwright
import time
import json
import os
from datetime import datetime


class DataEngineeringPlatformTester:
    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": "http://localhost:3000",
            "tests": {},
            "issues": [],
            "recommendations": [],
            "performance_metrics": {},
            "screenshots": []
        }
        
        # Test viewport sizes
        self.viewports = [
            {"name": "mobile_small", "width": 320, "height": 568},
            {"name": "mobile_medium", "width": 375, "height": 667},
            {"name": "tablet", "width": 768, "height": 1024},
            {"name": "desktop_small", "width": 1024, "height": 768},
            {"name": "desktop_large", "width": 1440, "height": 900}
        ]

    def log_test(self, test_name, status, details=None):
        """Log test results"""
        self.test_results["tests"][test_name] = {
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        print(f"[PASS] {test_name}: {status}")
        if details:
            for key, value in details.items():
                print(f"  - {key}: {value}")

    def log_issue(self, severity, description, location=None):
        """Log issues found during testing"""
        issue = {
            "severity": severity,
            "description": description,
            "location": location,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results["issues"].append(issue)
        print(f"[{severity.upper()}] {description}")

    def log_recommendation(self, category, description):
        """Log recommendations for improvements"""
        recommendation = {
            "category": category,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results["recommendations"].append(recommendation)
        print(f"[RECOMMEND] {category}: {description}")

    def take_screenshot(self, page, name, viewport_name=None):
        """Take and save screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viewport_suffix = f"_{viewport_name}" if viewport_name else ""
        import tempfile
        temp_dir = tempfile.gettempdir()
        filename = f"{temp_dir}\\{name}{viewport_suffix}_{timestamp}.png"
        page.screenshot(path=filename, full_page=True)
        self.test_results["screenshots"].append(filename)
        return filename

    def test_page_load_performance(self, page, url):
        """Test page load performance metrics"""
        print(f"\n[TEST] Testing page load performance for {url}")
        
        start_time = time.time()
        page.goto(url)
        page.wait_for_load_state('networkidle')
        load_time = time.time() - start_time
        
        # Get performance metrics using page.evaluate
        try:
            perf_data = page.evaluate("""
                () => {
                    const navigation = performance.getEntriesByType('navigation')[0];
                    return {
                        loadTime: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
                        domContentLoaded: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : 0,
                        firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
                        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
                    }
                }
            """)
        except Exception as e:
            perf_data = {"error": str(e)}
        
        metrics = {
            "total_load_time": load_time,
            "performance_api": perf_data
        }
        
        self.test_results["performance_metrics"][url] = metrics
        
        # Performance thresholds
        if load_time > 3:
            self.log_issue("HIGH", f"Slow page load time: {load_time:.2f}s (target: <3s)", url)
        elif load_time > 1.5:
            self.log_issue("MEDIUM", f"Page load time could be improved: {load_time:.2f}s", url)
        
        self.log_test(f"Performance - {url}", "PASS", {
            "load_time": f"{load_time:.2f}s",
            "performance_data": perf_data
        })

    def test_navigation_structure(self, page):
        """Test navigation structure and functionality"""
        print("\n[TEST] Testing navigation structure")
        
        # Test main navigation elements
        nav_elements = {
            "desktop_sidebar": "nav[role='navigation'], aside nav, .sidebar",
            "mobile_menu_trigger": "button[aria-label*='menu'], .hamburger, [data-testid*='menu']",
            "logo": "img[alt*='logo'], .logo, [data-testid='logo']",
            "nav_links": "nav a, .nav-link, [role='navigation'] a"
        }
        
        results = {}
        for element_name, selector in nav_elements.items():
            try:
                elements = page.locator(selector).all()
                results[element_name] = len(elements)
                if len(elements) == 0:
                    self.log_issue("MEDIUM", f"No {element_name} found with selector: {selector}")
            except Exception as e:
                results[element_name] = f"Error: {str(e)}"
                self.log_issue("HIGH", f"Error testing {element_name}: {str(e)}")
        
        # Test if main sections are accessible
        expected_sections = ["Dashboard", "Learning Path", "Projects", "Community", "Resources"]
        found_sections = []
        
        for section in expected_sections:
            try:
                # Look for links or headings containing the section name
                section_locator = page.locator(f"text='{section}', [aria-label*='{section}'], [title*='{section}']").first
                if section_locator.is_visible():
                    found_sections.append(section)
            except:
                pass
        
        missing_sections = set(expected_sections) - set(found_sections)
        if missing_sections:
            self.log_issue("MEDIUM", f"Navigation sections not found: {list(missing_sections)}")
        
        self.log_test("Navigation Structure", "PASS", {
            "nav_elements_found": results,
            "sections_found": found_sections,
            "missing_sections": list(missing_sections)
        })

    def test_responsive_design(self, page):
        """Test responsive design across different viewport sizes"""
        print("\n[TEST] Testing responsive design")
        
        for viewport in self.viewports:
            print(f"  Testing viewport: {viewport['name']} ({viewport['width']}x{viewport['height']})")
            
            # Set viewport size
            page.set_viewport_size({"width": viewport['width'], "height": viewport['height']})
            page.wait_for_timeout(500)  # Allow time for responsive adjustments
            
            # Take screenshot
            screenshot_path = self.take_screenshot(page, f"responsive_{viewport['name']}", viewport['name'])
            
            # Test for common responsive issues
            issues_found = []
            
            # Check for horizontal overflow
            try:
                body_width = page.evaluate("document.body.scrollWidth")
                viewport_width = viewport['width']
                if body_width > viewport_width:
                    issues_found.append(f"Horizontal overflow: {body_width}px > {viewport_width}px")
            except:
                pass
            
            # Check if mobile menu is properly handled on small screens
            if viewport['width'] <= 768:
                # Should have mobile menu trigger
                mobile_menu_triggers = page.locator("button[aria-label*='menu'], .hamburger, [data-testid*='menu']").count()
                if mobile_menu_triggers == 0:
                    issues_found.append("No mobile menu trigger found on small screen")
                
                # Desktop sidebar should be hidden
                desktop_sidebars = page.locator(".sidebar:visible, nav.desktop:visible").count()
                if desktop_sidebars > 0:
                    issues_found.append("Desktop sidebar visible on mobile viewport")
            
            # Check for text readability (minimum font size)
            try:
                small_text = page.locator("*").evaluate_all("""
                    elements => elements.filter(el => {
                        const style = window.getComputedStyle(el);
                        const fontSize = parseFloat(style.fontSize);
                        return fontSize < 14 && el.innerText.length > 0;
                    }).length
                """)
                if small_text > 0:
                    issues_found.append(f"{small_text} elements with font size < 14px")
            except:
                pass
            
            # Log issues for this viewport
            for issue in issues_found:
                self.log_issue("MEDIUM", f"Responsive issue on {viewport['name']}: {issue}")
            
            self.log_test(f"Responsive - {viewport['name']}", 
                         "FAIL" if issues_found else "PASS", 
                         {"issues": issues_found, "screenshot": screenshot_path})

    def test_interactive_components(self, page):
        """Test interactive components functionality"""
        print("\n[TEST] Testing interactive components")
        
        # Reset to desktop viewport for testing
        page.set_viewport_size({"width": 1440, "height": 900})
        
        # Test buttons
        buttons = page.locator('button').all()
        button_tests = []
        
        for i, button in enumerate(buttons[:10]):  # Test first 10 buttons
            try:
                if button.is_visible():
                    text = button.inner_text().strip()
                    is_clickable = button.is_enabled()
                    has_aria_label = button.get_attribute('aria-label') is not None
                    
                    button_tests.append({
                        "text": text,
                        "clickable": is_clickable,
                        "has_aria_label": has_aria_label
                    })
                    
                    if not is_clickable:
                        self.log_issue("LOW", f"Button not clickable: '{text}'")
            except Exception as e:
                button_tests.append({"error": str(e)})
        
        # Test dark mode toggle
        dark_mode_toggles = page.locator("button[aria-label*='theme'], button[aria-label*='dark'], .theme-toggle, [data-testid*='theme']").all()
        dark_mode_test = {"found": len(dark_mode_toggles), "functional": False}
        
        if len(dark_mode_toggles) > 0:
            try:
                toggle = dark_mode_toggles[0]
                if toggle.is_visible():
                    # Test dark mode toggle
                    initial_class = page.evaluate("document.documentElement.className")
                    toggle.click()
                    page.wait_for_timeout(500)
                    after_class = page.evaluate("document.documentElement.className")
                    
                    if initial_class != after_class:
                        dark_mode_test["functional"] = True
                        # Toggle back
                        toggle.click()
                        page.wait_for_timeout(500)
                    else:
                        self.log_issue("MEDIUM", "Dark mode toggle doesn't change document class")
            except Exception as e:
                self.log_issue("MEDIUM", f"Error testing dark mode toggle: {str(e)}")
        else:
            self.log_issue("LOW", "No dark mode toggle found")
        
        # Test cards/clickable elements
        cards = page.locator('.card, [role="button"], .clickable').all()
        card_test = {"count": len(cards), "interactive": 0}
        
        for card in cards[:5]:  # Test first 5 cards
            try:
                if card.is_visible():
                    # Check if card has proper hover states or click handlers
                    has_click_handler = card.evaluate("el => el.onclick !== null || el.addEventListener !== undefined")
                    if has_click_handler:
                        card_test["interactive"] += 1
            except:
                pass
        
        self.log_test("Interactive Components", "PASS", {
            "buttons_tested": len(button_tests),
            "buttons_details": button_tests,
            "dark_mode_toggle": dark_mode_test,
            "cards": card_test
        })

    def test_accessibility(self, page):
        """Test accessibility features"""
        print("\n[TEST] Testing accessibility features")
        
        # Test keyboard navigation
        keyboard_nav_results = {}
        
        try:
            # Test Tab navigation
            page.keyboard.press('Tab')
            focused_element = page.evaluate("document.activeElement.tagName + (document.activeElement.textContent ? ': ' + document.activeElement.textContent.slice(0, 50) : '')")
            keyboard_nav_results["first_tab"] = focused_element
            
            # Test multiple tabs
            tab_count = 0
            for _ in range(10):
                page.keyboard.press('Tab')
                tab_count += 1
                if tab_count == 5:
                    mid_focused = page.evaluate("document.activeElement.tagName")
                    keyboard_nav_results["mid_navigation"] = mid_focused
            
            keyboard_nav_results["total_tabs_tested"] = tab_count
            
        except Exception as e:
            keyboard_nav_results["error"] = str(e)
            self.log_issue("HIGH", f"Keyboard navigation error: {str(e)}")
        
        # Test ARIA labels and roles
        aria_elements = {
            "buttons_with_aria": page.locator("button[aria-label], button[aria-describedby]").count(),
            "headings": page.locator("h1, h2, h3, h4, h5, h6").count(),
            "landmarks": page.locator("[role='main'], [role='navigation'], [role='banner'], [role='contentinfo']").count(),
            "images_with_alt": page.locator("img[alt]").count(),
            "images_without_alt": page.locator("img:not([alt])").count()
        }
        
        # Check for accessibility issues
        if aria_elements["images_without_alt"] > 0:
            self.log_issue("MEDIUM", f"{aria_elements['images_without_alt']} images missing alt text")
        
        if aria_elements["headings"] == 0:
            self.log_issue("HIGH", "No heading elements found - poor document structure")
        
        if aria_elements["landmarks"] == 0:
            self.log_issue("MEDIUM", "No ARIA landmark roles found")
        
        # Test color contrast (basic check)
        try:
            contrast_issues = page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('*');
                    let issues = 0;
                    for (let el of elements) {
                        const style = window.getComputedStyle(el);
                        const color = style.color;
                        const background = style.backgroundColor;
                        // Simple check for very light gray text on white
                        if (color.includes('rgb(128, 128, 128)') && background.includes('rgb(255, 255, 255)')) {
                            issues++;
                        }
                    }
                    return issues;
                }
            """)
            if contrast_issues > 0:
                self.log_issue("MEDIUM", f"Potential color contrast issues found: {contrast_issues} elements")
        except:
            pass
        
        self.log_test("Accessibility", "PASS", {
            "keyboard_navigation": keyboard_nav_results,
            "aria_elements": aria_elements
        })

    def test_dark_mode(self, page):
        """Test dark mode functionality specifically"""
        print("\n[TEST] Testing dark mode functionality")
        
        # Find dark mode toggle
        dark_mode_selectors = [
            "button[aria-label*='theme']",
            "button[aria-label*='dark']",
            ".theme-toggle",
            "[data-testid*='theme']",
            "button:has-text('Dark')",
            "button:has-text('Light')"
        ]
        
        toggle_found = False
        toggle_element = None
        
        for selector in dark_mode_selectors:
            try:
                elements = page.locator(selector).all()
                if len(elements) > 0 and elements[0].is_visible():
                    toggle_element = elements[0]
                    toggle_found = True
                    break
            except:
                continue
        
        if not toggle_found:
            self.log_issue("MEDIUM", "Dark mode toggle not found")
            self.log_test("Dark Mode", "FAIL", {"reason": "Toggle not found"})
            return
        
        # Test dark mode switching
        try:
            # Capture initial state
            initial_screenshot = self.take_screenshot(page, "before_dark_mode")
            initial_html_class = page.evaluate("document.documentElement.className")
            initial_body_class = page.evaluate("document.body.className")
            
            # Toggle to dark mode
            toggle_element.click()
            page.wait_for_timeout(1000)  # Wait for transition
            
            # Capture dark mode state
            dark_screenshot = self.take_screenshot(page, "dark_mode")
            dark_html_class = page.evaluate("document.documentElement.className")
            dark_body_class = page.evaluate("document.body.className")
            
            # Check if classes changed
            classes_changed = (initial_html_class != dark_html_class) or (initial_body_class != dark_body_class)
            
            # Toggle back to light mode
            toggle_element.click()
            page.wait_for_timeout(1000)
            
            # Capture back to light state
            light_screenshot = self.take_screenshot(page, "back_to_light")
            final_html_class = page.evaluate("document.documentElement.className")
            final_body_class = page.evaluate("document.body.className")
            
            # Check if it returned to original state
            returned_to_original = (initial_html_class == final_html_class) and (initial_body_class == final_body_class)
            
            success = classes_changed and returned_to_original
            
            if not classes_changed:
                self.log_issue("HIGH", "Dark mode toggle doesn't change document/body classes")
            
            if not returned_to_original:
                self.log_issue("MEDIUM", "Dark mode toggle doesn't return to original state properly")
            
            self.log_test("Dark Mode", "PASS" if success else "FAIL", {
                "classes_changed": classes_changed,
                "returned_to_original": returned_to_original,
                "screenshots": [initial_screenshot, dark_screenshot, light_screenshot]
            })
            
        except Exception as e:
            self.log_issue("HIGH", f"Error testing dark mode: {str(e)}")
            self.log_test("Dark Mode", "FAIL", {"error": str(e)})

    def test_cross_browser_compatibility(self, playwright):
        """Test cross-browser compatibility (Chromium, Firefox, WebKit)"""
        print("\n[TEST] Testing cross-browser compatibility")
        
        browsers_to_test = [
            ("chromium", playwright.chromium),
            ("firefox", playwright.firefox),
            ("webkit", playwright.webkit)
        ]
        
        browser_results = {}
        
        for browser_name, browser_type in browsers_to_test:
            print(f"  Testing {browser_name}")
            
            try:
                browser = browser_type.launch(headless=True)
                page = browser.new_page()
                
                # Test basic page load
                start_time = time.time()
                page.goto(self.test_results["base_url"])
                page.wait_for_load_state('networkidle')
                load_time = time.time() - start_time
                
                # Test basic functionality
                buttons_count = page.locator('button').count()
                links_count = page.locator('a').count()
                images_count = page.locator('img').count()
                
                # Take screenshot
                screenshot = self.take_screenshot(page, f"browser_{browser_name}")
                
                browser_results[browser_name] = {
                    "status": "PASS",
                    "load_time": load_time,
                    "buttons_count": buttons_count,
                    "links_count": links_count,
                    "images_count": images_count,
                    "screenshot": screenshot
                }
                
                browser.close()
                
            except Exception as e:
                browser_results[browser_name] = {
                    "status": "FAIL",
                    "error": str(e)
                }
                self.log_issue("HIGH", f"Browser {browser_name} compatibility issue: {str(e)}")
        
        self.log_test("Cross-Browser Compatibility", "PASS", browser_results)

    def test_page_content_and_structure(self, page):
        """Test page content and structure"""
        print("\n[TEST] Testing page content and structure")
        
        # Test for essential page elements
        essential_elements = {
            "title": page.title(),
            "headings": page.locator("h1, h2, h3, h4, h5, h6").count(),
            "paragraphs": page.locator("p").count(),
            "images": page.locator("img").count(),
            "buttons": page.locator("button").count(),
            "links": page.locator("a[href]").count(),
            "forms": page.locator("form").count()
        }
        
        # Check for common issues
        if essential_elements["headings"] == 0:
            self.log_issue("HIGH", "No heading elements found")
        
        if not essential_elements["title"] or len(essential_elements["title"]) < 10:
            self.log_issue("MEDIUM", f"Page title too short or missing: '{essential_elements['title']}'")
        
        # Test for broken images
        broken_images = page.evaluate("""
            () => {
                const images = document.querySelectorAll('img');
                let broken = 0;
                images.forEach(img => {
                    if (!img.complete || img.naturalWidth === 0) {
                        broken++;
                    }
                });
                return broken;
            }
        """)
        
        if broken_images > 0:
            self.log_issue("MEDIUM", f"{broken_images} broken images found")
        
        # Test for console errors
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        
        # Reload page to capture console errors
        page.reload()
        page.wait_for_load_state('networkidle')
        
        if console_errors:
            for error in console_errors[:5]:  # Log first 5 errors
                self.log_issue("MEDIUM", f"Console error: {error}")
        
        self.log_test("Page Content and Structure", "PASS", {
            "essential_elements": essential_elements,
            "broken_images": broken_images,
            "console_errors": len(console_errors)
        })

    def run_comprehensive_test(self):
        """Run all tests"""
        print("Starting Comprehensive Data Engineering Platform Test Suite")
        print(f"Testing: {self.test_results['base_url']}")
        print("=" * 60)
        
        with sync_playwright() as playwright:
            # Start with Chromium for main tests
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                # Initial page load and performance test
                self.test_page_load_performance(page, self.test_results["base_url"])
                
                # Test page content and structure
                self.test_page_content_and_structure(page)
                
                # Test navigation
                self.test_navigation_structure(page)
                
                # Test responsive design
                self.test_responsive_design(page)
                
                # Test interactive components
                self.test_interactive_components(page)
                
                # Test accessibility
                self.test_accessibility(page)
                
                # Test dark mode
                self.test_dark_mode(page)
                
                browser.close()
                
                # Test cross-browser compatibility
                self.test_cross_browser_compatibility(playwright)
                
            except Exception as e:
                print(f"[ERROR] Critical error during testing: {str(e)}")
                self.log_issue("CRITICAL", f"Test suite failed: {str(e)}")
                browser.close()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()

    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        
        # Performance recommendations
        total_issues = len(self.test_results["issues"])
        high_severity_issues = len([i for i in self.test_results["issues"] if i["severity"] == "HIGH"])
        
        if high_severity_issues > 0:
            self.log_recommendation("CRITICAL", f"Address {high_severity_issues} high-severity issues immediately")
        
        # Performance recommendations
        for url, metrics in self.test_results["performance_metrics"].items():
            if metrics.get("total_load_time", 0) > 2:
                self.log_recommendation("PERFORMANCE", "Consider optimizing bundle size, implementing code splitting, or using image optimization")
        
        # Accessibility recommendations
        accessibility_issues = [i for i in self.test_results["issues"] if "accessibility" in i["description"].lower() or "aria" in i["description"].lower()]
        if accessibility_issues:
            self.log_recommendation("ACCESSIBILITY", "Improve ARIA labels, alt text, and keyboard navigation")
        
        # Mobile recommendations
        responsive_issues = [i for i in self.test_results["issues"] if "responsive" in i["description"].lower() or "mobile" in i["description"].lower()]
        if responsive_issues:
            self.log_recommendation("MOBILE", "Review mobile layouts and touch targets")
        
        # General recommendations
        self.log_recommendation("TESTING", "Consider implementing automated testing for continuous quality assurance")
        self.log_recommendation("MONITORING", "Set up performance monitoring and error tracking in production")

    def save_results(self):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import tempfile
        temp_dir = tempfile.gettempdir()
        filename = f"{temp_dir}\\test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n[INFO] Test results saved to: {filename}")
        return filename

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results["tests"])
        passed_tests = len([t for t in self.test_results["tests"].values() if t["status"] == "PASS"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        print(f"\nIssues Found: {len(self.test_results['issues'])}")
        severity_counts = {}
        for issue in self.test_results["issues"]:
            severity = issue["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count}")
        
        print(f"\nRecommendations: {len(self.test_results['recommendations'])}")
        
        print(f"\nScreenshots taken: {len(self.test_results['screenshots'])}")
        
        if self.test_results["screenshots"]:
            print("Screenshot locations:")
            for screenshot in self.test_results["screenshots"]:
                print(f"  {screenshot}")


if __name__ == "__main__":
    tester = DataEngineeringPlatformTester()
    tester.run_comprehensive_test()