#!/usr/bin/env python3
"""
Simple Navigation Test - Check if menu items work correctly
Tests navigation without Unicode characters for Windows compatibility
"""

from playwright.sync_api import sync_playwright
import time
import sys

def test_navigation_simple(base_url="http://localhost:3008"):
    """Test basic navigation functionality"""
    
    print("=" * 60)
    print("NAVIGATION TESTING - Menu Item Clicks")
    print("=" * 60)
    print(f"Testing: {base_url}")
    
    results = {
        "successful": [],
        "failed": [],
        "total": 0
    }
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible browser
        page = browser.new_page()
        
        try:
            # Load homepage
            print("\n[STEP 1] Loading homepage...")
            page.goto(base_url, timeout=30000)
            page.wait_for_load_state('networkidle', timeout=30000)
            print("PASS: Homepage loaded successfully")
            
            # Test main navigation items
            nav_tests = [
                {"name": "Dashboard", "url": "/", "selector": "text='My Dashboard'"},
                {"name": "Learning Path", "url": "/learning-path", "selector": "text='Learning Path'"},
                {"name": "Lessons", "url": "/lessons", "selector": "text='All Lessons'"},
                {"name": "Projects", "url": "/projects", "selector": "text='Projects'"},
                {"name": "Community", "url": "/community", "selector": "text='Community'"}
            ]
            
            print(f"\n[STEP 2] Testing {len(nav_tests)} navigation items...")
            
            for nav in nav_tests:
                results["total"] += 1
                print(f"\n--- Testing: {nav['name']} ---")
                
                try:
                    # Look for navigation item
                    nav_element = page.locator(nav["selector"]).first
                    
                    if nav_element.is_visible():
                        print(f"FOUND: {nav['name']} menu item")
                        
                        # Click navigation item
                        nav_element.click()
                        page.wait_for_load_state('networkidle', timeout=10000)
                        
                        # Check URL
                        current_url = page.url
                        if nav["url"] in current_url or current_url.endswith(nav["url"]):
                            print(f"PASS: Navigation to {current_url}")
                            results["successful"].append({
                                "name": nav["name"],
                                "expected": nav["url"],
                                "actual": current_url
                            })
                            
                            # Take screenshot
                            screenshot_name = f"nav_{nav['name'].lower().replace(' ', '_')}.png"
                            page.screenshot(path=f"C:\\Users\\bimal\\AppData\\Local\\Temp\\{screenshot_name}")
                            print(f"Screenshot: {screenshot_name}")
                            
                        else:
                            print(f"FAIL: Wrong URL - Expected {nav['url']}, Got {current_url}")
                            results["failed"].append({
                                "name": nav["name"],
                                "expected": nav["url"],
                                "actual": current_url,
                                "error": "Wrong URL"
                            })
                    else:
                        print(f"FAIL: {nav['name']} menu item not found")
                        results["failed"].append({
                            "name": nav["name"],
                            "expected": nav["url"],
                            "actual": "NOT_FOUND",
                            "error": "Menu item not visible"
                        })
                
                except Exception as e:
                    print(f"ERROR: {nav['name']} - {str(e)}")
                    results["failed"].append({
                        "name": nav["name"],
                        "expected": nav["url"],
                        "actual": "ERROR",
                        "error": str(e)
                    })
            
            # Test module navigation
            print(f"\n[STEP 3] Testing module pages...")
            
            # Go to learning path first
            page.goto(f"{base_url}/learning-path", timeout=30000)
            page.wait_for_load_state('networkidle', timeout=30000)
            
            # Test module links
            module_tests = [
                {"name": "Module 1", "url": "/learning-path/module-1"},
                {"name": "Module 2", "url": "/learning-path/module-2"}
            ]
            
            for module in module_tests:
                results["total"] += 1
                print(f"\n--- Testing: {module['name']} ---")
                
                try:
                    # Look for module link
                    module_link = page.locator(f"a[href='{module['url']}']").first
                    
                    if module_link.is_visible():
                        print(f"FOUND: {module['name']} link")
                        
                        module_link.click()
                        page.wait_for_load_state('networkidle', timeout=10000)
                        
                        current_url = page.url
                        if module["url"] in current_url:
                            print(f"PASS: Module page loaded - {current_url}")
                            results["successful"].append({
                                "name": f"Module Navigation - {module['name']}",
                                "expected": module["url"],
                                "actual": current_url
                            })
                        else:
                            print(f"FAIL: Module navigation failed - {current_url}")
                            results["failed"].append({
                                "name": f"Module Navigation - {module['name']}",
                                "expected": module["url"],
                                "actual": current_url,
                                "error": "Wrong URL"
                            })
                    else:
                        print(f"FAIL: {module['name']} link not found")
                        results["failed"].append({
                            "name": f"Module Navigation - {module['name']}",
                            "expected": module["url"],
                            "actual": "NOT_FOUND",
                            "error": "Link not visible"
                        })
                
                except Exception as e:
                    print(f"ERROR: {module['name']} - {str(e)}")
                    results["failed"].append({
                        "name": f"Module Navigation - {module['name']}",
                        "expected": module["url"],
                        "actual": "ERROR",
                        "error": str(e)
                    })
                
                # Go back to learning path
                page.goto(f"{base_url}/learning-path", timeout=30000)
                page.wait_for_load_state('networkidle', timeout=10000)
            
        except Exception as e:
            print(f"CRITICAL ERROR: {str(e)}")
        
        finally:
            browser.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("NAVIGATION TEST RESULTS")
    print("=" * 60)
    
    success_count = len(results["successful"])
    fail_count = len(results["failed"])
    total = results["total"]
    success_rate = (success_count / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if results["successful"]:
        print(f"\nSUCCESSFUL NAVIGATIONS:")
        for nav in results["successful"]:
            print(f"  - {nav['name']}: {nav['expected']} -> {nav['actual']}")
    
    if results["failed"]:
        print(f"\nFAILED NAVIGATIONS:")
        for nav in results["failed"]:
            print(f"  - {nav['name']}: {nav['error']} - {nav['actual']}")
    
    status = "PASS" if success_rate >= 80 else "NEEDS ATTENTION" if success_rate >= 60 else "FAIL"
    print(f"\nOverall Status: {status}")
    
    return results

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3008"
    test_navigation_simple(base_url)