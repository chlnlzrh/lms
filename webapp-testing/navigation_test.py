#!/usr/bin/env python3
"""
Navigation Test - Check if menu items click and open correct pages
Tests the navigation functionality of the Data Engineering Learning Platform
"""

from playwright.sync_api import sync_playwright
import time
import sys

def test_navigation_clicks(base_url="http://localhost:3006"):
    """Test that menu items click and navigate to correct pages"""
    
    print("=" * 60)
    print("NAVIGATION TESTING - Menu Item Clicks")
    print("=" * 60)
    print(f"Testing: {base_url}")
    
    results = {
        "successful_navigations": [],
        "failed_navigations": [],
        "menu_items_found": 0,
        "total_tests": 0
    }
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Visible browser for debugging
        page = browser.new_page()
        
        try:
            # Navigate to homepage
            print("\n[STEP 1] Loading homepage...")
            page.goto(base_url, timeout=30000)
            page.wait_for_load_state('networkidle', timeout=30000)
            print("✅ Homepage loaded successfully")
            
            # Test main navigation items
            main_nav_items = [
                {"text": "My Dashboard", "expected_url": "/", "selector": "text='My Dashboard'"},
                {"text": "Learning Path", "expected_url": "/learning-path", "selector": "text='Learning Path'"},
                {"text": "All Lessons", "expected_url": "/lessons", "selector": "text='All Lessons'"},
                {"text": "Projects", "expected_url": "/projects", "selector": "text='Projects'"},
                {"text": "Community", "expected_url": "/community", "selector": "text='Community'"},
                {"text": "Resource Library", "expected_url": "/resources", "selector": "text='Resource Library'"}
            ]
            
            print(f"\n[STEP 2] Testing {len(main_nav_items)} main navigation items...")
            
            for item in main_nav_items:
                results["total_tests"] += 1
                print(f"\n--- Testing: {item['text']} ---")
                
                try:
                    # Look for the navigation item
                    nav_element = page.locator(item["selector"]).first
                    
                    if nav_element.is_visible():
                        results["menu_items_found"] += 1
                        print(f"✅ Found menu item: {item['text']}")
                        
                        # Click the navigation item
                        nav_element.click()
                        page.wait_for_load_state('networkidle', timeout=10000)
                        
                        # Check if we navigated to the expected URL
                        current_url = page.url
                        expected_path = item["expected_url"]
                        
                        if current_url.endswith(expected_path) or expected_path in current_url:
                            print(f"✅ Successfully navigated to: {current_url}")
                            results["successful_navigations"].append({
                                "item": item["text"],
                                "expected": expected_path,
                                "actual": current_url,
                                "status": "SUCCESS"
                            })
                            
                            # Take a screenshot of the page
                            screenshot_name = f"nav_test_{item['text'].replace(' ', '_').lower()}.png"
                            page.screenshot(path=f"C:\\Users\\bimal\\AppData\\Local\\Temp\\{screenshot_name}")
                            print(f"📸 Screenshot saved: {screenshot_name}")
                            
                        else:
                            print(f"❌ Navigation failed - Expected: {expected_path}, Got: {current_url}")
                            results["failed_navigations"].append({
                                "item": item["text"],
                                "expected": expected_path,
                                "actual": current_url,
                                "status": "WRONG_URL"
                            })
                    else:
                        print(f"❌ Menu item not found: {item['text']}")
                        results["failed_navigations"].append({
                            "item": item["text"],
                            "expected": item["expected_url"],
                            "actual": "NOT_FOUND",
                            "status": "NOT_FOUND"
                        })
                
                except Exception as e:
                    print(f"❌ Error testing {item['text']}: {str(e)}")
                    results["failed_navigations"].append({
                        "item": item["text"],
                        "expected": item["expected_url"],
                        "actual": f"ERROR: {str(e)}",
                        "status": "ERROR"
                    })
            
            # Test module navigation (dropdown items)
            print(f"\n[STEP 3] Testing module navigation...")
            
            # Go back to homepage or learning path to test module navigation
            page.goto(f"{base_url}/learning-path", timeout=30000)
            page.wait_for_load_state('networkidle', timeout=30000)
            
            # Test a few module links
            module_tests = [
                {"text": "Module 1", "expected": "/learning-path/module-1"},
                {"text": "Module 2", "expected": "/learning-path/module-2"},
                {"text": "Module 5", "expected": "/learning-path/module-5"}
            ]
            
            for module in module_tests:
                results["total_tests"] += 1
                print(f"\n--- Testing Module Navigation: {module['text']} ---")
                
                try:
                    # Look for module link in navigation or on page
                    module_selector = f"text='{module['text']}', a[href*='{module['expected']}']"
                    module_element = page.locator(module_selector).first
                    
                    if module_element.is_visible():
                        print(f"✅ Found module link: {module['text']}")
                        module_element.click()
                        page.wait_for_load_state('networkidle', timeout=10000)
                        
                        current_url = page.url
                        if module["expected"] in current_url:
                            print(f"✅ Module navigation successful: {current_url}")
                            results["successful_navigations"].append({
                                "item": f"Module Navigation - {module['text']}",
                                "expected": module["expected"],
                                "actual": current_url,
                                "status": "SUCCESS"
                            })
                        else:
                            print(f"❌ Module navigation failed - Expected: {module['expected']}, Got: {current_url}")
                            results["failed_navigations"].append({
                                "item": f"Module Navigation - {module['text']}",
                                "expected": module["expected"],
                                "actual": current_url,
                                "status": "WRONG_URL"
                            })
                    else:
                        print(f"❌ Module link not found: {module['text']}")
                        results["failed_navigations"].append({
                            "item": f"Module Navigation - {module['text']}",
                            "expected": module["expected"],
                            "actual": "NOT_FOUND",
                            "status": "NOT_FOUND"
                        })
                
                except Exception as e:
                    print(f"❌ Error testing module {module['text']}: {str(e)}")
                    results["failed_navigations"].append({
                        "item": f"Module Navigation - {module['text']}",
                        "expected": module["expected"],
                        "actual": f"ERROR: {str(e)}",
                        "status": "ERROR"
                    })
                
                # Navigate back to learning path for next test
                page.goto(f"{base_url}/learning-path", timeout=30000)
                page.wait_for_load_state('networkidle', timeout=10000)
            
        except Exception as e:
            print(f"❌ Critical error during navigation testing: {str(e)}")
        
        finally:
            browser.close()
    
    # Print results summary
    print("\n" + "=" * 60)
    print("NAVIGATION TEST RESULTS")
    print("=" * 60)
    
    success_rate = (len(results["successful_navigations"]) / results["total_tests"] * 100) if results["total_tests"] > 0 else 0
    
    print(f"Total Tests: {results['total_tests']}")
    print(f"Menu Items Found: {results['menu_items_found']}")
    print(f"Successful Navigations: {len(results['successful_navigations'])}")
    print(f"Failed Navigations: {len(results['failed_navigations'])}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if results["successful_navigations"]:
        print(f"\n✅ SUCCESSFUL NAVIGATIONS:")
        for nav in results["successful_navigations"]:
            print(f"  • {nav['item']}: {nav['expected']} → {nav['actual']}")
    
    if results["failed_navigations"]:
        print(f"\n❌ FAILED NAVIGATIONS:")
        for nav in results["failed_navigations"]:
            print(f"  • {nav['item']}: {nav['status']} - {nav['actual']}")
    
    print(f"\nOverall Status: {'✅ PASS' if success_rate >= 80 else '⚠️ NEEDS ATTENTION' if success_rate >= 60 else '❌ FAIL'}")
    
    return results

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3006"
    test_navigation_clicks(base_url)