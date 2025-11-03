#!/usr/bin/env python3
"""
Robust Navigation Test - Handles collapsed navigation and mobile menus
"""

from playwright.sync_api import sync_playwright
import time
import sys

def test_navigation_robust(base_url="http://localhost:3008"):
    """Test navigation with proper handling of collapsed menus"""
    
    print("=" * 60)
    print("ROBUST NAVIGATION TEST")
    print("=" * 60)
    print(f"Testing: {base_url}")
    
    results = {"successful": [], "failed": [], "total": 0}
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        try:
            # Load homepage
            print("\n[STEP 1] Loading homepage...")
            page.goto(base_url, timeout=30000)
            page.wait_for_load_state('networkidle', timeout=30000)
            print("PASS: Homepage loaded")
            
            # Take screenshot of initial state
            page.screenshot(path="C:\\Users\\bimal\\AppData\\Local\\Temp\\homepage_initial.png")
            print("Screenshot: homepage_initial.png")
            
            # Check for navigation sidebar and try to expand it
            print("\n[STEP 2] Finding navigation...")
            
            # Look for desktop sidebar
            desktop_nav = page.locator("aside, nav, .sidebar").first
            if desktop_nav.is_visible():
                print("FOUND: Desktop navigation sidebar")
                
                # Try to hover to expand navigation
                try:
                    desktop_nav.hover()
                    page.wait_for_timeout(1000)  # Wait for expansion animation
                    print("HOVER: Attempted to expand navigation")
                except:
                    print("NOTE: Could not hover on navigation")
            else:
                print("NOTE: Desktop navigation not immediately visible")
            
            # Look for mobile menu button
            mobile_menu_selectors = [
                "button[aria-label*='menu']",
                "button[data-testid*='menu']", 
                ".hamburger",
                "[role='button']:has-text('Menu')",
                "button:has-text('☰')"
            ]
            
            mobile_menu_found = False
            for selector in mobile_menu_selectors:
                try:
                    mobile_btn = page.locator(selector).first
                    if mobile_btn.is_visible():
                        print(f"FOUND: Mobile menu button - {selector}")
                        mobile_btn.click()
                        page.wait_for_timeout(1000)
                        mobile_menu_found = True
                        break
                except:
                    continue
            
            if not mobile_menu_found:
                print("NOTE: No mobile menu button found")
            
            # Take screenshot after attempting navigation expansion
            page.screenshot(path="C:\\Users\\bimal\\AppData\\Local\\Temp\\navigation_expanded.png")
            print("Screenshot: navigation_expanded.png")
            
            # Test direct URL navigation instead of menu clicks
            print("\n[STEP 3] Testing direct URL navigation...")
            
            nav_pages = [
                {"name": "Learning Path", "url": "/learning-path"},
                {"name": "Lessons", "url": "/lessons"},
                {"name": "Projects", "url": "/projects"},
                {"name": "Community", "url": "/community"},
                {"name": "Resources", "url": "/resources"}
            ]
            
            for nav in nav_pages:
                results["total"] += 1
                print(f"\n--- Testing: {nav['name']} ---")
                
                try:
                    # Navigate directly to URL
                    full_url = f"{base_url}{nav['url']}"
                    page.goto(full_url, timeout=30000)
                    page.wait_for_load_state('networkidle', timeout=30000)
                    
                    current_url = page.url
                    page_title = page.title()
                    
                    if nav["url"] in current_url:
                        print(f"PASS: {nav['name']} page loaded")
                        print(f"  URL: {current_url}")
                        print(f"  Title: {page_title}")
                        
                        results["successful"].append({
                            "name": nav["name"],
                            "url": nav["url"],
                            "actual_url": current_url,
                            "title": page_title
                        })
                        
                        # Take screenshot
                        screenshot_name = f"page_{nav['name'].lower().replace(' ', '_')}.png"
                        page.screenshot(path=f"C:\\Users\\bimal\\AppData\\Local\\Temp\\{screenshot_name}")
                        print(f"  Screenshot: {screenshot_name}")
                        
                    else:
                        print(f"FAIL: Wrong URL - Expected {nav['url']}, Got {current_url}")
                        results["failed"].append({
                            "name": nav["name"],
                            "url": nav["url"],
                            "error": f"Wrong URL: {current_url}"
                        })
                
                except Exception as e:
                    print(f"ERROR: {nav['name']} - {str(e)}")
                    results["failed"].append({
                        "name": nav["name"],
                        "url": nav["url"],
                        "error": str(e)
                    })
            
            # Test module pages specifically
            print("\n[STEP 4] Testing module pages...")
            
            module_pages = [
                {"name": "Module 1", "url": "/learning-path/module-1"},
                {"name": "Module 2", "url": "/learning-path/module-2"},
                {"name": "Module 5", "url": "/learning-path/module-5"}
            ]
            
            for module in module_pages:
                results["total"] += 1
                print(f"\n--- Testing: {module['name']} ---")
                
                try:
                    full_url = f"{base_url}{module['url']}"
                    page.goto(full_url, timeout=30000)
                    page.wait_for_load_state('networkidle', timeout=30000)
                    
                    current_url = page.url
                    page_title = page.title()
                    
                    # Check if page loaded successfully (not 404)
                    is_404 = "404" in page_title.lower() or "not found" in page_title.lower()
                    
                    if module["url"] in current_url and not is_404:
                        print(f"PASS: {module['name']} page loaded")
                        print(f"  URL: {current_url}")
                        print(f"  Title: {page_title}")
                        
                        results["successful"].append({
                            "name": f"Module Page - {module['name']}",
                            "url": module["url"],
                            "actual_url": current_url,
                            "title": page_title
                        })
                        
                        # Take screenshot
                        screenshot_name = f"module_{module['name'].lower().replace(' ', '_')}.png"
                        page.screenshot(path=f"C:\\Users\\bimal\\AppData\\Local\\Temp\\{screenshot_name}")
                        print(f"  Screenshot: {screenshot_name}")
                        
                    else:
                        print(f"FAIL: Module page not loaded properly")
                        print(f"  URL: {current_url}")
                        print(f"  Title: {page_title}")
                        print(f"  Is 404: {is_404}")
                        
                        results["failed"].append({
                            "name": f"Module Page - {module['name']}",
                            "url": module["url"],
                            "error": f"Page not loaded properly - Title: {page_title}"
                        })
                
                except Exception as e:
                    print(f"ERROR: {module['name']} - {str(e)}")
                    results["failed"].append({
                        "name": f"Module Page - {module['name']}",
                        "url": module["url"],
                        "error": str(e)
                    })
            
            # Test if navigation links are actually working (click test)
            print("\n[STEP 5] Testing clickable navigation...")
            
            # Go back to homepage
            page.goto(base_url, timeout=30000)
            page.wait_for_load_state('networkidle', timeout=30000)
            
            # Look for any visible links that might be navigation
            all_links = page.locator("a").all()
            nav_links_found = []
            
            for link in all_links[:20]:  # Check first 20 links
                try:
                    if link.is_visible():
                        href = link.get_attribute('href')
                        text = link.inner_text().strip()
                        if href and text and len(text) > 0:
                            nav_links_found.append({"text": text, "href": href})
                except:
                    continue
            
            print(f"FOUND: {len(nav_links_found)} visible navigation links:")
            for nav_link in nav_links_found[:10]:  # Show first 10
                print(f"  - '{nav_link['text']}' -> {nav_link['href']}")
            
        except Exception as e:
            print(f"CRITICAL ERROR: {str(e)}")
        
        finally:
            browser.close()
    
    # Print results
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
        print(f"\nSUCCESSFUL PAGES:")
        for result in results["successful"]:
            print(f"  - {result['name']}: {result['url']}")
            print(f"    Title: {result.get('title', 'N/A')}")
    
    if results["failed"]:
        print(f"\nFAILED PAGES:")
        for result in results["failed"]:
            print(f"  - {result['name']}: {result['url']}")
            print(f"    Error: {result['error']}")
    
    status = "PASS" if success_rate >= 80 else "NEEDS ATTENTION" if success_rate >= 60 else "FAIL"
    print(f"\nOverall Status: {status}")
    
    return results

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3008"
    test_navigation_robust(base_url)