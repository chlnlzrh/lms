#!/usr/bin/env python3
"""
Test script for AI Learning Platform new features implementation
Tests the 10-point better design improvements
"""

import asyncio
from playwright.async_api import async_playwright
import json
import time

async def test_ai_learning_platform():
    """Test all new features of the AI Learning Platform"""
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": [],
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0
        }
    }
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            print("Testing AI Learning Platform New Features")
            print("=" * 60)
            
            # Test 1: Homepage loads with new typography
            print("Test 1: Homepage loads with enhanced typography...")
            await page.goto('http://localhost:3000')
            await page.wait_for_load_state('networkidle')
            
            # Check for display typography
            display_heading = await page.locator('h1.text-display').count()
            test_result = {
                "name": "Enhanced Typography - Display Heading",
                "passed": display_heading > 0,
                "details": f"Found {display_heading} display headings"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Display typography: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 2: Progressive Dashboard exists
            print("Test 2: Progressive dashboard accessible...")
            dashboard_link = await page.locator('a[href="/dashboard"]').count()
            test_result = {
                "name": "Progressive Dashboard Link",
                "passed": dashboard_link > 0,
                "details": f"Found {dashboard_link} dashboard links"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Dashboard link: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 3: Smart Search functionality
            print("Test 3: Smart search component...")
            search_component = await page.locator('input[placeholder*="Search"]').count()
            test_result = {
                "name": "Smart Search Component",
                "passed": search_component > 0,
                "details": f"Found {search_component} search inputs"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Search component: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 4: Progress Overview component
            print("Test 4: Progress tracking components...")
            progress_bars = await page.locator('.progress-bar').count()
            test_result = {
                "name": "Progress Tracking",
                "passed": progress_bars > 0,
                "details": f"Found {progress_bars} progress bars"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Progress tracking: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 5: Enhanced cards and buttons
            print("Test 5: Enhanced UI components...")
            cards = await page.locator('.card').count()
            primary_buttons = await page.locator('.btn-primary').count()
            test_result = {
                "name": "Enhanced UI Components",
                "passed": cards > 0 and primary_buttons > 0,
                "details": f"Found {cards} cards and {primary_buttons} primary buttons"
            }
            results["tests"].append(test_result)
            print(f"   [OK] UI components: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 6: Navigate to dashboard
            print("Test 6: Dashboard page navigation...")
            await page.click('a[href="/dashboard"]')
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(1000)  # Wait for animations
            
            current_url = page.url
            dashboard_loaded = '/dashboard' in current_url
            dashboard_content = await page.locator('h1:has-text("Learning Dashboard")').count()
            
            test_result = {
                "name": "Dashboard Navigation",
                "passed": dashboard_loaded and dashboard_content > 0,
                "details": f"URL: {current_url}, Dashboard title found: {dashboard_content > 0}"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Dashboard navigation: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 7: Dashboard features
            print("Test 7: Dashboard analytics features...")
            stats_cards = await page.locator('.card').count()
            recommendations = await page.locator('text=/Recommended for You|Recent Activity/').count()
            
            test_result = {
                "name": "Dashboard Analytics",
                "passed": stats_cards >= 3 and recommendations > 0,
                "details": f"Stats cards: {stats_cards}, Recommendation sections: {recommendations}"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Dashboard analytics: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 8: Mobile-responsive design
            print("Test 8: Mobile responsiveness...")
            await page.set_viewport_size({"width": 375, "height": 667})  # Mobile size
            await page.wait_for_timeout(500)
            
            # Check if mobile navigation exists
            mobile_nav = await page.locator('button[aria-label*="menu" i]').count()
            test_result = {
                "name": "Mobile Responsiveness",
                "passed": mobile_nav > 0,
                "details": f"Mobile navigation elements: {mobile_nav}"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Mobile responsive: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 9: Animation classes
            print("Test 9: Animation and design system...")
            await page.set_viewport_size({"width": 1200, "height": 800})  # Back to desktop
            await page.goto('http://localhost:3000')
            await page.wait_for_load_state('networkidle')
            
            animated_elements = await page.locator('.animate-fade-in').count()
            test_result = {
                "name": "Animation System",
                "passed": animated_elements > 0,
                "details": f"Animated elements: {animated_elements}"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Animation system: {'PASS' if test_result['passed'] else 'FAIL'}")
            
            # Test 10: Typography hierarchy
            print("Test 10: Typography hierarchy...")
            headings = {
                "display": await page.locator('.text-display').count(),
                "heading-1": await page.locator('.text-heading-1').count(),
                "heading-2": await page.locator('.text-heading-2').count(),
                "heading-3": await page.locator('.text-heading-3').count(),
                "body": await page.locator('.text-body').count()
            }
            
            typography_implemented = sum(headings.values()) > 0
            test_result = {
                "name": "Typography Hierarchy",
                "passed": typography_implemented,
                "details": f"Typography classes found: {headings}"
            }
            results["tests"].append(test_result)
            print(f"   [OK] Typography hierarchy: {'PASS' if test_result['passed'] else 'FAIL'}")
            
        except Exception as e:
            print(f"[ERROR] Error during testing: {str(e)}")
            test_result = {
                "name": "Testing Error",
                "passed": False,
                "details": str(e)
            }
            results["tests"].append(test_result)
        
        finally:
            await browser.close()
    
    # Calculate summary
    results["summary"]["total"] = len(results["tests"])
    results["summary"]["passed"] = sum(1 for test in results["tests"] if test["passed"])
    results["summary"]["failed"] = results["summary"]["total"] - results["summary"]["passed"]
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Success Rate: {(results['summary']['passed'] / results['summary']['total'] * 100):.1f}%")
    
    # Save results
    with open(f"new_features_test_report_{int(time.time())}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: new_features_test_report_{int(time.time())}.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_ai_learning_platform())