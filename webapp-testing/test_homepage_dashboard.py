#!/usr/bin/env python3
"""
Test the homepage as dashboard implementation
"""

import asyncio
from playwright.async_api import async_playwright

async def test_homepage_dashboard():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        try:
            print("Testing Homepage as Dashboard...")
            print("=" * 50)
            
            # Go directly to homepage
            print("1. Loading homepage (should be dashboard)...")
            await page.goto('http://localhost:3000')
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(2000)
            
            # Check for dashboard title
            print("2. Checking dashboard content on homepage...")
            dashboard_title = await page.locator('h1:has-text("Learning Dashboard")').count()
            display_headings = await page.locator('.text-display').count()
            
            print(f"   Dashboard title 'Learning Dashboard': {dashboard_title}")
            print(f"   Display typography elements: {display_headings}")
            
            # Check page content
            content = await page.content()
            has_dashboard_text = 'Learning Dashboard' in content
            has_progress_text = 'progress' in content.lower()
            print(f"   Page contains 'Learning Dashboard': {has_dashboard_text}")
            print(f"   Page contains 'progress': {has_progress_text}")
            
            # Check for dashboard components
            cards = await page.locator('.card').count()
            progress_sections = await page.locator('text=/Progress Overview|Recommended for You/').count()
            
            print(f"   Cards on page: {cards}")
            print(f"   Progress/Recommendation sections: {progress_sections}")
            
            # Check navigation active state
            print("3. Checking navigation state...")
            dashboard_nav_active = await page.locator('a[href="/"] >> [aria-current="page"]').count()
            dashboard_nav_highlighted = await page.locator('a[href="/"].text-black').count()
            
            print(f"   Dashboard nav marked as active: {dashboard_nav_active}")
            print(f"   Dashboard nav highlighted: {dashboard_nav_highlighted}")
            
            # Take screenshot
            await page.screenshot(path='homepage_dashboard.png', full_page=True)
            print("   Screenshot saved as homepage_dashboard.png")
            
            print("\n" + "=" * 50)
            print("SUMMARY:")
            print(f"✓ Homepage loads: Yes")
            print(f"✓ Dashboard title: {'Yes' if dashboard_title > 0 else 'No'}")
            print(f"✓ Dashboard content: {'Yes' if cards > 0 else 'No'}")
            print(f"✓ Progress sections: {'Yes' if progress_sections > 0 else 'No'}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_homepage_dashboard())