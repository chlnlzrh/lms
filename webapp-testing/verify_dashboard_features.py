#!/usr/bin/env python3
"""
Verify specific dashboard features that didn't pass in the first test
"""

import asyncio
from playwright.async_api import async_playwright

async def verify_dashboard():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Visual debugging
        page = await browser.new_page()
        
        try:
            print("Verifying Dashboard Features...")
            print("=" * 50)
            
            # Go to homepage first
            print("1. Loading homepage...")
            await page.goto('http://localhost:3000')
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(2000)
            
            # Check progress bars on homepage
            print("2. Checking progress elements on homepage...")
            progress_bars = await page.locator('.progress-bar').count()
            progress_fills = await page.locator('.progress-fill').count()
            print(f"   Progress bars: {progress_bars}")
            print(f"   Progress fills: {progress_fills}")
            
            # Navigate to dashboard
            print("3. Navigating to dashboard...")
            dashboard_links = await page.locator('a[href="/dashboard"]').all()
            print(f"   Found {len(dashboard_links)} dashboard links")
            
            if dashboard_links:
                await dashboard_links[0].click()
                await page.wait_for_load_state('networkidle')
                await page.wait_for_timeout(2000)
                
                print("4. Checking dashboard page content...")
                current_url = page.url
                print(f"   Current URL: {current_url}")
                
                # Check for dashboard title with different selectors
                dashboard_title1 = await page.locator('h1:has-text("Learning Dashboard")').count()
                dashboard_title2 = await page.locator('h1:has-text("Dashboard")').count()
                dashboard_title3 = await page.locator('.text-display').count()
                
                print(f"   Dashboard title 'Learning Dashboard': {dashboard_title1}")
                print(f"   Dashboard title 'Dashboard': {dashboard_title2}")
                print(f"   Display typography elements: {dashboard_title3}")
                
                # Check page content
                content = await page.content()
                has_dashboard_text = 'dashboard' in content.lower()
                print(f"   Page contains 'dashboard' text: {has_dashboard_text}")
                
                # Check for specific dashboard components
                cards = await page.locator('.card').count()
                recommendations_text = await page.locator('text=Recommended').count()
                activity_text = await page.locator('text=Activity').count()
                
                print(f"   Cards on dashboard: {cards}")
                print(f"   'Recommended' text: {recommendations_text}")
                print(f"   'Activity' text: {activity_text}")
                
                # Check progress elements on dashboard
                dashboard_progress_bars = await page.locator('.progress-bar').count()
                dashboard_progress_fills = await page.locator('.progress-fill').count()
                
                print(f"   Progress bars on dashboard: {dashboard_progress_bars}")
                print(f"   Progress fills on dashboard: {dashboard_progress_fills}")
                
                # Take a screenshot for debugging
                await page.screenshot(path='dashboard_debug.png', full_page=True)
                print("   Screenshot saved as dashboard_debug.png")
                
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(verify_dashboard())