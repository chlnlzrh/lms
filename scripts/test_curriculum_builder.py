"""
Test script for the Curriculum Builder API.

This script tests the curriculum builder agent by sending a POST request
to the Next.js API route.
"""
import os
import sys
import json
import requests
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# API endpoint (adjust port if needed)
API_URL = "http://localhost:3000/api/curriculum-builder"

def test_curriculum_builder(subject: str):
    """Test the curriculum builder with a given subject."""
    print(f"\n{'='*80}")
    print(f"Testing Curriculum Builder")
    print(f"Subject: {subject}")
    print(f"API URL: {API_URL}")
    print(f"{'='*80}\n")
    
    try:
        # Make POST request
        response = requests.post(
            API_URL,
            json={"subject": subject},
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes timeout for long generations
        )
        
        response.raise_for_status()
        result = response.json()
        
        print("✓ Success!")
        print(f"\nResults:")
        print(f"  Subject: {result.get('subject')}")
        print(f"  Folder: {result.get('folderName')}")
        print(f"  Path: {result.get('path')}")
        print(f"  File existed: {result.get('fileExists', False)}")
        print(f"  Overwritten: {result.get('overwritten', False)}")
        print(f"\nToken Usage:")
        print(f"  Input tokens: {result.get('usage', {}).get('input_tokens', 0):,}")
        print(f"  Output tokens: {result.get('usage', {}).get('output_tokens', 0):,}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {API_URL}")
        print(f"  Make sure the Next.js dev server is running:")
        print(f"    npm run dev")
        return False
        
    except requests.exceptions.Timeout:
        print(f"✗ Error: Request timed out after 5 minutes")
        print(f"  This might be normal for large curriculum generations")
        return False
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {e}")
        try:
            error_detail = response.json()
            print(f"  Detail: {error_detail.get('error', 'Unknown error')}")
        except:
            print(f"  Response: {response.text[:500]}")
        return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1:
        subject = " ".join(sys.argv[1:])
    else:
        # Default test subject
        subject = "Machine Learning"
        print("No subject provided, using default: 'Machine Learning'")
        print("Usage: python scripts/test_curriculum_builder.py 'Your Subject Name'")
        print()
    
    success = test_curriculum_builder(subject)
    
    if success:
        print(f"\n{'='*80}")
        print("✓ Test completed successfully!")
        print(f"{'='*80}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*80}")
        print("✗ Test failed")
        print(f"{'='*80}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()




