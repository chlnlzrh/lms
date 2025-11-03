"""
Test OpenAI API key and connection
"""
import os
from openai import OpenAI

print("="*80)
print("TESTING OPENAI API CONNECTION")
print("="*80)

# Read API key
try:
    with open("openaiapikey.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except Exception as e:
    print(f"ERROR: Could not read openaiapikey.txt: {e}")
    exit(1)

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: API key not found!")
    exit(1)

# Check key format
if not api_key.startswith("sk-"):
    print("WARNING: API key doesn't start with 'sk-'. May not be valid OpenAI key.")
    print(f"Key starts with: {api_key[:10]}...")

print(f"\nAPI Key found: {api_key[:10]}...{api_key[-4:]}")
print(f"Key length: {len(api_key)} characters")

# Test connection
print("\n" + "-"*80)
print("Testing API connection...")
print("-"*80)

try:
    client = OpenAI(api_key=api_key)
    
    # Simple test call
    print("Making test API call...")
    response = client.chat.completions.create(
        model="gpt-5-instant",
        messages=[
            {"role": "user", "content": "Say 'API connection successful' in one sentence."}
        ],
        max_tokens=50,
        temperature=0.7
    )
    
    result = response.choices[0].message.content
    print(f"\nResponse received: {result}")
    
    # Check usage
    if hasattr(response, 'usage'):
        usage = response.usage
        print(f"\nToken usage:")
        print(f"  Input tokens: {usage.prompt_tokens}")
        print(f"  Output tokens: {usage.completion_tokens}")
        print(f"  Total tokens: {usage.total_tokens}")
    
    print("\n" + "="*80)
    print("SUCCESS: OpenAI API is working correctly!")
    print("="*80)
    print(f"Model tested: gpt-5-instant")
    print(f"Ready to generate lessons.")
    print("="*80)
    
except Exception as e:
    print("\n" + "="*80)
    print("ERROR: API call failed!")
    print("="*80)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    
    if "Invalid API key" in str(e) or "Incorrect API key" in str(e):
        print("\nThe API key appears to be invalid.")
        print("Please check that the key in openaiapikey.txt is correct.")
    elif "insufficient_quota" in str(e) or "billing" in str(e).lower():
        print("\nThe API key is valid but account has billing/quota issues.")
        print("Please check your OpenAI account billing status.")
    else:
        print("\nUnknown error. Please check:")
        print("1. API key is correct")
        print("2. You have internet connection")
        print("3. OpenAI service is accessible")
    
    print("="*80)
    exit(1)

