"""
Test gpt-5-nano without max_tokens parameter
"""
import os
from openai import OpenAI

print("="*80)
print("TESTING GPT-5-NANO (without max_tokens)")
print("="*80)

with open("openaiapikey.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

print("\nTesting gpt-5-nano without max_tokens parameter...")
try:
    # Try without max_tokens
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say 'test successful' in one sentence."}]
    )
    result = response.choices[0].message.content
    print(f"SUCCESS: gpt-5-nano works!")
    print(f"Response: {result}")
    if hasattr(response, 'usage'):
        print(f"Tokens used: {response.usage.total_tokens}")
    print("\n" + "="*80)
    print("gpt-5-nano is available and working!")
    print("Note: This model does not support max_tokens parameter")
    print("="*80)
except Exception as e:
    print(f"ERROR: {type(e).__name__}")
    print(f"Error message: {str(e)[:200]}")
    print("\n" + "="*80)
    print("gpt-5-nano test failed")
    print("="*80)

