"""
Test if gpt-5-nano is available
"""
import os
from openai import OpenAI

print("="*80)
print("TESTING GPT-5-NANO AVAILABILITY")
print("="*80)

# Read API key
with open("openaiapikey.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

models_to_test = ["gpt-5-nano", "gpt-5-instant", "gpt-4o-mini"]

for model in models_to_test:
    print(f"\nTesting {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'test successful' in one sentence."}],
            max_tokens=20
        )
        result = response.choices[0].message.content
        print(f"  SUCCESS: {model} is available!")
        print(f"  Response: {result}")
        if hasattr(response, 'usage'):
            print(f"  Tokens used: {response.usage.total_tokens}")
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "model_not_found" in error_msg:
            print(f"  NOT AVAILABLE: Model not found or no access")
        else:
            print(f"  ERROR: {type(e).__name__}: {error_msg[:100]}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

