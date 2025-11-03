"""
Helper script to check status of an OpenAI Batch API job and download results when complete.
"""
import os
import sys
from openai import OpenAI

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Read API key
try:
    with open("openaiapikey.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OpenAI API key not found!")
    sys.exit(1)

client = OpenAI(api_key=api_key)

def check_batch_status(batch_id, download_results=False):
    """Check status of a batch job and optionally download results."""
    batch = client.batches.retrieve(batch_id)
    
    print(f"\n{'='*80}")
    print(f"Batch Status: {batch.id}")
    print(f"{'='*80}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    
    if hasattr(batch, 'request_counts'):
        counts = batch.request_counts
        print(f"\nRequest Counts:")
        print(f"  Total: {counts.get('total', 'N/A')}")
        print(f"  Completed: {counts.get('completed', 'N/A')}")
        print(f"  Failed: {counts.get('failed', 'N/A')}")
    
    if hasattr(batch, 'errors') and batch.errors:
        print(f"\nErrors: {batch.errors}")
    
    if batch.status == "completed" and batch.output_file_id:
        print(f"\nOutput file ID: {batch.output_file_id}")
        
        if download_results:
            print("\nDownloading results...")
            result_content = client.files.content(batch.output_file_id)
            output_file = f"scripts/batch_api/batch_{batch_id}_results.jsonl"
            os.makedirs("scripts/batch_api", exist_ok=True)
            
            with open(output_file, 'wb') as f:
                f.write(result_content.read())
            
            print(f"✓ Downloaded to: {output_file}")
    
    print(f"{'='*80}\n")
    return batch


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_batch_status.py <batch_id> [--download]")
        sys.exit(1)
    
    batch_id = sys.argv[1]
    download = '--download' in sys.argv or '-d' in sys.argv
    
    check_batch_status(batch_id, download)

