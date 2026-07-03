import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check for 'Tutorial is loading' in scraped Codeforces problems.")
    parser.add_argument("file", help="Path to the JSONL file to check", default="data/codeforces_problems.jsonl", nargs='?')
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    loading_count = 0
    total_count = 0
    loading_problems = []

    print(f"Scanning {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                total_count += 1
                
                editorial = data.get("editorial_text", "")
                if editorial and "Tutorial is loading" in editorial:
                    loading_count += 1
                    loading_problems.append(data.get("id", "Unknown"))
            except json.JSONDecodeError:
                print("Warning: Skipping invalid JSON line.")

    print("\n--- Results ---")
    print(f"Total problems scanned: {total_count}")
    print(f"Problems with 'Tutorial is loading': {loading_count}")
    
    if loading_count > 0:
        print("\nProblem IDs with 'Tutorial is loading':")
        # Print up to 20 IDs as a sample
        sample = loading_problems[:20]
        print(", ".join(sample))
        if loading_count > 20:
            print(f"... and {loading_count - 20} more.")

if __name__ == "__main__":
    main()
