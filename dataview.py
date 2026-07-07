import json

FILE_PATH = "data/consolidated_rag_data.json"  # adjust if needed

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):
    print("Top-level keys:")
    print(list(data.keys()))

elif isinstance(data, list):
    print(f"Top-level list with {len(data):,} items")
    print("First item:")
    print(data[0])