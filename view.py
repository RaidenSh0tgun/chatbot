import json

FILE_PATH = "data/rutgers_spaa_data.json"

# Load JSON
with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

while True:
    target_url = input("\nEnter a URL (or press Enter to quit): ").strip()

    if not target_url:
        break

    match = next(
        (item for item in data if item.get("url") == target_url),
        None
    )

    if match:
        print("\nFOUND\n")
        print(match.get("content", "[No content]"))
    else:
        print("\nURL not found.")