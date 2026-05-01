import json

# Configuration
INPUT_FILE = "data\\rutgers_spaa_data.json"
OUTPUT_FILE = "data\\rutgers_spaa_data_filtered.json"
# Load the JSON data

# Load your JSON file
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

urls_to_remove = [
    "https://spaa.newark.rutgers.edu/phd-student-annual-evaluation-3rd-year",
    "https://spaa.newark.rutgers.edu/gwc-writing-a-strong-research-paper",
    "https://spaa.newark.rutgers.edu/gwc-capstone-checklist",
    "https://spaa.newark.rutgers.edu/gwc-writing-a-strong-research-paper",
    "https://spaa.newark.rutgers.edu/tad-15-local-and-ground-transportation",
    "https://spaa.newark.rutgers.edu/cnld-september-2025-brief",
    "https://spaa.newark.rutgers.edu/tad15-baruch-map"
]

remove_set = set(urls_to_remove)

# Filter out the unwanted URL
filtered_data = [
    item for item in data
    if item.get("url") not in remove_set
]


# Save the cleaned JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)