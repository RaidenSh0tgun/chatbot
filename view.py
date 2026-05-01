import json
from pprint import pprint

FILE_PATH = "data/rutgers_spaa_data.json"  # adjust if needed

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Type:", type(data))
#print("Top-level preview:")
#pprint(data[0:10], depth=3, width=120)
target_url = "https://spaa.newark.rutgers.edu/phd"

match = next(
    (item for item in data if item.get("url") == target_url),
    None
)

if match:
    print("FOUND")
    print(match["content"])  # preview content
else:
    print("URL not found")
