import json

INPUT_FILE = "rutgers_spaa_data.json"
OUTPUT_FILE = "rutgers_spaa_data_unique.json"

def normalize_url(url):
    """Standardizes URLs to find duplicates."""
    # 1. Force lowercase
    url = url.lower().strip()
    # 2. Standardize protocol to https
    if url.startswith("http://"):
        url = url.replace("http://", "https://", 1)
    # 3. Remove internal page anchors (#)
    url = url.split('#')[0]
    # 4. Remove trailing slashes
    url = url.rstrip('/')
    return url

def remove_duplicates():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data)
    seen_urls = set()
    unique_data = []

    for item in data:
        raw_url = item['url']
        clean_url = normalize_url(raw_url)

        if clean_url not in seen_urls:
            seen_urls.add(clean_url)
            # We keep the item, but we update the URL to the 'clean' version
            item['url'] = clean_url 
            unique_data.append(item)

    # Save compact JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, separators=(',', ':'))

    print(f"--- Deduplication Results ---")
    print(f"Total pages before: {original_count}")
    print(f"Total pages after:  {len(unique_data)}")
    print(f"Duplicate pages removed: {original_count - len(unique_data)}")

if __name__ == "__main__":
    remove_duplicates()