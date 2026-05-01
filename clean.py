import json

# Configuration
INPUT_FILE = "data\\rutgers_spaa_data.json"
OUTPUT_FILE = "data\\rutgers_spaa_data_filtered.json"
EXCLUDE_KEYWORDS = ['beta', 'taxonomy', 'newsroom','spaa_person','directory',
                    'courses','media','spaa-ba-declare-major','bainternshipquestionnairepdf',
                    'form', 'worksheet', '-application','contract','request','advisement',
                    'handbook','event','selc','conference','annual-report','manual',
                    'research-brief','cannabis','info-sheet','issue-brief','report','capstone-titles',
                    'manifesto','-ppt',"celebrating","photo","schedule","-brief","staff-highlights","baruch-map",
                    "tad-15"]

def filter_data():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data)
    
    # 2. Robust Filtering Logic
    filtered_data = []
    removed_items = []

    for item in data:
        url_lower = item['url'].lower()
        
        # This checks if any of our lowercase keywords are in the lowercase URL
        should_exclude = any(word.lower() in url_lower for word in EXCLUDE_KEYWORDS)
        
        if not should_exclude:
            filtered_data.append(item)
        else:
            removed_items.append(item['url'])

    # Save the new file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, separators=(',', ':'))

    print(f"--- Filter Results ---")
    print(f"Total pages before: {original_count}")
    print(f"Total pages after:  {len(filtered_data)}")
    print(f"Pages removed:      {len(removed_items)}")
    
    # Show a few things that were removed to confirm it's working
    if removed_items:
        print("\nExamples of removed URLs:")
        for url in removed_items[:5]: # Show first 5
            print(f" - {url}")

if __name__ == "__main__":
    filter_data()