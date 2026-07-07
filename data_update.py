import json
import re
from pathlib import Path
from docx import Document

# ==========================
# Configuration
# ==========================
FILE_PATH = "data/"
WORD_PATH = "url"
JSON_FILE = FILE_PATH + "rutgers_spaa_data.json"
WORD_FOLDER = FILE_PATH + WORD_PATH     # folder containing new Word files
OUTPUT_FILE = FILE_PATH + "rutgers_spaa_data_updated.json"

# ==========================
# Read Word document
# ==========================
def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_url_and_content(text):
    """
    Expected format:

    url:
    https://xxxx

    content:
    xxxxx
    """

    url_match = re.search(
        r'url\s*:\s*(.*?)\s*(?=content\s*:)',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    content_match = re.search(
        r'content\s*:\s*(.*)',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not url_match:
        raise ValueError("No URL found.")

    url = url_match.group(1).strip()
    content = content_match.group(1).strip() if content_match else ""

    return url, content


# ==========================
# Load existing JSON
# ==========================
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Build lookup table
url_lookup = {item["url"]: item for item in data}

updated = 0
added = 0

# ==========================
# Process every Word file
# ==========================
for docx_file in Path(WORD_FOLDER).glob("*.docx"):
    print(f"Processing {docx_file.name}")

    try:
        text = read_docx(docx_file)
        url, content = extract_url_and_content(text)

        if url in url_lookup:
            url_lookup[url]["content"] = content
            updated += 1
            print("  Updated")
        else:
            new_item = {
                "url": url,
                "content": content
            }
            data.append(new_item)
            url_lookup[url] = new_item
            added += 1
            print("  Added")

    except Exception as e:
        print(f"  Skipped: {e}")

# ==========================
# Save
# ==========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("\nDone!")
print(f"Updated: {updated}")
print(f"Added:   {added}")
print(f"Saved to: {OUTPUT_FILE}")