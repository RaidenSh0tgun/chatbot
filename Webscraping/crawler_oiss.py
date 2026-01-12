import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import json
import csv
from typing import List, Tuple, Optional

# ------------------------
# CONFIGURATION
# ------------------------

# Option A (recommended): put your target URLs here directly
URL_LIST = [
    "https://myrun.newark.rutgers.edu/oiss",
    "https://myrun.newark.rutgers.edu/curricular-practical-training-cpt-0",
    "https://myrun.newark.rutgers.edu/campus-employment-0",
    "https://myrun.newark.rutgers.edu/optional-practical-training-opt-0",
    "https://myrun.newark.rutgers.edu/f-1-student",
    "https://myrun.newark.rutgers.edu/online-check-office-international-student-and-scholar-services-oiss",
    "https://myrun.newark.rutgers.edu/income-tax-filing-information-international-students-and-scholars-rutgers-university-newark",
    "https://myrun.newark.rutgers.edu/j-1-exchange-visitor",
    "https://myrun.newark.rutgers.edu/health-insurance",
    "https://myrun.newark.rutgers.edu/international",
    "https://myrun.newark.rutgers.edu/preparing-enrollment",
    "https://myrun.newark.rutgers.edu/covid-financial-resources",
    "https://myrun.newark.rutgers.edu/financing-your-education-0",
    "https://myrun.newark.rutgers.edu/academic-advisement"
    # "https://spaa.newark.rutgers.edu/...",  # add more
]

# Option B: load URLs from a file (one URL per line). Leave as None if not used.
URLS_FILE = None  # e.g., "url_list.txt"

# Optional: restrict scraping to a domain (set to None to disable domain restriction)
#ALLOWED_DOMAIN = "spaa.newark.rutgers.edu"
ALLOWED_DOMAIN = None

# Respectful delay between requests (seconds)
REQUEST_DELAY = 1.0

# Output files
OUT_JSON = "rutgers_oiss_data.json"
OUT_CSV = "rutgers_oiss_data.csv"

# Identify as a browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# File extensions to skip (non-HTML assets)
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z",
)

# HTML elements to remove before extracting text
REMOVE_TAGS = ("script", "style", "nav", "footer", "header", "noscript")


# ------------------------
# HELPERS
# ------------------------

def load_urls_from_file(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)
    return urls

def normalize_url(url: str) -> str:
    """
    Normalize to reduce duplicates:
    - remove fragments
    - strip trailing slash (except root)
    """
    parsed = urlparse(url)
    # Drop fragment
    parsed = parsed._replace(fragment="")
    clean = parsed.geturl()
    if clean.endswith("/") and len(clean) > len(f"{parsed.scheme}://{parsed.netloc}/"):
        clean = clean.rstrip("/")
    return clean

def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if any(parsed.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    if ALLOWED_DOMAIN and parsed.netloc != ALLOWED_DOMAIN:
        return False
    return True

def fetch_page(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (clean_text, error_message). If success, error_message is None.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"

        # Optional: content-type gate (skip obvious non-HTML even without extension)
        ctype = resp.headers.get("Content-Type", "").lower()
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            return None, f"Skipped non-HTML content-type: {ctype or 'unknown'}"

        soup = BeautifulSoup(resp.text, "html.parser")

        # remove noisy elements
        for tag_name in REMOVE_TAGS:
            for el in soup.find_all(tag_name):
                el.decompose()

        text = soup.get_text(separator=" ")
        clean_text = " ".join(text.split())

        if not clean_text:
            return None, "Empty text after cleaning"

        return clean_text, None

    except requests.exceptions.Timeout:
        return None, "Timeout"
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


# ------------------------
# MAIN
# ------------------------

def main():
    # 1) Assemble URL targets
    targets = list(URL_LIST)

    if URLS_FILE:
        targets.extend(load_urls_from_file(URLS_FILE))

    # 2) Normalize + filter + deduplicate (preserve order)
    seen = set()
    final_urls: List[str] = []
    for u in targets:
        u2 = normalize_url(u)
        if u2 in seen:
            continue
        seen.add(u2)
        if is_allowed(u2):
            final_urls.append(u2)
        else:
            print(f"Skipping (not allowed): {u2}")

    if not final_urls:
        print("No valid URLs to scrape. Check URL_LIST / URLS_FILE / ALLOWED_DOMAIN.")
        return

    print(f"Starting scraper for {len(final_urls)} provided URLs...")

    results = []
    failures = []

    for idx, url in enumerate(final_urls, start=1):
        print(f"[{idx}/{len(final_urls)}] Scraping: {url}")
        text, err = fetch_page(url)

        if text:
            results.append({"url": url, "content": text})
        else:
            failures.append({"url": url, "error": err or "Unknown error"})
            print(f"  Failed: {err}")

        time.sleep(REQUEST_DELAY)

    # 3) Export JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 4) Export CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "content"])
        writer.writeheader()
        writer.writerows(results)

    # 5) Optional: export failures log
    if failures:
        with open("scrape_failures.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "error"])
            writer.writeheader()
            writer.writerows(failures)

    print("\nDone.")
    print(f"Success pages: {len(results)}")
    print(f"Failed pages:  {len(failures)}")
    print(f"Files created: {OUT_JSON}, {OUT_CSV}" + (", scrape_failures.csv" if failures else ""))


if __name__ == "__main__":
    main()
