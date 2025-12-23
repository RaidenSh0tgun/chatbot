import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import csv
from collections import deque

# --- CONFIGURATION ---
START_URL = "https://spaa.newark.rutgers.edu/admissions"
DOMAIN = "spaa.newark.rutgers.edu"
visited_urls = set()
scraped_results = []
queue = deque([START_URL])

# Identifying as a browser to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_page_data(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None, None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Clean the HTML: Remove menus, footers, and scripts to get "pure" content
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Extract text and clean up whitespace
        text = soup.get_text(separator=' ')
        clean_text = ' '.join(text.split())
        
        return clean_text, soup
    except Exception as e:
        print(f"Error accessing {url}: {e}")
        return None, None

print("Starting Scraper...")

while queue:
    current_url = queue.popleft()
    
    # Skip if already visited
    if current_url in visited_urls:
        continue
        
    print(f"Scraping: {current_url}")
    visited_urls.add(current_url)
    
    content, soup = get_page_data(current_url)
    
    if content and soup:
        scraped_results.append({
            "url": current_url,
            "content": content
        })
        
        # Find all links on this page
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            # Convert relative (/apply) to absolute (https://...)
            full_url = urljoin(current_url, link).split('#')[0].rstrip('/')
            
            # THE FILTER:
            # 1. Must be on the same domain
            # 2. Must not be a file (PDF, etc.)
            # 3. Must not be an external site (Google, Facebook)
            if DOMAIN in full_url and full_url not in visited_urls:
                if not any(ext in full_url.lower() for ext in ['.pdf', '.jpg', '.png', '.docx']):
                    # Only follow links that are part of the main SPAA site
                    # (Avoid crawling the whole university by staying on spaa.newark.rutgers.edu)
                    queue.append(full_url)
    
    # Wait 1 second between pages to be respectful to the server
    time.sleep(1)

# --- EXPORT DATA ---

# 1. Save to JSON
with open("rutgers_spaa_data.json", "w", encoding="utf-8") as f:
    json.dump(scraped_results, f, indent=4, ensure_ascii=False)

# 2. Save to CSV
with open("rutgers_spaa_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["url", "content"])
    writer.writeheader()
    writer.writerows(scraped_results)

print(f"\nSuccess! Total pages found and scraped: {len(scraped_results)}")
print("Files created: rutgers_spaa_data.json, rutgers_spaa_data.csv")