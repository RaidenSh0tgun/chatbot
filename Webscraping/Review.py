import json
from collections import Counter, defaultdict

INPUT_FILE = "rutgers_spaa_data.json"

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 1. Containers for our data
path_counts = Counter()
path_examples = defaultdict(list)
path_lengths = defaultdict(list)

for item in data:
    url = item['url']
    content = item.get('content', '')
    
    # Extract the directory (e.g., /news/, /events/)
    parts = url.split('/')
    path = f"/{parts[3]}/" if len(parts) > 3 else "/"
    
    # Store stats
    path_counts[path] += 1
    path_lengths[path].append(len(content))
    
    # Keep up to 3 examples for each path
    if len(path_examples[path]) < 3:
        path_examples[path].append(url)

# 2. Calculate Averages and create a sortable list
results = []
for path, count in path_counts.items():
    avg_len = sum(path_lengths[path]) // count
    results.append({
        'path': path,
        'count': count,
        'avg_len': avg_len,
        'examples': ", ".join(path_examples[path])
    })

# 3. Sort the results by 'avg_len' in descending order
# Change reverse=True to reverse=False if you want to see the shortest pages first
sorted_results = sorted(results, key=lambda x: x['avg_len'], reverse=True)

# 4. Print the Table
print(f"{'PATH':<25} | {'PAGES':<7} | {'AVG LEN':<8} | {'EXAMPLES'}")
print("-" * 120)

for res in sorted_results[:20]:  # Show top 20 heaviest paths
    print(f"{res['path']:<25} | {res['count']:<7} | {res['avg_len']:<8} | {res['examples']}")

print("-" * 120)
print(f"Total Pages in JSON: {len(data)}")