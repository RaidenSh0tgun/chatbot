import os
import json
import re
from pathlib import Path
from docx import Document
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

DATA_DIR = "./Data"
OUTPUT_FILE = "./Data/consolidated_rag_data.json"

model = OllamaLLM(model="qwen2.5")


def extract_urls(text):
    pattern = r"https?://[^\s\)\]\}>,]+"
    return re.findall(pattern, text or "")


def clean_text(text):
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_docx(file_path):
    doc = Document(file_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return clean_text(" ".join(paragraphs))


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    records = []
    for item in data:
        records.append({
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "retrieval_phrases": item.get(
                "retrieval_phrases",
                item.get("keyword", [])
            ),
            "content": clean_text(item.get("content", ""))
        })

    return records


def extract_json_from_response(response):
    response = response.strip()
    response = response.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        return match.group(0)

    return response


def generate_metadata(content):
    prompt = f"""
    You are helping prepare a RAG database for the School of Public Affairs and Administration at Rutgers University-Newark.

    Based on the content below, generate:
    1. A short and accurate title.
    2. 3 to 10 retrieval phrases.

    Retrieval phrase requirements:
    - Each retrieval phrase should be a natural search phrase that a user might ask or that strongly identifies the content.
    - Prefer phrase-level anchors, not isolated generic keywords.
    - Prioritize exact names, official program names, role titles, offices, procedures, services, forms, policies, degrees, certificates, courses, acronyms, and contact-related phrases.
    - Include exact official names when important for retrieval.
    - Include likely user query phrases when appropriate.
    - Avoid overly broad or low-information phrases such as "SPAA", "Rutgers", "University", "school", "program", or "student" by themselves.
    - Do NOT create vague phrases such as "public administration", "academic programs", or "student services" unless the content is specifically about that topic.
    - Retrieval phrases should usually contain 2 to 6 words.
    - Do not invent information not supported by the content.

    Good examples:
    - "PhD program director"
    - "Yahong Zhang"
    - "MPA application deadline"
    - "graduate certificate admission"
    - "international student requirements"
    - "public administration PhD curriculum"
    - "Urban Education Policy certificate"
    - "Nonprofit Management certificate"

    Bad examples:
    - "SPAA"
    - "Rutgers"
    - "faculty"
    - "students"
    - "program"
    - "research"
    - "education"

    Return ONLY valid JSON. Do not include markdown, explanation, or code fences.

    Format:
    {{
    "title": "...",
    "retrieval_phrases": ["...", "..."]
    }}

    Content:
    {content[:5000]}
    """

    response = model.invoke(prompt)

    try:
        json_text = extract_json_from_response(response)
        result = json.loads(json_text)

        title = result.get("title", "")
        retrieval_phrases = result.get("retrieval_phrases", [])

        if isinstance(retrieval_phrases, str):
            retrieval_phrases = [
                phrase.strip()
                for phrase in retrieval_phrases.split(",")
                if phrase.strip()
            ]

    except Exception as e:
        print("\nMetadata generation failed.")
        print("Raw response:", response[:500])
        print("Error:", e)

        title = ""
        retrieval_phrases = []

    return title, retrieval_phrases


def build_consolidated_json():
    all_records = []

    all_files = list(Path(DATA_DIR).iterdir())

    for file_path in tqdm(all_files, desc="Processing files"):

        if file_path.name == Path(OUTPUT_FILE).name:
            continue

        if file_path.suffix.lower() == ".json":
            records = load_json_file(file_path)
            all_records.extend(records)

        elif file_path.suffix.lower() == ".docx":
            content = read_docx(file_path)
            urls = extract_urls(content)

            if content:
                all_records.append({
                    "url": urls[0] if urls else "",
                    "title": file_path.stem,
                    "retrieval_phrases": [],
                    "content": content
                })

    final_records = []

    for record in tqdm(all_records, desc="Generating metadata"):
        content = record["content"]

        if not content:
            continue

        title = record.get("title", "")
        retrieval_phrases = record.get("retrieval_phrases", [])

        if not title or not retrieval_phrases:
            generated_title, generated_phrases = generate_metadata(content)

            if not title:
                title = generated_title

            if not retrieval_phrases:
                retrieval_phrases = generated_phrases

        final_records.append({
            "url": record.get("url", ""),
            "title": title,
            "retrieval_phrases": retrieval_phrases,
            "content": content
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(final_records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_consolidated_json()