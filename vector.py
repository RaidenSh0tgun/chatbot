import os
import json
import hashlib
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------
# SETTINGS
# ------------------------
DATA_DIR = "./Data"
CONSOLIDATED_JSON = "./Data/consolidated_rag_data.json"      # folder containing rutgers_spaa_data.json, rutgers_oiss_data.json, etc.
DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

# Chunking: keep these consistent after you start building the DB
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
SAFE_TEXT_CAP = 2000

# If True: wipe DB and rebuild from scratch every time
REBUILD_FROM_SCRATCH = False



# ------------------------
# HELPERS
# ------------------------
def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def list_json_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")
    files = []
    for name in os.listdir(data_dir):
        if name.lower().endswith(JSON_SUFFIX):
            files.append(os.path.join(data_dir, name))
    files.sort()
    return files

def load_records(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a JSON list of records.")

    cleaned = []

    for item in data:
        if not isinstance(item, dict):
            continue

        url = item.get("url", "")
        title = item.get("title", "")
        keyword = item.get("keyword", [])
        content = item.get("content", "")

        if isinstance(keyword, list):
            keyword_text = ", ".join(str(k).strip() for k in keyword if str(k).strip())
        else:
            keyword_text = str(keyword).strip()

        if isinstance(content, str) and content.strip():
            cleaned.append({
                "url": str(url).strip(),
                "title": str(title).strip(),
                "keyword": keyword,
                "keyword_text": keyword_text,
                "content": content.strip()
            })

    return cleaned

def build_documents_from_records(
        records: List[Dict[str, str]],
        source_file: str,
        splitter: RecursiveCharacterTextSplitter
    ) -> Tuple[List[Document], List[str]]:
    """
    Returns (documents, ids).
    ids are stable so we can upsert incrementally.
    """
    documents: List[Document] = []
    ids: List[str] = []

    for rec in records:
        url = rec.get("url", "")
        title = rec.get("title", "")
        keyword = rec.get("keyword", [])
        keyword_text = rec.get("keyword_text", "")
        content = rec.get("content", "")

        indexed_text = content.strip()

        # Create a record-level hash, then chunk-level hash
        record_fp = stable_hash(url + "\n" + title + "\n" + keyword_text + "\n" + content)

        chunks = splitter.split_text(indexed_text)

        for chunk_idx, chunk in enumerate(chunks):
            safe_text = chunk[:SAFE_TEXT_CAP]

            # Stable chunk id: file + url + record fingerprint + chunk index + chunk hash
            chunk_fp = stable_hash(safe_text)
            doc_id = stable_hash(f"{source_file}|{url}|{record_fp}|{chunk_idx}|{chunk_fp}")

            metadata = {
                "source_url": url,
                "title": title,
                "keyword": keyword_text,
                "source_file": os.path.basename(source_file),
                "record_fp": record_fp,
                "chunk_idx": chunk_idx,
            }

            documents.append(Document(page_content=safe_text, metadata=metadata))
            ids.append(doc_id)

    return documents, ids


# ------------------------
# MAIN
# ------------------------

def create_or_update_database():
    # Optional rebuild
    if REBUILD_FROM_SCRATCH and os.path.exists(DB_PATH):
        # safer to remove the folder only if you are sure you want full rebuild
        import shutil
        shutil.rmtree(DB_PATH)
        print("Cleared old database (REBUILD_FROM_SCRATCH=True).")

    # Initialize
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="rutgers_corpus",
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    json_files = [CONSOLIDATED_JSON]
    total_added = 0
    for jp in json_files:
        print(f"\nLoading: {jp}")
        records = load_records(jp)
        print(f"  Records: {len(records)}")

        documents, ids = build_documents_from_records(records, source_file=jp, splitter=splitter)
        print(f"  Chunks:   {len(documents)}")

        if not documents:
            continue

        # Incremental behavior:
        # - If ids already exist, Chroma will keep existing ones.
        # - We will only add truly missing ids by checking in batches.
        # (This avoids re-embedding everything every run.)
        batch_size = 200
        add_docs: List[Document] = []
        add_ids: List[str] = []

        for i in tqdm(range(0, len(ids), batch_size), desc="Checking new chunks"):
            batch_ids = ids[i:i + batch_size]

            # Chroma get() returns only existing IDs (if any)
            existing = vector_store.get(ids=batch_ids, include=[])
            existing_ids = set(existing.get("ids", []) or [])

            for j, doc_id in enumerate(batch_ids):
                if doc_id not in existing_ids:
                    add_ids.append(doc_id)
                    add_docs.append(documents[i + j])

        if add_docs:
            # Add in batches to reduce memory spikes
            for i in tqdm(range(0, len(add_docs), batch_size), desc="Adding embeddings"):
                vector_store.add_documents(
                    documents=add_docs[i:i + batch_size],
                    ids=add_ids[i:i + batch_size],
                )
            total_added += len(add_docs)
            print(f"  Added new chunks: {len(add_docs)}")
        else:
            print("  No new chunks to add (already indexed).")

    print(f"\nDatabase ready at: {DB_PATH}")
    print(f"Total new chunks added this run: {total_added}")


if __name__ == "__main__":
    create_or_update_database()
