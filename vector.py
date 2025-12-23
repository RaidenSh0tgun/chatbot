import os
import shutil
import json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Settings
DB_PATH = "./chroma_db"
DATA_FILE = "rutgers_spaa_data.json"

def create_database():
    # 1. Clear old database
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("Cleared old database.")

    # 2. Load and Split
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,      # Safe for mxbai-embed-large's 512 token limit
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []
    for item in data:
        chunks = text_splitter.split_text(item['content'])
        for chunk in chunks:
            # Safety cap: force truncate any weirdly long strings
            safe_text = chunk[:1500] 
            documents.append(Document(page_content=safe_text, metadata={"source": item['url']}))

    print(f"Generated {len(documents)} chunks.")

    # 3. Embed and Store
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Batching (prevents memory and context errors)
    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size), desc="Building DB"):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)

    print(f"Database ready at {DB_PATH}")

if __name__ == "__main__":
    create_database()