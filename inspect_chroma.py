from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DB_PATH = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

target_url = "https://spaa.newark.rutgers.edu/phd"

results = db._collection.get(
    where={"source": target_url}
)

print("Chunks found:", len(results["documents"]))

for i, doc in enumerate(results["documents"][:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(doc[:500])
