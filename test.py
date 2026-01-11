import pandas as pd
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import json
import re

# ----------- SETTINGS -----------
input_file = 'Test/test_list.csv'
output_file = 'Test/spaa_test_results.csv'
model_name = "gemma3:4b"
embedding_model = "mxbai-embed-large"
db_directory = "./chroma_db"

# ----------- INITIALIZE RAG COMPONENTS -----------
print("Connecting to Vector Database...")
embeddings = OllamaEmbeddings(model=embedding_model)
vector_db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5}) # Matched to main.py k=5

print("Initializing Model...")
model = OllamaLLM(model=model_name)

# ----------- PROMPTS (Synced with main.py) -----------
router_template = """
You are a routing module for a Rutgers SPAA chatbot.

Goal:
Decide whether you must consult the knowledge base (vector database) to answer accurately.
If needed, generate an effective search query grounded in the user's question AND the conversation history.

Use retrieval when:
- The user asks for SPAA-specific facts (people, programs, admissions, deadlines, tuition/fees, offices, contacts, policies, forms, procedures, locations).
- The user refers to something likely in SPAA webpages or internal documents.
- The user asks for URLs, official details, step-by-step SPAA instructions, or factual claims about SPAA.

Do NOT use retrieval when:
- The user asks for generic advice, general concepts, brainstorming, or rewriting text.
- The user is only asking to format/rephrase something already in conversation history.
- The question can be answered without SPAA-specific facts.

Conversation History (most recent last):
{context}

User Question:
{question}

Return ONLY valid JSON with exactly these keys:
- use_retrieval: true/false
- search_query: string
- reason: string

Rules:
- If use_retrieval is false, set search_query to "".
- If use_retrieval is true, search_query MUST be a short, high-signal query (not a full paragraph).
- Do not include any keys beyond the three keys above.

JSON:
"""

answer_template = """
Your name is Friday. You are a helpful assistant for the School of Public Affairs and Administration (SPAA) at Rutgers University-Newark.

Always answer in 1-5 sentences unless the user explicitly requests more detail.

Conversation History:
{context}

Related Information (may be empty if retrieval not needed):
{info}

Question: {question}

Instructions:
- If Related Information is provided, ground your answer in it. Do not invent SPAA-specific facts that are not supported by it.
- If Related Information is empty, answer using general knowledge and the Conversation History only, but do not fabricate SPAA-specific facts (names, dates, requirements, contacts). If the question requires SPAA-specific facts and you lack them, say you don't know and suggest what to ask for.
"""

router_prompt = ChatPromptTemplate.from_template(router_template)
answer_prompt = ChatPromptTemplate.from_template(answer_template)

router_chain = router_prompt | model
answer_chain = answer_prompt | model

# ----------- HELPER FUNCTIONS -----------
def parse_router_json(text) -> dict:
    if not isinstance(text, str): text = str(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict): return obj
    except:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return {"use_retrieval": True, "search_query": "", "reason": "Failed to parse"}

# ----------- LOAD TEST DATA -----------
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: Could not find {input_file}")
    exit()

# ----------- RUN TESTING -----------
for run in range(1, 4):
    ans_col = f'answer_{run}'
    src_col = f'sources_{run}'
    qry_col = f'search_query_{run}' # NEW COLUMN
    
    print(f"\nStarting Run {run}/3...")
    
    generated_answers = []
    retrieved_sources = []
    generated_queries = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Run {run}"):
        question = row['question']
        
        try:
            # 1. ROUTER STEP (Generate search query)
            # Context is empty for testing unless your CSV has a history column
            router_raw = router_chain.invoke({"context": "", "question": question})
            decision = parse_router_json(router_raw)
            
            use_retrieval = decision.get("use_retrieval", True)
            search_query = decision.get("search_query", "").strip()
            
            # 2. RETRIEVAL STEP
            info_text = ""
            sources = "N/A"
            
            if use_retrieval:
                # Use generated search query if available, else fallback to raw question
                effective_query = search_query if search_query else question
                docs = retriever.invoke(effective_query)
                info_text = "\n\n".join([d.page_content for d in docs])
                sources = ", ".join(list(set([d.metadata.get('source', 'Unknown') for d in docs])))

            # 3. ANSWER STEP
            result = answer_chain.invoke({
                "context": "",
                "info": info_text,
                "question": question
            })
            
            generated_answers.append(result.strip())
            retrieved_sources.append(sources)
            generated_queries.append(search_query if use_retrieval else "[NO RETRIEVAL]")
            
        except Exception as e:
            print(f"Error: {e}")
            generated_answers.append("ERROR")
            retrieved_sources.append("ERROR")
            generated_queries.append("ERROR")

    # Update DataFrame
    df[ans_col] = generated_answers
    df[src_col] = retrieved_sources
    df[qry_col] = generated_queries

# ----------- SAVE RESULTS -----------
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nTesting Complete! Saved to: {output_file}")