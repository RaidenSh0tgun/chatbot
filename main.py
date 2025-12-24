# main.py
# Fully edited version with:
# - Conversation logging to CSV
# - In-memory per-session history (last N lines)
# - LLM router that decides whether to use retrieval
# - If retrieval is needed, LLM generates a search query using question + conversation history
# - Robust JSON parsing + safe fallbacks
# - Citations rendered as clickable [1], [2], ... links when sources exist

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import csv
import os
import json
import re
from datetime import datetime

# ----------------------------
# 1) LOGGING DIRECTORY
# ----------------------------
os.makedirs("conversation", exist_ok=True)

def save_to_csv(session_id: str, sender: str, message: str) -> None:
    filename = f"conversation/{session_id}.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "session_id", "sender", "message"])
        writer.writerow([datetime.now().isoformat(), session_id, sender, message])

# ----------------------------
# 2) FLASK & MEMORY
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/chat/*": {"origins": "*"}})

# session_id -> list[str] like "User: ...", "Assistant: ..."
conversation_memory = {}

# Keep memory small for speed
MAX_MEMORY_LINES = 10  # (each turn adds 2 lines: user + assistant)

# ----------------------------
# 3) VECTOR DB (RAG)
# ----------------------------
print("Connecting to Vector Database...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Default retriever settings (router can optionally override k if you extend it)
retriever = vector_db.as_retriever(
    search_kwargs={"k": 5}
)

# ----------------------------
# 4) LLM
# ----------------------------
# You can swap models; keep router+answer on same model for simplicity
model = OllamaLLM(model="gemma3:4b")

# ----------------------------
# 5) ROUTER PROMPT (LLM decides retrieval + search query)
# ----------------------------
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
router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | model

def parse_router_json(text) -> dict:
    """
    Gemma may occasionally output extra text around JSON.
    This function attempts strict JSON parse, then falls back to extracting the first {...} block.
    """
    if not isinstance(text, str):
        text = str(text)

    # Attempt direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Extract first JSON object-like block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {
            "use_retrieval": True,
            "search_query": "",
            "reason": "Router output not parseable; defaulting to retrieval."
        }

    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return {
        "use_retrieval": True,
        "search_query": "",
        "reason": "Router JSON extraction failed; defaulting to retrieval."
    }

# ----------------------------
# 6) ANSWER PROMPT (uses retrieval results if provided)
# ----------------------------
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
answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain = answer_prompt | model

# ----------------------------
# 7) CHAT ENDPOINT
# ----------------------------
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not question or not session_id:
        return jsonify({"error": "Missing question or session_id"}), 400

    # Initialize session memory if new
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    # Save user input to CSV
    save_to_csv(session_id, "User", question)

    # Prepare history for routing + answering
    history_string = "\n".join(conversation_memory[session_id]).strip()

    # --- STEP A0: ROUTE (LLM decides retrieval + builds query) ---
    router_raw = router_chain.invoke({
        "context": history_string,
        "question": question
    })
    router_decision = parse_router_json(router_raw)

    use_retrieval = bool(router_decision.get("use_retrieval", True))
    search_query = (router_decision.get("search_query") or "").strip()
    router_reason = (router_decision.get("reason") or "").strip()

    # Optional: log router decisions for debugging
    # Comment out if you prefer not to log internal behavior.
    save_to_csv(session_id, "Router", f"use_retrieval={use_retrieval}; search_query={search_query}; reason={router_reason}")

    # --- STEP A1: CONDITIONAL RETRIEVAL ---
    docs = []
    info_text = ""
    sources = []

    if use_retrieval:
        effective_query = search_query if search_query else question
        try:
            docs = retriever.invoke(effective_query)
        except Exception as e:
            # If retrieval fails, proceed without info
            docs = []
            save_to_csv(session_id, "System", f"Retriever error: {repr(e)}")

        info_text = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
        sources = list(set([doc.metadata.get('source', 'Unknown source') for doc in docs])) if docs else []
    else:
        info_text = ""
        sources = []

    # --- STEP B: GENERATE RESPONSE ---
    ai_response_text = answer_chain.invoke({
        "context": history_string,
        "info": info_text,
        "question": question
    })

    # Ensure ai_response_text is a string
    if not isinstance(ai_response_text, str):
        ai_response_text = str(ai_response_text)

    # --- STEP C: FORMAT HTML CITATIONS ---
    # Only show citations if sources look meaningful
    cleaned_sources = [s for s in sources if s and s != "Unknown source"]
    if cleaned_sources:
        # Stable ordering for consistent numbering
        cleaned_sources = sorted(set(cleaned_sources))
        links_html = " ".join([
            f'<a href="{url}" target="_blank" class="source-link">[{i+1}]</a>'
            for i, url in enumerate(cleaned_sources)
        ])
        final_display_answer = f"{ai_response_text}<br><br><small>Sources: {links_html}</small>"
    else:
        final_display_answer = ai_response_text
        cleaned_sources = []

    # --- STEP D: LOGGING & MEMORY UPDATE ---
    save_to_csv(session_id, "Assistant", final_display_answer)

    # Save clean text to memory (avoid HTML tags cluttering next turn)
    conversation_memory[session_id].append(f"User: {question}")
    conversation_memory[session_id].append(f"Assistant: {ai_response_text}")

    # Limit memory size
    if len(conversation_memory[session_id]) > MAX_MEMORY_LINES:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_LINES:]

    # --- STEP E: SEND RESPONSE ---
    return jsonify({
        "answer": final_display_answer,   # HTML version for display
        "raw_text": ai_response_text,     # Plain text version
        "sources": cleaned_sources,       # Raw list of URLs (cleaned)
        "session_id": session_id,
        "routing": {                      # Optional: helpful for debugging frontend
            "use_retrieval": use_retrieval,
            "search_query": search_query,
            "reason": router_reason
        }
    })

# ----------------------------
# 8) RUN SERVER
# ----------------------------
if __name__ == '__main__':
    # host='0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
