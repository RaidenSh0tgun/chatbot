# main.py
# Fully edited version with:
# - Conversation logging to CSV
# - In-memory per-session history (last N lines)
# - LLM router that decides whether to use retrieval
# - If retrieval is needed, LLM generates a search query using question + conversation history
# - Robust JSON parsing + safe fallbacks
# - Citations rendered as clickable [1], [2], ... links when sources exist
# NEW
try:
    from langdetect import detect, LangDetectException
except Exception:
    detect = None
    LangDetectException = Exception
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

def save_to_csv(session_id: str, sender: str, message: str, search_query: str = "") -> None:
    filename = f"conversation/{session_id}.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        if not file_exists:
            # Added "search_query" to the header
            writer.writerow(["timestamp", "session_id", "sender", "message", "search_query"])
        
        # Added search_query to the row data
        writer.writerow([datetime.now().isoformat(), session_id, sender, message, search_query])

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
    embedding_function=embeddings,
    collection_name="rutgers_corpus"   # <-- add this
)
# Default retriever settings (router can optionally override k if you extend it)
retriever = vector_db.as_retriever(
    search_kwargs={"k": 15}
)

# ----------------------------
# 4) LLM
# ----------------------------
# You can swap models; keep router+answer on same model for simplicity
model = OllamaLLM(model="qwen2.5")

# ----------------------------
# 5) ROUTER PROMPT (LLM decides retrieval + search query)
# ----------------------------
router_template = """
You are a routing module for a Rutgers SPAA chatbot.

User language: {user_lang_name} ({user_lang})

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

Important:
- The vector database content is primarily in English.
- If the user language is not English, translate the search query into concise English keywords for best retrieval.

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
- If use_retrieval is true, search_query MUST be short, high-signal English keywords (not a full paragraph).
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
# 5.1) POST-RETRIEVAL FILTER PROMPT (NEW)
# ----------------------------
filter_template = """
You are a Document Relevance Filter for Rutgers SPAA.

User language: {user_lang_name} ({user_lang})

User Question: {question}
Conversation History: {context}

Retrieved Documents:
{docs_content}

Task:
1. Review each document above.
2. Select ONLY the documents that are factually relevant to the user's specific question.
3. Return the results as a JSON list of indices (0-indexed).

Example Output:
{{ "selected_indices": [0, 2] }}

Return ONLY JSON.
JSON:
"""

filter_prompt = ChatPromptTemplate.from_template(filter_template)
filter_chain = filter_prompt | model

def parse_filter_json(text) -> list:
    """Extracts the list of selected indices from the LLM output."""
    try:
        # Simple extraction logic similar to your router parser
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            return obj.get("selected_indices", [])
    except:
        pass
    return [] # If parsing fails, we handle it in the endpoint


# ----------------------------
# 6) ANSWER PROMPT (uses retrieval results if provided)
# ----------------------------
answer_template = """
Your name is SPAA Bot. You are a helpful assistant for the School of Public Affairs and Administration (SPAA) at Rutgers University-Newark.

User language: {user_lang_name} ({user_lang})

### INSTRUCTIONS:
- Do NOT introduce yourself or state your name in your response. 
- Do NOT say "Hello" or "Hi" unless the user is specifically greeting you for the first time.
- Answer in 1-5 sentences unless requested otherwise.
- If the user language is not English, respond in {user_lang_name}. Keep proper nouns (program names, office names) in English if they appear in the source text.
- Ground your answer ONLY in the Related Information provided.

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

#Extra: Language Detect
# NEW: very small mapping for prompt friendliness
LANG_NAME = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi"
}

def normalize_lang(code: str) -> str:
    if not code:
        return "en"
    c = code.lower().strip()
    # langdetect returns 'zh-cn'/'zh-tw' sometimes; keep as-is
    return c

def detect_user_language(text: str) -> str:
    """
    Returns a short language code like 'en', 'es', 'zh-cn', etc.
    Falls back to 'en' if uncertain.
    """
    t = (text or "").strip()
    if not t:
        return "en"

    # Quick heuristic: if lots of CJK chars, force 'zh' (helps reliability)
    cjk = sum(1 for ch in t if '\u4e00' <= ch <= '\u9fff')
    if cjk >= 2:
        return "zh"

    if detect is None:
        return "en"

    try:
        code = detect(t)
        return normalize_lang(code)
    except LangDetectException:
        return "en"

def lang_display(code: str) -> str:
    return LANG_NAME.get(code, code)

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
    save_to_csv(session_id, "User", question, search_query="")

    # Prepare history for routing + answering
    history_string = "\n".join(conversation_memory[session_id]).strip()

    # NEW: detect language from the current question (or you can include history too)
    user_lang = detect_user_language(question)
    user_lang_name = lang_display(user_lang)

    # --- STEP A0: ROUTE (LLM decides retrieval + builds query) ---
    router_raw = router_chain.invoke({
        "context": history_string,
        "question": question,
        "user_lang": user_lang,
        "user_lang_name": user_lang_name
    })

    router_decision = parse_router_json(router_raw)

    use_retrieval = bool(router_decision.get("use_retrieval", True))
    search_query = (router_decision.get("search_query") or "").strip()
    router_reason = (router_decision.get("reason") or "").strip()

    # Optional: log router decisions for debugging
    # Comment out if you prefer not to log internal behavior.
    # Log router decisions with the search_query column populated
    save_to_csv(
        session_id, 
        "Router", 
        f"use_retrieval={use_retrieval}; reason={router_reason}", 
        search_query=search_query
    )
    
    # --- STEP A1: RETRIEVAL ---
    docs = []
    if use_retrieval:
        effective_query = search_query if search_query else question
        try:
            docs = retriever.invoke(effective_query)
        except Exception as e:
            save_to_csv(session_id, "System", f"Retriever error: {repr(e)}")

        # --- STEP A2: POST-RETRIEVAL FILTERING (NEW) ---
        if docs:
            # Prepare the documents for the LLM to review
            docs_content_for_filter = ""
            for i, d in enumerate(docs):
                docs_content_for_filter += (
                    f"[{i}] SOURCE_URL: {d.metadata.get('source_url')}\n"
                    f"SOURCE_FILE: {d.metadata.get('source_file')}\n"
                    f"CONTENT: {d.page_content}\n---\n"
                )
            filter_raw = filter_chain.invoke({
                "question": question,
                "context": history_string,
                "docs_content": docs_content_for_filter,
                "user_lang": user_lang,
                "user_lang_name": user_lang_name
            })

            
            selected_indices = parse_filter_json(filter_raw)
            
            # If the LLM selected specific docs, filter the list. 
            # Otherwise, keep all docs as a fallback.
            if selected_indices:
                filtered_docs = [docs[i] for i in selected_indices if i < len(docs)]
                if filtered_docs:
                    docs = filtered_docs
                    save_to_csv(session_id, "System", f"Filtered docs from {len(docs_content_for_filter.split('---'))-1} down to {len(docs)}")

        # Final preparation of text for the Answer Chain
        info_text = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get("source_url", "Unknown source") for doc in docs]))
    else:
        info_text = ""
        sources = []

    # --- STEP B: GENERATE RESPONSE ---
    # Now uses the filtered info_text
    ai_response_text = answer_chain.invoke({
        "context": history_string,
        "info": info_text,
        "question": question,
        "user_lang": user_lang,
        "user_lang_name": user_lang_name
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
    # Log Assistant response with the search_query used for this turn
    save_to_csv(session_id, "Assistant", final_display_answer, search_query=search_query)

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
