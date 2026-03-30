
# main.py
# Revised version with:
# - Conversation logging to CSV
# - In-memory per-session history
# - Language detection
# - Persona detection based on background / current occupation
# - Optional one-time acknowledgment for service-relevant personas
# - LLM router for retrieval decisions
# - Post-retrieval LLM filtering
# - Answer generation grounded in retrieved content
# - HTML citations

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
            writer.writerow(["timestamp", "session_id", "sender", "message", "search_query"])
        writer.writerow([datetime.now().isoformat(), session_id, sender, message, search_query])


# ----------------------------
# 2) FLASK & MEMORY
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/chat/*": {"origins": "*"}})

# session_id -> ["User: ...", "Assistant: ...", ...]
conversation_memory = {}

# session_id -> persona info dict
persona_memory = {}

# Keep memory small for speed
MAX_MEMORY_LINES = 10  # each turn adds 2 lines: user + assistant


# ----------------------------
# 3) VECTOR DB (RAG)
# ----------------------------
print("Connecting to Vector Database...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="rutgers_corpus"
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 15}
)


# ----------------------------
# 4) LLM
# ----------------------------
# Keep the same model for persona, router, filter, and answer for simplicity.
model = OllamaLLM(model="qwen2.5")


# ----------------------------
# 5) ROUTER PROMPT
# ----------------------------
router_template = """
You are a routing module for a chatbot of Rutgers School of Public Affairs and Administration (SPAA).

User language: {user_lang_name} ({user_lang})
Detected persona: {persona}
Persona confidence: {persona_confidence}

Goal:
Decide whether you must consult the knowledge base (vector database) to answer accurately.
If needed, generate an effective search query grounded in the user's question AND the conversation history.

Use retrieval when:
- The user asks for school-specific facts (phd, mpa, undergraduate, people, programs, admissions, deadlines, tuition/fees, offices, contacts, policies, forms, procedures, locations, courses, classes).
- The user refers to something likely in school webpages or internal documents.
- The user asks for URLs, official details, step-by-step instructions, or factual claims about the school.

Persona guidance:
- veteran: prioritize veteran benefits, military-connected student support, funding, transition support, and relevant offices if supported by the school content.
- government_employee: prioritize career-relevant program information, executive/public service relevance, and professional advancement information.
- nonprofit_professional: prioritize nonprofit management, leadership, governance, fundraising, and mission-driven career relevance.
- law_enforcement: prioritize public service, public safety leadership, administration, ethics, and policy relevance when supported by school content.
- student: prioritize current student procedures, registration, advising, graduation, and support services.
- prospective_student: prioritize admissions, deadlines, requirements, tuition, and program fit.
- international_user: prioritize international admissions, visa/I-20 support, and international student services when supported by the content.
- faculty_or_staff: prioritize administrative, academic, and office information.
- general_public / unknown: use the user's actual question without forcing a role-specific angle.

Important:
- The vector database content is primarily in English.
- If the user language is not English, translate the search query into concise English keywords for best retrieval.
- Do not overuse persona. Use it only to improve relevance.

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
    Attempts strict JSON parse, then falls back to extracting the first {...} block.
    """
    if not isinstance(text, str):
        text = str(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

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
# 5.1) POST-RETRIEVAL FILTER PROMPT
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
    """
    Extract the list of selected indices from the LLM output.
    """
    if not isinstance(text, str):
        text = str(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            selected = obj.get("selected_indices", [])
            if isinstance(selected, list):
                return selected
    except Exception:
        pass

    try:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            selected = obj.get("selected_indices", [])
            if isinstance(selected, list):
                return selected
    except Exception:
        pass

    return []


# ----------------------------
# 5.2) PERSONA DETECTION PROMPT
# ----------------------------
persona_template = """
You are a persona detection module for a Rutgers SPAA chatbot.

User language: {user_lang_name} ({user_lang})

Conversation History (most recent last):
{context}

Current User Question:
{question}

Task:
Identify whether the user explicitly states or strongly indicates a service-relevant background or current occupation.

Choose exactly one persona from:
- law_enforcement
- veteran
- government_employee
- nonprofit_professional
- student
- prospective_student
- international_user
- faculty_or_staff
- general_public
- unknown

Also decide whether a brief acknowledgment should be used in the assistant response.

Rules:
- Use law_enforcement only if the user explicitly says they work in policing, corrections, public safety, law enforcement, or similar.
- Use veteran only if the user explicitly says they are a veteran, former military member, or similar.
- Use government_employee only if the user clearly indicates they work for government, public administration, or a public agency.
- Use nonprofit_professional only if the user clearly indicates they work for, manage, or lead a nonprofit or NGO.
- Use student if the user clearly indicates they are a current student.
- Use prospective_student if the user is clearly asking as an applicant or potential future student.
- Use international_user if the user clearly indicates international admissions, visa, I-20, or related international student identity or needs.
- Use faculty_or_staff if the user clearly indicates they are an instructor, professor, administrator, or university staff member.
- Use general_public for broad public questions with no meaningful role signal.
- Use unknown when evidence is weak or ambiguous.
- Only recommend acknowledgment when it is clearly appropriate and natural.
- Do not infer sensitive or personal identity traits beyond what the user states.

Return ONLY valid JSON with exactly these keys:
- persona: string
- confidence: number
- use_acknowledgment: true/false
- acknowledgment: string
- reason: string

Rules:
- confidence must be between 0 and 1
- if use_acknowledgment is false, acknowledgment must be ""
- no extra text

JSON:
"""

persona_prompt = ChatPromptTemplate.from_template(persona_template)
persona_chain = persona_prompt | model


def parse_persona_json(text) -> dict:
    if not isinstance(text, str):
        text = str(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {
        "persona": "unknown",
        "confidence": 0.0,
        "use_acknowledgment": False,
        "acknowledgment": "",
        "reason": "Persona output not parseable."
    }


# ----------------------------
# 6) ANSWER PROMPT
# ----------------------------
answer_template = """
Your name is SPAA Bot. You are a helpful assistant for the School of Public Affairs and Administration (SPAA) at Rutgers University-Newark.

User language: {user_lang_name} ({user_lang})
Detected persona: {persona}
Persona confidence: {persona_confidence}
Acknowledgment to use: {acknowledgment_to_use}

### INSTRUCTIONS:
- Do NOT introduce yourself or state your name in your response.
- Do NOT say "Hello" or "Hi" unless the user is specifically greeting you for the first time.
- Answer in 1-5 sentences unless requested otherwise.
- If the user language is not English, respond in {user_lang_name}. Keep proper nouns (program names, office names) in English if they appear in the source text.
- Ground your answer ONLY in the Related Information provided.
- Tailor the response to the user's likely background when relevant.
- If Acknowledgment to use is not empty, you may include it naturally at the beginning of the response.
- Do not overdo personalization and do not repeat acknowledgment unless it is supplied for this turn.
- Do not explicitly mention persona classification.

Conversation History:
{context}

Related Information (may be empty if retrieval not needed):
{info}

Question: {question}

Instructions:
- If Related Information is provided, ground your answer in it. Do not invent SPAA-specific facts that are not supported by it.
- If Related Information is empty, answer using general knowledge and the Conversation History only, but do not fabricate SPAA-specific facts (names, dates, requirements, contacts).
- If the question requires SPAA-specific facts and you lack them, say you do not know and suggest what information to ask for next.
"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain = answer_prompt | model


# ----------------------------
# EXTRA: Language Detect
# ----------------------------
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
    return code.lower().strip()

def detect_user_language(text: str) -> str:
    """
    Returns a short language code like 'en', 'es', 'zh-cn', etc.
    Falls back to 'en' if uncertain.
    """
    t = (text or "").strip()
    if not t:
        return "en"

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
# HELPERS
# ----------------------------
def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def sanitize_persona_label(label: str) -> str:
    allowed = {
        "law_enforcement",
        "veteran",
        "government_employee",
        "nonprofit_professional",
        "student",
        "prospective_student",
        "international_user",
        "faculty_or_staff",
        "general_public",
        "unknown"
    }
    label = (label or "").strip()
    return label if label in allowed else "unknown"

def sanitize_acknowledgment(persona: str, use_acknowledgment: bool, acknowledgment: str) -> str:
    """
    Restrict acknowledgment language to a small safe set so the model does not improvise
    something overly strong or awkward.
    """
    if not use_acknowledgment:
        return ""

    persona_to_ack = {
        "veteran": "Thank you for your service.",
        "law_enforcement": "Thank you for your public service."
    }
    return persona_to_ack.get(persona, "")


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

    # Save user input
    save_to_csv(session_id, "User", question, search_query="")

    # Prepare history
    history_string = "\n".join(conversation_memory[session_id]).strip()

    # --- STEP A0: LANGUAGE DETECTION ---
    user_lang = detect_user_language(question)
    user_lang_name = lang_display(user_lang)

    # --- STEP A1: PERSONA DETECTION ---
    previous_persona = persona_memory.get(session_id, {
        "persona": "unknown",
        "confidence": 0.0,
        "use_acknowledgment": False,
        "acknowledgment": ""
    })

    persona_raw = persona_chain.invoke({
        "context": history_string,
        "question": question,
        "user_lang": user_lang,
        "user_lang_name": user_lang_name
    })

    persona_result = parse_persona_json(persona_raw)

    detected_persona = sanitize_persona_label(persona_result.get("persona"))
    persona_confidence = safe_float(persona_result.get("confidence"), 0.0)
    use_acknowledgment = bool(persona_result.get("use_acknowledgment", False))
    acknowledgment = sanitize_acknowledgment(
        detected_persona,
        use_acknowledgment,
        persona_result.get("acknowledgment", "")
    )
    persona_reason = (persona_result.get("reason") or "").strip()

    # Stabilize persona so the role does not flip too easily across the session
    if previous_persona["persona"] != "unknown" and persona_confidence < 0.65:
        detected_persona = previous_persona["persona"]
        persona_confidence = previous_persona["confidence"]
        use_acknowledgment = previous_persona["use_acknowledgment"]
        acknowledgment = previous_persona["acknowledgment"]
        persona_reason = f"Kept previous persona due to low confidence. {persona_reason}"

    # Only use acknowledgment when persona is newly detected with strong confidence
    first_time_ack = (
        previous_persona["persona"] == "unknown"
        and detected_persona != "unknown"
        and persona_confidence >= 0.80
    )

    # Also allow acknowledgment if persona has just changed to a new high-confidence role
    changed_persona_ack = (
        previous_persona["persona"] != detected_persona
        and previous_persona["persona"] != "unknown"
        and detected_persona != "unknown"
        and persona_confidence >= 0.85
    )

    acknowledgment_to_use = acknowledgment if (
        use_acknowledgment and (first_time_ack or changed_persona_ack)
    ) else ""

    persona_memory[session_id] = {
        "persona": detected_persona,
        "confidence": persona_confidence,
        "use_acknowledgment": use_acknowledgment,
        "acknowledgment": acknowledgment
    }

    save_to_csv(
        session_id,
        "Persona",
        f"persona={detected_persona}; confidence={persona_confidence}; ack={acknowledgment_to_use}; reason={persona_reason}",
        search_query=""
    )

    # --- STEP A2: ROUTE ---
    router_raw = router_chain.invoke({
        "context": history_string,
        "question": question,
        "user_lang": user_lang,
        "user_lang_name": user_lang_name,
        "persona": detected_persona,
        "persona_confidence": persona_confidence
    })

    router_decision = parse_router_json(router_raw)

    use_retrieval = bool(router_decision.get("use_retrieval", True))
    search_query = (router_decision.get("search_query") or "").strip()
    router_reason = (router_decision.get("reason") or "").strip()

    save_to_csv(
        session_id,
        "Router",
        f"use_retrieval={use_retrieval}; reason={router_reason}",
        search_query=search_query
    )

    # --- STEP A3: RETRIEVAL ---
    docs = []
    info_text = ""
    sources = []

    if use_retrieval:
        effective_query = search_query if search_query else question
        try:
            docs = retriever.invoke(effective_query)
        except Exception as e:
            save_to_csv(session_id, "System", f"Retriever error: {repr(e)}")

        # --- STEP A4: POST-RETRIEVAL FILTERING ---
        if docs:
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

            if selected_indices:
                filtered_docs = [docs[i] for i in selected_indices if isinstance(i, int) and 0 <= i < len(docs)]
                if filtered_docs:
                    original_count = len(docs)
                    docs = filtered_docs
                    save_to_csv(
                        session_id,
                        "System",
                        f"Filtered docs from {original_count} down to {len(docs)}"
                    )

        info_text = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get("source_url", "Unknown source") for doc in docs]))

    # --- STEP B: GENERATE RESPONSE ---
    ai_response_text = answer_chain.invoke({
        "context": history_string,
        "info": info_text,
        "question": question,
        "user_lang": user_lang,
        "user_lang_name": user_lang_name,
        "persona": detected_persona,
        "persona_confidence": persona_confidence,
        "acknowledgment_to_use": acknowledgment_to_use
    })

    if not isinstance(ai_response_text, str):
        ai_response_text = str(ai_response_text)

    # --- STEP C: FORMAT HTML CITATIONS ---
    cleaned_sources = [s for s in sources if s and s != "Unknown source"]
    if cleaned_sources:
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
    save_to_csv(session_id, "Assistant", final_display_answer, search_query=search_query)

    conversation_memory[session_id].append(f"User: {question}")
    conversation_memory[session_id].append(f"Assistant: {ai_response_text}")

    if len(conversation_memory[session_id]) > MAX_MEMORY_LINES:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_LINES:]

    # --- STEP E: SEND RESPONSE ---
    return jsonify({
        "answer": final_display_answer,
        "raw_text": ai_response_text,
        "sources": cleaned_sources,
        "session_id": session_id,
        "persona": {
            "label": detected_persona,
            "confidence": persona_confidence,
            "acknowledgment_used": acknowledgment_to_use,
            "reason": persona_reason
        },
        "routing": {
            "use_retrieval": use_retrieval,
            "search_query": search_query,
            "reason": router_reason
        }
    })


# ----------------------------
# 8) RUN SERVER
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
