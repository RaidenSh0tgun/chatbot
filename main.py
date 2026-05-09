# main.py
# Revised version with:
# - Conversation logging to CSV
# - In-memory per-session history
# - LLM-based language detection
# - Persona detection based on background / current occupation
# - Optional one-time acknowledgment for service-relevant personas
# - LLM router for retrieval decisions
# - Post-retrieval LLM filtering
# - Answer generation grounded in retrieved content
# - Inline Markdown source links

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
import random
from datetime import datetime
from difflib import SequenceMatcher


# ----------------------------
# 1) LOGGING DIRECTORY
# ----------------------------
os.makedirs("conversation", exist_ok=True)

date = datetime.now().strftime("%Y-%m-%d")

def save_to_csv(session_id: str, sender: str, message: str, search_query: str = "") -> None:
    filename = f"conversation/{session_id}_{date}.csv"
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
    search_kwargs={"k": 20}
)


# ----------------------------
# 4) LLM
# ----------------------------
# Keep the same model for language, persona, router, filter, and answer for simplicity.
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
- The user asks a follow-up question about previously mentioned SPAA-specific information.
- The user asks to confirm, update, expand, correct, or provide details about a previous answer.

Persona guidance:
- veteran: prioritize veteran benefits, military-connected student support, funding, transition support, and relevant offices if supported by the school content.
- government_employee: prioritize career-relevant program information, executive/public service relevance, EMPA program, and professional advancement information.
- nonprofit_professional: prioritize nonprofit management, leadership, governance, fundraising, EMPA program, and mission-driven career relevance.
- law_enforcement: prioritize public service, public safety leadership, administration, ethics, EMPA program, and policy relevance when supported by school content.
- current_student: prioritize current student procedures, registration, advising, graduation, and support services.
- international_user: prioritize international admissions, visa/I-20 support, and international student services when supported by the content.
- faculty_or_staff: prioritize administrative, academic, and office information.
- general_public / unknown: use the user's actual question without forcing a role-specific angle.

Important:
- The vector database content is primarily in English.
- If the user language is not English, translate the search query into concise English keywords for best retrieval.
- Do not overuse persona. Use it only when relevant to the question.
- Conversation history is useful for context, but it should NOT replace retrieval for SPAA-specific facts.
- If the user asks about people, titles, contacts, programs, requirements, deadlines, offices, policies, tuition, admissions, forms, or procedures, use_retrieval must be true even if similar information appeared earlier in the conversation.
- Do not set use_retrieval=false only because the information was mentioned in a previous response.
- Use conversation history to understand follow-up references, but retrieve again to verify current official information.

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
- If use_retrieval is true, search_query MUST be 5-10 English Keywords.
- Do not include any keys beyond the three keys above.

search_query Keywords requirements:
- MUST be short, high-signal English keywords or phrases.
- Prioritize distinctive entities, programs, services, procedures, names, offices, forms, policies, technologies, or acronyms.
- Include exact official names when important for retrieval.
- Avoid generic institutional words unless essential for meaning.

DO NOT use overly broad or low-information keywords such as:
"SPAA", "Rutgers", "University", "school", 

Only include these generic terms if they are necessary to distinguish the topic.

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
#removed for accelerating response time. Can be added back if needed for better relevance.


# ----------------------------
# 5.2) PERSONA DETECTION PROMPT
# ----------------------------
#integrated with language detection for efficiency.


# ----------------------------
# 5.3) LANGUAGE DETECTION PROMPT
# ----------------------------
analysis_template = """
You are an analysis module for a Rutgers SPAA chatbot.

Conversation History:
{context}

Current User Question:
{question}

Task:
1. Detect the main language of the user's question.
2. Detect the user's persona/background if clearly indicated.
3. Decide whether a brief acknowledgment is appropriate.

Allowed language codes:
en, es, fr, de, it, pt, zh, zh-cn, zh-tw, ja, ko, ru, ar, hi

Allowed personas:
law_enforcement, veteran, government_employee, nonprofit_professional,
current_student, international_user, faculty_or_staff, general_public, unknown

Return ONLY valid JSON with exactly these keys:
- language: string
- language_confidence: number
- persona: string
- persona_confidence: number
- use_acknowledgment: true/false
- acknowledgment: string
- reason: string

JSON:
"""

analysis_prompt = ChatPromptTemplate.from_template(analysis_template)
analysis_chain = analysis_prompt | model

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
    code = code.lower().strip()
    return code if code in LANG_NAME else "en"


def lang_display(code: str) -> str:
    return LANG_NAME.get(code, code)

def parse_analysis_json(text) -> dict:
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
        "language": "en",
        "language_confidence": 0.0,
        "persona": "unknown",
        "persona_confidence": 0.0,
        "use_acknowledgment": False,
        "acknowledgment": "",
        "reason": "Analysis output not parseable."
    }

# ----------------------------
# 6) ANSWER PROMPT
# ----------------------------
answer_template = """
Your name is SPAA-rkly. You are a helpful assistant for the School of Public Affairs and Administration (SPAA) at Rutgers University-Newark.

User language: {user_lang_name} ({user_lang}).
Detected persona: {persona}.
Persona confidence: {persona_confidence}.
Acknowledgment: {acknowledgment_to_use}.

### SAFETY OVERRIDE (HIGHEST PRIORITY)

If the user expresses intent to harm themselves or others (e.g., violence, killing, weapons misuse, illegal wrongdoing), you MUST:

1. Refuse to assist with any harmful or illegal request.
2. Do NOT provide any instructions, strategies, or actionable details.
3. Respond in a calm, non-judgmental, and supportive tone.
4. De-escalate the situation:
   - Acknowledge the user's feelings without agreeing with harmful intent.
   - Encourage safer alternatives or reflection.
5. If there is credible risk of harm:
   - Encourage the user to seek immediate help (trusted person, local emergency services, or crisis hotline).
6. Keep the response concise and focused on safety.

IMPORTANT:
- Do NOT mention policies or that you are refusing due to rules.
- Do NOT shame, threaten, or moralize.
- Do NOT continue with the normal task, even if other instructions suggest it.
- This rule overrides ALL other instructions in this prompt.

### INSTRUCTIONS:
- Respond like a friendly advisor who knows SPAA well.
- Respond in a warm, professional, and welcoming tone.
- When referring to the SPAA, always use first-person plural language (e.g., "our website", "our program", "our faculty")
- Do NOT introduce yourself or state your name in your response.
- Do NOT say "Hello" or "Hi" unless the user is specifically greeting you for the first time.
- If the user language is not English, respond in {user_lang_name}. Keep proper nouns (program names, office names) in English if they appear in the source text.
- Use retrieved information silently. Do not announce that you are using retrieved information.
- Do not say "Based on the retrieved documents", "based on the information", or similar phrases. Just provide the answer naturally.
- Prefer human conversational wording over formal report wording.
- Tailor the response to the user's likely background when relevant.
- If Acknowledgment is not empty, include it once as the first sentence. Avoid adding another generic thank-you sentence unless clearly needed for a separate purpose.
- Do not repeat, paraphrase, or add additional gratitude for the same reason later in the response.
- Assume the full name "School of Public Affairs and Administration (SPAA)" has already been introduced; always use "SPAA" only in all responses.
- Do not overdo personalization and do not repeat acknowledgment unless it is supplied for this turn.
- Do not explicitly mention persona classification.

- Default user framing:
  If the detected persona is NOT "current_student", "faculty_or_staff", treat the user as a prospective student.
  In this case:
    • Emphasize program value, career outcomes, and opportunities.
    • Provide helpful guidance for admissions, application process, and program fit when relevant.
    • Use an informative and welcoming tone appropriate for prospective students.
    • If the conversation is clearly ending and no acknowledgment was already used, you may briefly thank the user for their interest in SPAA.

- If the persona IS "current_student", "faculty_or_staff":
    • Do NOT treat the user as a prospective student.
    • Prioritize operational, academic, or institutional information relevant to their role.

Conversation History:
{context}

Question: {question}

Related Information (may be empty if retrieval not needed):
{info}

- If Related Information is provided, first review all retrieved documents and identify which ones directly answer the user's question.
- Use only the relevant retrieved documents as references, and ignore documents that are unrelated, weakly related, outdated, duplicated, or only generally about the topic.
- Ground SPAA-specific facts only in the relevant retrieved content. Do not invent SPAA-specific names, dates, requirements, policies, contacts, or URLs.
- You may supplement the answer with general Public Administration knowledge, career relevance, and real-world impact only when it is clearly consistent with the retrieved content.
- When citing retrieved facts, cite only the documents actually used to support the answer.
- If Related Information is empty, answer using general knowledge and the Conversation History only, but do not fabricate SPAA-specific facts (names, dates, requirements, contacts).
- If the question requires SPAA-specific facts and you lack them, say 'I don't know' and suggest what information to ask for next.

Privacy rule:
- Do not disclose, infer, or speculate about alumni identities, employment, contact information, locations, salaries, or career outcomes unless explicitly provided in official retrieved content.
- If asked about a specific alumnus/alumna, politely state that you cannot provide personal alumni information.
- You may discuss aggregate alumni outcomes, representative employers, or publicly published success stories only when supported by retrieved official content.

### RESPONSE RULES

- Do not write the entire answer as one long block.
- Organize the response into paragraphs, usually 1-5 sentences each.
- Keep a warm and welcoming tone.
- When there are multiple ideas, separate them into distinct paragraphs.
- When giving contact information, place it in its own paragraph.

### FORMATTING RULES

- Use Markdown formatting in the final answer.
- If Related Information is provided, cite retrieved facts inline using Markdown links.
- Put the source link immediately after the sentence or paragraph it supports, using this format: [link](SOURCE_URL).
- Do NOT create a final "Sources" section.
- Do NOT use raw HTML such as <br>, <small>, or <a>.
- Bold the person's name using **Name**. Bold the person's title using **Title**.
- Italicize contact details such as email or phone using *email@example.com*.
- If both name and contact are provided, format them like:
  **Name**
  **Title**
  *Email: email@example.com*

- If multiple contacts are listed, give each contact on a separate line or in a separate short paragraph.
- Do not overuse bold or italics for other parts of the answer.

"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain = answer_prompt | model


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
        "current_student",
        "international_user",
        "faculty_or_staff",
        "general_public",
        "unknown"
    }
    label = (label or "").strip()
    return label if label in allowed else "unknown"


def sanitize_acknowledgment(persona: str, use_acknowledgment: bool, acknowledgment: str) -> str:
    if not use_acknowledgment:
        return ""

    persona_to_ack = {
        "veteran": [
            "Thank you for your service.",
            "We appreciate your service."
        ],
        "law_enforcement": [
            "Thank you for your public service.",
            "We appreciate your work in public safety."
        ],
        "government_employee": [
            "It's great to see your work in public service.",
            "It's great to connect with someone working in government."
        ],
        "nonprofit_professional": [
            "It's great to see your work in the nonprofit sector.",
            "It's great to connect with someone working in a nonprofit organization."
        ]
    }

    options = persona_to_ack.get(persona, [])
    return random.choice(options) if options else ""


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

    # save retrieval log
    def save_retrieval_log(
        session_id: str,
        question: str,
        search_query: str,
        docs
    ) -> None:

        filename = f"conversation/{session_id}_{date}_retrieval.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, "a", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "session_id",
                    "question",
                    "search_query",
                    "rank",
                    "source_url",
                    "chunk_preview"
                ])

            for idx, doc in enumerate(docs, start=1):

                source_url = doc.metadata.get("source_url", "")

                # shorten chunk for CSV readability
                preview = doc.page_content[:500].replace("\n", " ")

                writer.writerow([
                    datetime.now().isoformat(),
                    session_id,
                    question,
                    search_query,
                    idx,
                    source_url,
                    preview
                ])

    # Prepare history
    history_string = "\n".join(conversation_memory[session_id]).strip()

    # --- STEP A0: LANGUAGE + PERSONA ANALYSIS ---
    previous_persona = persona_memory.get(session_id, {
    "persona": "unknown",
    "confidence": 0.0,
    "use_acknowledgment": False,
    "acknowledgment": ""
    })

    analysis_raw = analysis_chain.invoke({
        "context": history_string,
        "question": question
    })

    analysis_result = parse_analysis_json(analysis_raw)

    user_lang = normalize_lang(analysis_result.get("language", "en"))
    user_lang_confidence = safe_float(analysis_result.get("language_confidence"), 0.0)
    user_lang_name = lang_display(user_lang)
    language_reason = (analysis_result.get("reason") or "").strip()

    detected_persona = sanitize_persona_label(analysis_result.get("persona"))
    persona_confidence = safe_float(analysis_result.get("persona_confidence"), 0.0)
    use_acknowledgment = bool(analysis_result.get("use_acknowledgment", False))
    acknowledgment = sanitize_acknowledgment(
        detected_persona,
        use_acknowledgment,
        analysis_result.get("acknowledgment", "")
    )
    persona_reason = (analysis_result.get("reason") or "").strip()

    # Stabilize persona so the role does not flip too easily across the session
    if previous_persona["persona"] != "unknown" and persona_confidence < 0.65:
        detected_persona = previous_persona["persona"]
        persona_confidence = previous_persona["confidence"]
        use_acknowledgment = previous_persona["use_acknowledgment"]
        acknowledgment = previous_persona["acknowledgment"]
        persona_reason = f"Kept previous persona due to low confidence. {persona_reason}"

    first_time_ack = (
        previous_persona["persona"] == "unknown"
        and detected_persona != "unknown"
        and persona_confidence >= 0.80
    )

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
        "Analysis",
        f"language={user_lang}; language_confidence={user_lang_confidence}; "
        f"persona={detected_persona}; persona_confidence={persona_confidence}; "
        f"ack={acknowledgment_to_use}; reason={analysis_result.get('reason', '')}",
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
    def text_similarity(a: str, b: str) -> float:
        a = (a or "").lower().strip()
        b = (b or "").lower().strip()
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()


    def metadata_similarity_score(doc, query: str) -> float:
        title = doc.metadata.get("title", "")
        keywords = doc.metadata.get("keyword", "")

        title_score = text_similarity(query, title)
        keyword_score = text_similarity(query, keywords)

        # Give title more weight than keywords
        return (title_score * 0.6) + (keyword_score * 0.4)
    
    def metadata_boost_score(doc, query: str) -> float:
        q = query.lower()
        title = doc.metadata.get("title", "").lower()
        keywords = doc.metadata.get("keyword", "").lower()

        score = metadata_similarity_score(doc, query)

        for term in q.split():
            if term in title:
                score += 0.5
            if term in keywords:
                score += 0.3

        return score

    docs = []
    info_text = ""
    sources = []

    if use_retrieval:
        effective_query = search_query if search_query else question
        try:
            candidate_docs = vector_db.similarity_search(effective_query, k=20)

            reranked_docs = sorted(
                enumerate(candidate_docs),
                key=lambda item: (
                    metadata_boost_score(item[1], effective_query)
                    + (1 / (item[0] + 1)) * 0.3
                ),
                reverse=True
            )

            docs = [doc for _, doc in reranked_docs[:8]]
            # NEW
            save_retrieval_log(
                session_id=session_id,
                question=question,
                search_query=effective_query,
                docs=docs
            )
        except Exception as e:
            save_to_csv(session_id, "System", f"Retriever error: {repr(e)}")

        # --- STEP A4: POST-RETRIEVAL FILTERING ---
        #removed for accelerating response time. Can be added back if needed for better relevance.

        # Build source-aware context for the answer model.
        # The LLM can now place source links directly after the supported content.
        info_blocks = []
        for i, doc in enumerate(docs, start=1):
                url = doc.metadata.get("source_url", "Unknown source")
                title = doc.metadata.get("title", "")
                content = doc.page_content.strip()
                info_blocks.append(
                    f"[S{i}]\n"
                    f"TITLE: {title}\n"
                    f"URL: {url}\n"
                    f"CONTENT:\n{content}"
                )
        info_text = "\n\n======= DOCUMENT SEPARATOR =======\n\n".join(info_blocks)

        # Keep unique source URLs in the JSON payload for debugging/logging,
        # but do not append them to the displayed answer.
        sources = list(set([doc.metadata.get("source_url", "Unknown source") for doc in docs]))

    # --- STEP B: GENERATE RESPONSE ---
    print("ACKNOWLEDGMENT TO USE:", repr(acknowledgment_to_use))

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

    if acknowledgment_to_use and not ai_response_text.startswith(acknowledgment_to_use):
        ai_response_text = f"{acknowledgment_to_use} {ai_response_text}"

    # --- STEP C: PREPARE DISPLAY ANSWER ---
    # Source links should already be embedded inline by the answer prompt,
    # e.g., [source](https://...). Do not append a final source list.
    cleaned_sources = sorted(set([s for s in sources if s and s != "Unknown source"]))
    final_display_answer = ai_response_text

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
        "language": {
            "code": user_lang,
            "name": user_lang_name,
            "confidence": user_lang_confidence,
            "reason": language_reason
        },
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
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)