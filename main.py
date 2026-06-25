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
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
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
# 3.1) BM25 INDEX
# ----------------------------

def tokenize_for_bm25(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    return text.split()


print("Building BM25 index...")

# Pull all Chroma documents into memory for BM25.
# Adjust limit if your database grows much larger.
chroma_data = vector_db.get(
    include=["documents", "metadatas"],
    limit=10000
)

bm25_docs = []
bm25_corpus = []

for content, metadata in zip(chroma_data["documents"], chroma_data["metadatas"]):
    metadata = metadata or {}

    bm25_text = f"""
    {metadata.get("title", "")}
    {metadata.get("retrieval_phrases", "")}
    {metadata.get("contextual_summary", "")}
    {metadata.get("keywords", "")}
    {content}
    """

    bm25_docs.append(
        Document(
            page_content=content,
            metadata=metadata
        )
    )
    bm25_corpus.append(tokenize_for_bm25(bm25_text))

bm25_index = BM25Okapi(bm25_corpus)

print(f"BM25 index built with {len(bm25_docs)} documents.")

# ----------------------------
# 4) LLM
# ----------------------------
# Keep the same model for language, persona, router, filter, and answer for simplicity.
model = OllamaLLM(model="qwen3")


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
- Do not include any keys beyond the three keys above.
- If use_retrieval is false, set search_query to "".
- If use_retrieval is true, search_query MUST be a concise English retrieval phrase or a small set of high-signal phrases. DO NOT use overly broad or low-information words in search_query such as: "SPAA", "Rutgers", "University", "school". 

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

Privacy and Contact Rules:

- If asked about a specific alumnus/alumna or student, politely state that you cannot provide personal alumni or student information unless it is publicly published in official SPAA content.
- You may discuss aggregate alumni outcomes, representative employers, or publicly published success stories only when supported by retrieved official content.
- Do NOT provide alumni or student names, emails, phone numbers, LinkedIn profiles, or other contact information unless the user explicitly requests contact information and the information is publicly listed in official retrieved content.
- When answering general questions about programs, admissions, curriculum, careers, or student experience, do NOT volunteer alumni or student contact information unless directly relevant to the user's request.
- If retrieved documents contain alumni or student contact information that is not necessary to answer the question, ignore it.
- Prefer official SPAA office contacts, faculty contacts, or program contacts over individual student or alumni contacts unless the user explicitly asks for peer/student/alumni connections.

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

def phrase_overlap_score(query: str, phrases: str) -> float:
        query = (query or "").lower()
        phrases = (phrases or "").lower()

        if not query or not phrases:
            return 0.0

        score = 0.0

        phrase_list = [
            p.strip()
            for p in re.split(r"\||,|;", phrases)
            if p.strip()
        ]

        for phrase in phrase_list:
            if phrase in query:
                score += 2.0

            phrase_terms = phrase.split()
            matched_terms = sum(1 for term in phrase_terms if term in query)

            if phrase_terms:
                score += matched_terms / len(phrase_terms)

        return score


def metadata_boost_score(doc, query: str) -> float:
        q = (query or "").lower()

        title = doc.metadata.get("title", "").lower()
        retrieval_phrases = doc.metadata.get("retrieval_phrases", "").lower()
        contextual_summary = doc.metadata.get("contextual_summary", "").lower()

        score = 0.0

        if title and title in q:
            score += 2.0

        for term in q.split():
            if term in title:
                score += 0.4
            if term in contextual_summary:
                score += 0.2

        score += phrase_overlap_score(q, retrieval_phrases) * 1.5

        return score


def hybrid_retrieve(query: str, k_final: int = 8, k_chroma: int = 20, k_bm25: int = 20):
    """
    Hybrid retrieval:
    - Chroma captures semantic similarity.
    - BM25 captures exact keywords, names, titles, acronyms, and role phrases.
    - Reciprocal Rank Fusion combines both.
    """

    # 1. Chroma semantic retrieval
    chroma_results = vector_db.similarity_search(query, k=k_chroma)

    # 2. BM25 keyword retrieval
    tokenized_query = tokenize_for_bm25(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k_bm25]

    bm25_results = [bm25_docs[i] for i in top_bm25_indices if bm25_scores[i] > 0]

    # 3. Reciprocal Rank Fusion
    fused = {}

    def doc_key(doc):
        return (
            doc.metadata.get("source_url", ""),
            doc.metadata.get("title", ""),
            doc.page_content[:120]
        )

    for rank, doc in enumerate(chroma_results, start=1):
        key = doc_key(doc)
        if key not in fused:
            fused[key] = {"doc": doc, "score": 0.0}
        fused[key]["score"] += 0.70 * (1 / rank)

    for rank, doc in enumerate(bm25_results, start=1):
        key = doc_key(doc)
        if key not in fused:
            fused[key] = {"doc": doc, "score": 0.0}
        fused[key]["score"] += 0.30 * (1 / rank)

    # 4. Add your existing metadata boost
    for item in fused.values():
        item["score"] += metadata_boost_score(item["doc"], query)

    ranked = sorted(
        fused.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [item["doc"] for item in ranked[:k_final]]

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
                contextual_summary = doc.metadata.get("contextual_summary", "")
                preview_text = f"{contextual_summary} {doc.page_content}"
                preview = preview_text[:500].replace("\n", " ")

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
    

    docs = []
    info_text = ""
    sources = []

    if use_retrieval:
        effective_query = search_query if search_query else question
        try:
            docs = hybrid_retrieve(
                effective_query,
                k_final=8,
                k_chroma=20,
                k_bm25=20
            )

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
                phrases = doc.metadata.get("retrieval_phrases", "")
                contextual_summary = doc.metadata.get("contextual_summary", "")

                info_blocks.append(
                    f"[S{i}]\n"
                    f"TITLE: {title}\n"
                    f"RETRIEVAL_PHRASES: {phrases}\n"
                    f"CONTEXTUAL_SUMMARY: {contextual_summary}\n"
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