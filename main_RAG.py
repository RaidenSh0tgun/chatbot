# main_rag_only.py

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
from datetime import datetime


# ----------------------------
# 1) LOGGING
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


def save_retrieval_log(session_id: str, question: str, search_query: str, docs) -> None:
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


# ----------------------------
# 2) FLASK & MEMORY
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/chat/*": {"origins": "*"}})

conversation_memory = {}
MAX_MEMORY_LINES = 10


# ----------------------------
# 3) VECTOR DB
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
# 4) BM25 INDEX
# ----------------------------
def tokenize_for_bm25(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    return text.split()


print("Building BM25 index...")

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
# 5) LLM
# ----------------------------
model = OllamaLLM(model="qwen2.5")


# ----------------------------
# 6) LANGUAGE DETECTION ONLY
# ----------------------------
language_template = """
You are a language detection module for a Rutgers SPAA chatbot.

Conversation History:
{context}

Current User Question:
{question}

Task:
Detect the main language of the user's question.

Allowed language codes:
en, es, fr, de, it, pt, zh, zh-cn, zh-tw, ja, ko, ru, ar, hi

Return ONLY valid JSON with exactly these keys:
- language: string
- language_confidence: number
- reason: string

JSON:
"""

language_prompt = ChatPromptTemplate.from_template(language_template)
language_chain = language_prompt | model

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


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_language_json(text) -> dict:
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
        "reason": "Language output not parseable."
    }


# ----------------------------
# 7) ROUTER PROMPT — RAG ONLY
# ----------------------------
router_template = """
You are a routing module for a chatbot of Rutgers School of Public Affairs and Administration (SPAA).

User language: {user_lang_name} ({user_lang})

Goal:
Decide whether you must consult the knowledge base to answer accurately.
If needed, generate an effective search query grounded in the user's question AND the conversation history.

Use retrieval when:
- The user asks for school-specific facts.
- The user asks about programs, admissions, deadlines, tuition, fees, offices, contacts, policies, forms, courses, faculty, staff, procedures, locations, or student resources.
- The user asks for URLs, official details, step-by-step instructions, or factual claims about SPAA.
- The user asks a follow-up question about previously mentioned SPAA-specific information.
- The user asks to confirm, update, expand, correct, or provide details about a previous answer.

Important:
- The vector database content is primarily in English.
- If the user language is not English, translate the search query into concise English keywords for retrieval.
- Conversation history is useful for context, but it should NOT replace retrieval for SPAA-specific facts.
- If the user asks about people, titles, contacts, programs, requirements, deadlines, offices, policies, tuition, admissions, forms, or procedures, use_retrieval must be true.
- Do not set use_retrieval=false only because the information was mentioned in a previous response.
- Do NOT use user persona, background, or occupation. This is a RAG-only chatbot.
- Do NOT personalize the search query based on user identity.
- Do NOT use overly broad words in search_query such as: "SPAA", "Rutgers", "University", "school".

Conversation History:
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
- If use_retrieval is true, search_query MUST be a concise English retrieval phrase.

JSON:
"""

router_prompt = ChatPromptTemplate.from_template(router_template)
router_chain = router_prompt | model


def parse_router_json(text) -> dict:
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
# 8) HYBRID RETRIEVAL
# ----------------------------
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

    bm25_results = [
        bm25_docs[i]
        for i in top_bm25_indices
        if bm25_scores[i] > 0
    ]

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

    # 4. Metadata boost
    for item in fused.values():
        item["score"] += metadata_boost_score(item["doc"], query)

    ranked = sorted(
        fused.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [item["doc"] for item in ranked[:k_final]]


# ----------------------------
# 9) ANSWER PROMPT — RAG ONLY
# ----------------------------
answer_template = """
Your name is SPAA-rkly. You are a helpful assistant for the School of Public Affairs and Administration (SPAA) at Rutgers University-Newark.

User language: {user_lang_name} ({user_lang}).

### INSTRUCTIONS:
- Do NOT introduce yourself or state your name in your response. 
- Do NOT say "Hello" or "Hi" unless the user is specifically greeting you for the first time.
- Answer in 1-5 sentences unless requested otherwise.
- Ground your answer ONLY in the Related Information provided.

Conversation History:
{context}

Question:
{question}

Related Information:
{info}

Instructions:
- If Related Information is provided, ground your answer in it. Do not invent SPAA-specific facts that are not supported by it.
- If Related Information is empty, answer using general knowledge and the Conversation History only, but do not fabricate SPAA-specific facts (names, dates, requirements, contacts). If the question requires SPAA-specific facts and you lack them, say you don't know and suggest what to ask for.
"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain = answer_prompt | model


# ----------------------------
# 10) CHAT ENDPOINT
# ----------------------------
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not question or not session_id:
        return jsonify({"error": "Missing question or session_id"}), 400

    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    save_to_csv(session_id, "User", question, search_query="")

    history_string = "\n".join(conversation_memory[session_id]).strip()

    # STEP 1: Language detection only
    language_raw = language_chain.invoke({
        "context": history_string,
        "question": question
    })

    language_result = parse_language_json(language_raw)

    user_lang = normalize_lang(language_result.get("language", "en"))
    user_lang_confidence = safe_float(language_result.get("language_confidence"), 0.0)
    user_lang_name = lang_display(user_lang)
    language_reason = (language_result.get("reason") or "").strip()

    save_to_csv(
        session_id,
        "Language",
        f"language={user_lang}; confidence={user_lang_confidence}; reason={language_reason}",
        search_query=""
    )

    # STEP 2: Routing
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

    save_to_csv(
        session_id,
        "Router",
        f"use_retrieval={use_retrieval}; reason={router_reason}",
        search_query=search_query
    )

    # STEP 3: Retrieval
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

            save_retrieval_log(
                session_id=session_id,
                question=question,
                search_query=effective_query,
                docs=docs
            )

        except Exception as e:
            save_to_csv(session_id, "System", f"Retriever error: {repr(e)}")

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

        sources = list(set([
            doc.metadata.get("source_url", "Unknown source")
            for doc in docs
        ]))

    # STEP 4: Generate answer
    ai_response_text = answer_chain.invoke({
        "context": history_string,
        "info": info_text,
        "question": question,
        "user_lang": user_lang,
        "user_lang_name": user_lang_name
    })

    if not isinstance(ai_response_text, str):
        ai_response_text = str(ai_response_text)

    cleaned_sources = sorted(set([
        s for s in sources
        if s and s != "Unknown source"
    ]))

    final_display_answer = ai_response_text

    # STEP 5: Logging and memory
    save_to_csv(
        session_id,
        "Assistant",
        final_display_answer,
        search_query=search_query
    )

    conversation_memory[session_id].append(f"User: {question}")
    conversation_memory[session_id].append(f"Assistant: {ai_response_text}")

    if len(conversation_memory[session_id]) > MAX_MEMORY_LINES:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_LINES:]

    # STEP 6: Return response
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
        "routing": {
            "use_retrieval": use_retrieval,
            "search_query": search_query,
            "reason": router_reason
        },
        "version": "rag_only"
    })


# ----------------------------
# 11) HEALTH CHECK
# ----------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "version": "rag_only"
    }), 200


# ----------------------------
# 12) RUN SERVER
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)