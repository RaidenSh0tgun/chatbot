# main_persona_only.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import csv
import os
import json
import re
import random
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


# ----------------------------
# 2) FLASK & MEMORY
# ----------------------------
app = Flask(__name__)
CORS(app, resources={r"/chat/*": {"origins": "*"}})

conversation_memory = {}
persona_memory = {}

MAX_MEMORY_LINES = 10


# ----------------------------
# 3) LLM
# ----------------------------
model = OllamaLLM(model="qwen2.5")


# ----------------------------
# 4) LANGUAGE + PERSONA ANALYSIS
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
# 5) ANSWER PROMPT — PERSONA ONLY
# ----------------------------
answer_template = """
Your name is SPAA-rkly. You are a helpful assistant for the School of Public Affairs and Administration (SPAA) at Rutgers University-Newark.

User language: {user_lang_name} ({user_lang}).
Detected persona: {persona}.
Persona confidence: {persona_confidence}.
Acknowledgment: {acknowledgment_to_use}.

### GENERAL INSTRUCTIONS

- Respond like a friendly advisor.
- Use a warm, professional, and welcoming tone.
- When referring to SPAA, use first-person plural language, such as "our program" or "our students."
- Do NOT introduce yourself or state your name.
- Do NOT say "Hello" or "Hi" unless the user is specifically greeting you.
- If the user language is not English, respond in {user_lang_name}.
- Keep proper nouns and program names in English when appropriate.
- Tailor the response to the user's likely background when relevant.
- If Acknowledgment is not empty, include it once as the first sentence.
- Do not repeat, paraphrase, or add additional gratitude for the same reason later.
- Do not explicitly mention persona classification.

Persona guidance:
- veteran: emphasize transition to civilian public service, leadership, public-sector careers, and support needs.
- government_employee: emphasize professional advancement, leadership, public management, budgeting, policy analysis, and administrative growth.
- nonprofit_professional: emphasize nonprofit leadership, governance, fundraising, mission-driven management, and cross-sector career growth.
- law_enforcement: emphasize public safety leadership, supervision, ethics, policy, administration, budgeting, and local government leadership.
- current_student: emphasize academic planning, advising, registration, graduation, and student support, but avoid inventing office details.
- international_user: emphasize international admissions considerations, visa/I-20 questions, academic fit, and campus adjustment, but avoid inventing immigration details.
- faculty_or_staff: emphasize operational or academic information, but avoid inventing internal procedures.
- general_public / unknown: give general, non-personalized guidance.

Default user framing:
- If the detected persona is NOT current_student or faculty_or_staff, treat the user as a prospective student.
- Emphasize program value, career outcomes, and opportunities.
- Use an informative and welcoming tone appropriate for prospective students.

Question:
{question}

Grounding limitations:
- Do NOT claim to know exact SPAA-specific facts unless they are already provided in the conversation.
- You may provide general guidance about public administration, graduate education, public service careers, and program fit.
- When specific SPAA facts are needed, clearly state that the user should verify details on the official SPAA website or with the relevant SPAA office.
- Do not invent SPAA-specific names, dates, requirements, deadlines, tuition, policies, contacts, course numbers, or URLs.
- If the user asks for exact SPAA-specific information, provide general guidance and recommend checking the official SPAA website or contacting SPAA directly.
- Do not create fake citations.

Response rules:
- Answer in 1-5 sentences unless requested otherwise.
- Keep the response concise but helpful.
- Use Markdown formatting.

Answer:
"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain = answer_prompt | model


# ----------------------------
# 6) CHAT ENDPOINT
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

    # ----------------------------
    # STEP 1: LANGUAGE + PERSONA ANALYSIS
    # ----------------------------
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

    # Stabilize persona across session
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

    # ----------------------------
    # STEP 2: GENERATE PERSONA-ONLY RESPONSE
    # ----------------------------
    ai_response_text = answer_chain.invoke({
        "context": history_string,
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

    final_display_answer = ai_response_text

    # ----------------------------
    # STEP 3: LOGGING & MEMORY UPDATE
    # ----------------------------
    save_to_csv(
        session_id,
        "Assistant",
        final_display_answer,
        search_query=""
    )

    conversation_memory[session_id].append(f"User: {question}")
    conversation_memory[session_id].append(f"Assistant: {ai_response_text}")

    if len(conversation_memory[session_id]) > MAX_MEMORY_LINES:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_LINES:]

    # ----------------------------
    # STEP 4: RETURN RESPONSE
    # ----------------------------
    return jsonify({
        "answer": final_display_answer,
        "raw_text": ai_response_text,
        "sources": [],
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
            "use_retrieval": False,
            "search_query": "",
            "reason": "Persona-only version: retrieval disabled."
        },
        "version": "persona_only"
    })


# ----------------------------
# 7) HEALTH CHECK
# ----------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "version": "persona_only"
    }), 200


# ----------------------------
# 8) RUN SERVER
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)