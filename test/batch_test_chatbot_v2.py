"""
Batch test script for SPAA chatbot.

What it does:
1. Reads questions from QA_test.xlsx
2. Sends each question to the running chatbot endpoint
3. Saves chatbot responses, response time, and error messages to an Excel file

Before running:
- Start your chatbot backend first, for example:
    python main_revised_fast.py
- Make sure QA_test.xlsx is in the same folder as this script, or update INPUT_FILE.

Expected Excel input:
- A column named "question" is preferred.
- If there is no "question" column, the script will use the first column as questions.
"""

import time
import uuid
from pathlib import Path

import pandas as pd
import requests


# =========================
# Config
# =========================
INPUT_FILE = "QA_test.xlsx"
OUTPUT_FILE = "QA_test_with_chatbot_responses.xlsx"

# Change this if your Flask route is different.
# Common options may be:
#   http://127.0.0.1:5000/chat
#   http://localhost:5000/chat
#   http://127.0.0.1:8080/chat
CHATBOT_URL = "http://127.0.0.1:5000/chat"

# If your backend expects a different JSON key, revise build_payload().
TIMEOUT_SECONDS = 120
DELAY_BETWEEN_REQUESTS = 0.5


# =========================
# Helper functions
# =========================
def build_payload(question: str, session_id: str) -> dict:
    """
    Revise this function if main_revised_fast.py expects different field names.

    Common Flask chatbot payloads include:
        {"message": question, "session_id": session_id}
    or:
        {"question": question, "session_id": session_id}
    """
    return {
        "message": question,
        "session_id": session_id,
    }


def extract_response(response_json: dict) -> str:
    """
    Extract chatbot answer from common response formats.
    Revise this if your backend returns a different key.
    """
    possible_keys = ["response", "answer", "reply", "message", "text"]

    for key in possible_keys:
        if key in response_json:
            return str(response_json[key])

    # Fallback: save the whole JSON if no known key exists.
    return str(response_json)


def find_question_column(df: pd.DataFrame) -> str:
    """Find the question column. Prefer a column named 'question'."""
    normalized = {str(col).strip().lower(): col for col in df.columns}

    for candidate in ["question", "questions", "query", "prompt"]:
        if candidate in normalized:
            return normalized[candidate]

    return df.columns[0]


def test_one_question(question: str, session_id: str) -> tuple[str, float, str]:
    """
    Send one question to the chatbot.

    Returns:
        chatbot_response, response_time_seconds, error_message
    """
    start_time = time.time()

    try:
        payload = build_payload(question, session_id)
        r = requests.post(CHATBOT_URL, json=payload, timeout=TIMEOUT_SECONDS)
        elapsed = round(time.time() - start_time, 2)

        r.raise_for_status()
        data = r.json()
        answer = extract_response(data)
        return answer, elapsed, ""

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        return "", elapsed, str(e)


# =========================
# Main workflow
# =========================
def main() -> None:
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Put it in the same folder as this script, "
            f"or update INPUT_FILE in the script."
        )

    df = pd.read_excel(input_path)

    if df.empty:
        raise ValueError(f"{INPUT_FILE} is empty.")

    question_col = find_question_column(df)

    responses = []
    response_times = []
    errors = []
    session_ids = []

    total = len(df)

    for i, row in df.iterrows():
        question = str(row[question_col]).strip()
        session_id = f"batch_test_{uuid.uuid4().hex[:12]}"

        if not question or question.lower() == "nan":
            responses.append("")
            response_times.append(0)
            errors.append("Empty question")
            session_ids.append(session_id)
            continue

        print(f"[{i + 1}/{total}] Testing: {question[:100]}")

        answer, elapsed, error = test_one_question(question, session_id)

        responses.append(answer)
        response_times.append(elapsed)
        errors.append(error)
        session_ids.append(session_id)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    df["chatbot_response"] = responses
    df["response_time_seconds"] = response_times
    df["error"] = errors
    df["test_session_id"] = session_ids

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nDone. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
