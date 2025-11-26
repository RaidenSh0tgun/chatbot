from flask import Flask, request, jsonify
from flask_cors import CORS # NEW IMPORT
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever 
import csv
import os
from datetime import datetime

# Create folder if it doesn't exist
os.makedirs("conversation", exist_ok=True)

def save_to_csv(session_id, sender, message):
    filename = f"conversation/{session_id}.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header only once
        if not file_exists:
            writer.writerow(["timestamp", "session_id", "sender", "message"])

        writer.writerow([
            datetime.now().isoformat(),
            session_id,
            sender,
            message
        ])

app = Flask(__name__)

# NEW - store memory for each session
conversation_memory = {}  # session_id → list of messages

# --- START OF CORS CONFIGURATION ---
# This allows browsers from any origin (your website) to send requests to the /chat endpoint.
CORS(app, resources={r"/chat/*": {"origins": "*"}}) 
# --- END OF CORS CONFIGURATION ---

# --- Load components ONCE at startup ---
# LLM and prompt template from original main.py
model = OllamaLLM(model="llama3.2")

template = """
Your name is Friday. You are a helpful assistent of School of Public Affairs and Administration (SPAA) at Rutgers University-Newark. 

Your duty is to answer questions related to the School.

Always answer in 1-3 sentences unless the user explicitly requests more detail.
Do not provide long explanations, lists, or paragraphs unless asked.
Be clear, direct, and to the point.

Here is the conversation history: {context}

Here are related information: {info}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
# --- End of startup components ---

@app.route('/chat', methods=['POST'])
def chat_endpoint():

    data = request.get_json() or {}

    question = data.get("question")
    session_id = data.get("session_id")  # NEW

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400
    if not session_id:
        return jsonify({"error": "Missing 'session_id'"}), 400

    # 🧠 Initialize memory for new session
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    # Save user message to CSV
    save_to_csv(session_id, "User", question)
    
    # 🧠 Convert memory list into a history string
    context = "\n".join(conversation_memory[session_id])

    # 🔍 RAG retrieval
    info = retriever.invoke(question)

    # 🤖 LLM call
    result = chain.invoke({
        "context": context,
        "info": info,
        "question": question
    })

    # Save assistant reply
    save_to_csv(session_id, "Assistant", result)

    # 🧠 Save memory
    conversation_memory[session_id].append(f"User: {question}")
    conversation_memory[session_id].append(f"Assistant: {result}")

    # Return response
    return jsonify({
        "answer": result,
        "session_id": session_id  # returned for frontend continuity
    })

if __name__ == '__main__':
    # Run the server on a specific local port (e.g., 5000)
    # The debug=True setting is helpful during development.
    app.run(host='0.0.0.0', port=5000, debug=True)