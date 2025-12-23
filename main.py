from flask import Flask, request, jsonify
from flask_cors import CORS 
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma  # NEW MODERN IMPORT
from langchain_ollama import OllamaEmbeddings # NEW MODERN IMPORT
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

# Store memory for each session
conversation_memory = {}  # session_id → list of messages

CORS(app, resources={r"/chat/*": {"origins": "*"}}) 

# --- START OF NEW VECTOR DATABASE CONFIGURATION ---
# These components are loaded ONCE when the server starts
print("Connecting to Vector Database...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
# --- END OF VECTOR CONFIGURATION ---

# LLM and prompt template
model = OllamaLLM(model="gemma3:4b") # Updated to gemma3:4b based on your tests

template = """
Your name is Friday. You are a helpful assistant of School of Public Affairs and Administration (SPAA) at Rutgers University-Newark. 

Your duty is to answer questions related to the School.
Always answer in 1-5 sentences unless the user explicitly requests more detail.
Be clear, direct, and only use the provided information.

Here is the conversation history: {context}

Here is related information: {info}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json() or {}

    question = data.get("question")
    session_id = data.get("session_id") 

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400
    if not session_id:
        return jsonify({"error": "Missing 'session_id'"}), 400

    # Initialize memory for new session
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    # Save user message to CSV
    save_to_csv(session_id, "User", question)
    
    # Convert memory list into a history string
    history_context = "\n".join(conversation_memory[session_id])

    # 🔍 RAG retrieval - Get relevant text from chunks
    docs = retriever.invoke(question)
    info_text = "\n\n".join([doc.page_content for doc in docs])

    # 🤖 LLM call
    result = chain.invoke({
        "context": history_context,
        "info": info_text,
        "question": question
    })

    # Save assistant reply
    save_to_csv(session_id, "Assistant", result)

    # Save to memory (Keep only last 10 messages to prevent context bloat)
    conversation_memory[session_id].append(f"User: {question}")
    conversation_memory[session_id].append(f"Assistant: {result}")
    if len(conversation_memory[session_id]) > 10:
        conversation_memory[session_id] = conversation_memory[session_id][-10:]

    # Return response
    return jsonify({
        "answer": result,
        "session_id": session_id 
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)