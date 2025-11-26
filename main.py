from flask import Flask, request, jsonify
from flask_cors import CORS # NEW IMPORT
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever 

app = Flask(__name__)

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
    # 1. Get the question from the incoming JSON request
    try:
        data = request.get_json()
        question = data.get("question")
        context = data.get("context", "") 
    except:
        # 400 Bad Request error
        return jsonify({"error": "Invalid JSON or missing 'question'"}), 400

    if not question:
        # 400 Bad Request error
        return jsonify({"error": "Missing question parameter"}), 400
    
    # 2. RAG Logic (as in your original main.py)
    info = retriever.invoke(question) 
    
    # 3. Invoke the LangChain
    result = chain.invoke({"context": context, "info": info, "question": question})
    
    # 4. Return the result as JSON
    return jsonify({"answer": result})

if __name__ == '__main__':
    # Run the server on a specific local port (e.g., 5000)
    # The debug=True setting is helpful during development.
    app.run(host='0.0.0.0', port=5000, debug=True)