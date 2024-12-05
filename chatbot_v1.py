# -*- coding: utf-8 -*-
"""chatbot_v1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tXsADFhH5YhWzbO1TL_dk-WOMPNh3zvf
"""

import os
from groq import Groq
from flask import Flask, request, jsonify
import textwrap

os.environ["GROQ_API_KEY"] = "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"

# Initialize Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Initialize Flask app
app = Flask(__name__)

def process_user_input(user_input):
    # Create chat completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion

def format_response(response):
    # Format the response for better readability
    response_txt = response.choices[0].message.content
    formatted_response = []
    for chunk in response_txt.split("\n"):
        if not chunk:
            formatted_response.append("")
            continue
        formatted_response.append("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))
    return "\n".join(formatted_response)

@app.route('/chat', methods=['POST'])
def chat():
    # Get user input from POST request
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Process the input and get response
    try:
        response = process_user_input(user_input)
        formatted_response = format_response(response)
        return jsonify({"response": formatted_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)